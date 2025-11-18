from contextlib import contextmanager
import gc
import glob
import logging
import pickle
import numpy as np

from model import CausalDiffusion
from utils.dataset import ShardingLMDBDataset, cycle
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from torchvision.io import write_video
from pipeline import CausalInferencePipeline, CausalDiffusionInferencePipeline, BidirectionalInferencePipeline, BidirectionalDiffusionInferencePipeline
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        assert self.global_rank % 8 == self.local_rank

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = self.global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + self.global_rank)

        if self.is_main_process and not self.disable_wandb:
            # wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                # entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )
        
        self.run_name = config.config_name

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.inference_only = getattr(config, "inference_only", False)

        if not self.inference_only:
            # Step 3: Initialize the dataloader
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True, drop_last=True)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=8)

            if dist.get_rank() == 0:
                print("DATASET SIZE %d" % len(dataset))
            self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        checkpoint_folders = glob.glob(os.path.join(self.output_path, "checkpoint_model_*"))
        if False:#len(checkpoint_folders) > 0:
            checkpoint_folders = sorted(checkpoint_folders, key=lambda x: int(x.split("_")[-1]))
            latest_folder = checkpoint_folders[-1]
            checkpoint_path = os.path.join(latest_folder, "model.pt")
            self.step = int(latest_folder.split("_")[-1])
            print(f"Resuming from checkpoint {checkpoint_path} at step {self.step}")
        else:
            checkpoint_path = getattr(config, "generator_ckpt", None)
            print("Starting training from scratch")

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if checkpoint_path is not None:
            print(f"Loading pretrained generator from {checkpoint_path}")
            if checkpoint_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(config.generator_ckpt, device="cpu")
                state_dict = {f"model.{k}" : v for k, v in state_dict.items()}
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
            # if "generator_ema" in state_dict and self.generator_ema is not None:
            #     self.generator_ema.load_state_dict(
            #         state_dict["generator_ema"], strict=True
            #     )
            init_ema = getattr(config, "init_ema", False)
            if init_ema and "generator_ema" in state_dict:
                state_dict = state_dict["generator_ema"]
            elif "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {rename_param(k): v for k, v in state_dict.items()}
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None

        inf_pipeline_type = (CausalInferencePipeline if self.causal else BidirectionalInferencePipeline) \
            if hasattr(config, 'denoising_step_list') else (CausalDiffusionInferencePipeline if self.causal else BidirectionalDiffusionInferencePipeline)

        self.val_pipeline = inf_pipeline_type(config, self.device, generator=self.model.generator, text_encoder=self.model.text_encoder, vae=self.model.vae)
        self.cpu_group = torch.distributed.new_group(list(range(self.world_size)), backend="gloo")
        if config.validation_prompts_file is not None:
            with open(config.validation_prompts_file, "r") as f:
                all_prompts = [line.strip() for line in f.readlines()]
            validate_first_n = config.validate_first_n if hasattr(config, 'validate_first_n') else len(all_prompts)
            all_prompts = all_prompts[:validate_first_n]
            self.val_prompts = all_prompts
        else:
            self.val_prompts = None
        
        if config.benchmark_prompts_file is not None:
            with open(config.benchmark_prompts_file, "r") as f:
                all_prompts = [line.strip() for line in f.readlines()]
            self.benchmark_prompts = all_prompts
            self.benchmark_samples = config.benchmark_samples if hasattr(config, 'benchmark_samples') else 1
            self.rename_prompts = (not config.disable_benchmark_rename) if hasattr(config, 'disable_benchmark_rename') else True
        else:
            self.benchmark_prompts = None
            self.benchmark_samples = 0

    def send_object(self, obj, dst: int) -> None:
        """Send the input object list to the destination rank."""
        """NOTE: `dst` is the local rank of the destination rank."""

        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert dst != self.global_rank, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank.")

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor([object_tensor.numel()],
                                   dtype=torch.long,
                                   device="cpu")

        # Send object size

        torch.distributed.send(size_tensor,
                               dst=dst,
                               group=self.cpu_group)

        # Send object
        torch.distributed.send(object_tensor,
                               dst=dst,
                               group=self.cpu_group)

        return None

    def recv_object(self, src: int):
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"

        assert src != self.global_rank, (
            "Invalid source rank. Source rank is the same as the current rank.")

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(size_tensor,
                                           src=src,
                                           group=self.cpu_group)

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu")

        rank_object = torch.distributed.recv(object_tensor,
                                             src=src,
                                             group=self.cpu_group)

        assert rank_object == rank_size, (
            "Received object sender rank does not match the size sender rank.")

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self, batch):
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["ode_latent"][:, -1].to(
                device=self.device, dtype=self.dtype)
        else:  # encode raw video to latent
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
        image_latent = clean_latent[:, 0:1, ]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent
        )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()
        if self.generator_ema is not None:
            self.generator_ema.update(self.model.generator)

        # Increment the step since we finished gradient update
        self.step += 1

         # Create EMA params (if not already created)
        if (self.step >= self.config.ema_start_step) and \
                (self.generator_ema is None) and (self.config.ema_weight > 0):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Step 4: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

        # Step 5. Create EMA params
        # TODO: Implement EMA

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        sampled_noise = torch.randn(
            [batch_size, self.model.num_training_frames, 16, 60, 104], device="cuda", dtype=self.dtype
        )
        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    @contextmanager
    def use_generator_ema(self):
        """Temporarily load EMA weights into the FSDP-wrapped generator."""
        if self.generator_ema is None:
            # EMA not active yet
            yield False
            return

        # Backup current (non-EMA) params on CPU
        backup = {}
        with FSDP.summon_full_params(self.model.generator, writeback=False):
            for n, p in self.model.generator.module.named_parameters():
                backup[n] = p.detach().clone().cpu()

        # Load EMA params into the live generator
        self.generator_ema.copy_to(self.model.generator)

        try:
            yield True
        finally:
            # Restore training params
            with FSDP.summon_full_params(self.model.generator, writeback=True):
                for n, p in self.model.generator.module.named_parameters():
                    p.data.copy_(backup[n].to(device=p.device, dtype=p.dtype))

    def run_validation(self, label="validation_videos", prompts=None, samples=1, upload=True, broadcast=False, filename_fn=None):
        with torch.no_grad():
            self.model.generator.eval()
            # prompts = [
            #     "In the video, two people are working at a wooden desk, using an iMac computer. One person, wearing a white knit sweater, is using the apple wireless mouse with their right hand, while their left hand rests on the sleek white keyboard. Their movements are smooth yet intentional, suggesting they are focused on a task on the computer screen. The monitor displays a well-organized array of files and folders, hinting at a task that involves detailed organization or detailed data navigation. The second person, only subtly visible, sits closely by and appears to observe or assist, creating a collaborative atmosphere. Their presence adds a quiet dynamic to the scene, as if they are ready to provide input or guidance. Sticky notes with handwritten notes are attached to the monitor’s stand, adding a touch of personal organization amidst the digital workspace. The focus on the keyboard and mouse emphasizes a streamlined workflow, indicative of a productive work environment. The overall ambiance is calm and focuses on teamwork, technology, and efficient workspace management.",
            #     "A determined climber is scaling a massive rock face, showcasing exceptional strength and skill. The person, clad in a teal shirt and dark pants, climbs with precision, their movements measured and deliberate. They are secured by climbing gear, which includes ropes and a harness, emphasizing their commitment to safety. The rugged texture of the sandy-colored rock provides an imposing backdrop, adding drama and scale to the climb. In the distance, other large rock formations and sparse vegetation can be seen under a bright, overcast sky, contributing to the natural and adventurous atmosphere. The scene captures a moment of focus and challenge, highlighting the climber's tenacity and the breathtaking environment.",
            #     "In the video, a lone musician stands gracefully in front of a grand cathedral, playing an accordion while surrounded by the lively water display of a central fountain. Dressed in a casual ensemble, he wears a light-colored shirt, dark pants, and a flat cap that gives him a vintage charm. His posture is relaxed, yet engaged, as he sways gently in rhythm with the music, casting soft shadows on the cobblestone steps beneath him. The backdrop features the cathedral's towering twin spires, with intricate stonework that casts a rich, historical aura around the scene. Sunlight bathes the entire setting, enhancing the golden hues of the cathedral facade and creating a halo-like effect around the musician. The fountain's water jets splash playfully, catching glimmers of light and adding a dynamic element to the tranquil atmosphere. The scene captures a harmonious blend of architectural majesty and human creativity, framed by the clear, azure sky that extends infinitely above. It's a vivid depiction of solitude and artistry, set against a timeless urban landscape.",
            #     "In the video, a fluffy dog with brown patches is intently engaged with a bright red toy shaped like a fire hydrant, which has a yellow and orange rope attached. The dog's body is relaxed as it lies on a plain white background, concentrating on nudging and playfully biting the toy. Its ears perk up slightly with curiosity, and its eyes are fixated on the toy, suggesting a scene of focused playfulness. The neutral tones of the dog's fur contrast starkly against the vivid red of the toy, creating a visually striking moment.",
            # ]
            
            prompts = prompts if prompts is not None else self.val_prompts
            bsz_per_gpu = max(1, (len(prompts) + self.world_size - 1) // self.world_size)
            local_prompts = prompts[self.global_rank * bsz_per_gpu: (self.global_rank + 1) * bsz_per_gpu]
            for sample_num in range(samples):
                validation = []
                for prompt in local_prompts:
                    validation.append(
                        self.generate_video(
                            self.val_pipeline,
                            prompts=[prompt],
                        )
                    )
                if len(local_prompts) < bsz_per_gpu:
                    for _ in range(bsz_per_gpu - len(local_prompts)):
                        self.generate_video(
                            self.val_pipeline,
                            prompts=["dummy_prompt"],
                        )
                if self.is_main_process:
                    all_prompts = list(local_prompts)
                    all_videos = list(validation)

                    for rank in range(1, self.world_size):
                        recv_prompts = self.recv_object(src=rank)
                        recv_videos = self.recv_object(src=rank)
                        all_prompts.extend(recv_prompts)
                        all_videos.extend(recv_videos)

                    all_videos = np.concatenate(all_videos, axis=0)
                    if broadcast:
                        for dst in range(8, self.world_size, 8):
                            self.send_object(all_prompts, dst=dst)
                            self.send_object(all_videos, dst=dst)

                    log_videos = []
                    for i, prompt in enumerate(all_prompts):
                        filename = f"/workspace/temp_{self.step}_{i}_{sample_num}.mp4" if filename_fn is None else filename_fn(self.step, i, sample_num, prompt)
                        write_video(filename, all_videos[i], fps=16)
                        if upload:
                            log_videos.append(wandb.Video(filename, caption=prompt))
                    if upload:
                        logs = { f"{label}_sample{sample_num}" if sample_num > 1 else label : log_videos }
                        wandb.log(logs, step=self.step)
                else:
                    self.send_object(local_prompts, dst=0)
                    self.send_object(validation, dst=0)
                    if broadcast and self.local_rank == 0:
                        all_prompts = self.recv_object(src=0)
                        all_videos = self.recv_object(src=0)
                        for i, prompt in enumerate(all_prompts):
                            filename = f"/workspace/temp_{self.step}_{i}_{sample_num}.mp4" if filename_fn is None else filename_fn(self.step, i, sample_num, prompt)
                            write_video(filename, all_videos[i], fps=16)

        self.model.generator.train()
        barrier()

    def run_final_validation(self, prompts=None, samples=1, filename_fn=None):
        prompts = prompts if prompts is not None else self.val_prompts
        if self.rename_prompts:
            with open("prompts/vbench/all_dimension.txt", "r") as f:
                names = [line.strip() for line in f.readlines()]
        else:
            names = [prompt[:100] for prompt in prompts]
        with torch.no_grad():
            self.model.generator.eval()
            bsz_per_gpu = len(prompts) // self.world_size
            print(f"rank {self.global_rank} processing {self.global_rank * bsz_per_gpu} to {(self.global_rank + 1) * bsz_per_gpu}")
            if self.global_rank < len(prompts) % self.world_size:
                print(f"rank {self.global_rank} processing extra prompt")
            local_prompts = prompts[self.global_rank * bsz_per_gpu: (self.global_rank + 1) * bsz_per_gpu]
            local_names = names[self.global_rank * bsz_per_gpu: (self.global_rank + 1) * bsz_per_gpu]
            if self.global_rank < len(prompts) % self.world_size:
                local_prompts.append(prompts[-self.global_rank - 1])
                local_names.append(names[-self.global_rank - 1])
            for sample_num in range(samples):
                validation = []
                for prompt in local_prompts:
                    validation.append(
                        self.generate_video(
                            self.val_pipeline,
                            prompts=[prompt],
                        )
                    )
                if len(local_prompts) % self.world_size != 0 and self.global_rank >= len(prompts) % self.world_size:
                    # run extra prompt to keep sync
                    self.generate_video(
                        self.val_pipeline,
                        prompts=["dummy_prompt"],
                    )
                all_prompts = list(local_prompts)
                all_names = list(local_names)
                all_videos = list(validation)
                if len(all_prompts) > 0:
                    all_videos = np.concatenate(all_videos, axis=0)
                    for i, (prompt, name) in enumerate(zip(all_prompts, all_names)):
                        filename = f"/workspace/temp_{self.step}_{i}_{sample_num}.mp4" if filename_fn is None else filename_fn(self.step, i, sample_num, name)
                        write_video(filename, all_videos[i], fps=16)
        self.model.generator.train()
        barrier()

    def train(self):
        start_step = self.step

        while self.step <= 1000 and not self.inference_only:
            if self.step % 100 == 0 and not (self.disable_wandb or self.val_prompts is None):
                self.run_validation()
                if self.generator_ema is not None:
                    with self.use_generator_ema():
                        self.run_validation("validation_videos_ema")

            batch = next(self.dataloader)
            self.train_one_step(batch)
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                if self.step == 1000:
                    self.save()
                    torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

        if self.benchmark_prompts is not None:
            if self.generator_ema is not None:
                os.makedirs("/workspace/vbench_videos_ema", exist_ok=True)
                with self.use_generator_ema():
                    filename_fn = lambda step, i, sample_num, prompt: f"/workspace/vbench_videos_ema/{prompt}-{sample_num}.mp4"
                    self.run_final_validation(prompts=self.benchmark_prompts, samples=self.benchmark_samples, filename_fn=filename_fn)
                    if self.local_rank == 0:
                        os.system(f"aws s3 cp /workspace/vbench_videos_ema s3://agi-mm-training-shared-us-east-2/beidchen/data/{self.run_name}_ema_vbench_videos/ --region us-east-2 --recursive")

            os.makedirs("/workspace/vbench_videos", exist_ok=True)
            filename_fn = lambda step, i, sample_num, prompt: f"/workspace/vbench_videos/{prompt}-{sample_num}.mp4"
            self.run_final_validation(prompts=self.benchmark_prompts, samples=self.benchmark_samples, filename_fn=filename_fn)
            if self.local_rank == 0:
                print(f"uploading data from rank {self.global_rank}")
                os.system("ls -l /workspace/vbench_videos | wc -l")
                os.system(f"aws s3 cp /workspace/vbench_videos s3://agi-mm-training-shared-us-east-2/beidchen/data/{self.run_name}_vbench_videos/ --region us-east-2 --recursive")
            barrier()

        torch.distributed.destroy_process_group(self.cpu_group)
        self.cpu_group = None
