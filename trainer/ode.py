from contextlib import contextmanager
import gc
import glob
import logging
import pickle
import numpy as np
from collections import defaultdict

from utils.dataset import ODERegressionLMDBDataset, cycle
from model import ODERegression
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed
)
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
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

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer

        assert config.distribution_loss == "ode", "Only ODE loss is supported for ODE training"
        self.model = ODERegression(config, device=self.device)

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
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
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

        # Step 3: Initialize the dataloader
        dataset = ODERegressionLMDBDataset(
            config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        total_batch_size = getattr(config, "total_batch_size", None)
        if total_batch_size is not None:
            assert total_batch_size == config.batch_size * self.world_size, "Gradient accumulation is not supported for ODE training"
        self.dataloader = cycle(dataloader)

        self.step = 0
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
            if "generator_ema" in state_dict and self.generator_ema is not None:
                self.generator_ema.load_state_dict(
                    state_dict["generator_ema"], strict=True
                )
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
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
        state_dict = {
            "generator": generator_state_dict
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self):
        VISUALIZE = self.step % 100 == 0
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        ode_latent = batch["ode_latent"].to(
            device=self.device, dtype=self.dtype)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            ode_latent=ode_latent,
            conditional_dict=conditional_dict
        )

        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        if self.world_size > 1:
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape],
                dtype=unnormalized_loss.dtype, device=self.device)
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape],
                dtype=timestep.dtype, device=self.device)

            dist.all_gather_into_tensor(
                gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        loss_breakdown = defaultdict(list)
        stats = {}

        for index, t in enumerate(timestep):
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(
                unnormalized_loss[index].item())

        for key_t in loss_breakdown.keys():
            stats["loss_at_time_" + key_t] = sum(loss_breakdown[key_t]) / \
                len(loss_breakdown[key_t])

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Step 4: Visualization
        if VISUALIZE and not self.config.no_visualize and not self.config.disable_wandb and self.is_main_process:
            # Visualize the input, output, and ground truth
            input = log_dict["input"]
            output = log_dict["output"]
            ground_truth = ode_latent[:, -1]

            input_video = self.model.vae.decode_to_pixel(input)
            output_video = self.model.vae.decode_to_pixel(output)
            ground_truth_video = self.model.vae.decode_to_pixel(ground_truth)
            input_video = 255.0 * (input_video.cpu().numpy() * 0.5 + 0.5)
            output_video = 255.0 * (output_video.cpu().numpy() * 0.5 + 0.5)
            ground_truth_video = 255.0 * (ground_truth_video.cpu().numpy() * 0.5 + 0.5)

            # Visualize the input, output, and ground truth
            wandb.log({
                "input": wandb.Video(input_video, caption="Input", fps=16, format="mp4"),
                "output": wandb.Video(output_video, caption="Output", fps=16, format="mp4"),
                "ground_truth": wandb.Video(ground_truth_video, caption="Ground Truth", fps=16, format="mp4"),
            }, step=self.step)

        # Step 5: Logging
        if self.is_main_process and not self.disable_wandb:
            wandb_loss_dict = {
                "generator_loss": generator_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                **stats
            }
            wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

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

    def train(self):
        while self.step <= 1000:
            if self.step % 100 == 0 and not (self.disable_wandb or self.val_prompts is None):
                with torch.no_grad(), self.use_generator_ema():
                    self.model.generator.eval()
                    # prompts = [
                    #     "In the video, two people are working at a wooden desk, using an iMac computer. One person, wearing a white knit sweater, is using the apple wireless mouse with their right hand, while their left hand rests on the sleek white keyboard. Their movements are smooth yet intentional, suggesting they are focused on a task on the computer screen. The monitor displays a well-organized array of files and folders, hinting at a task that involves detailed organization or detailed data navigation. The second person, only subtly visible, sits closely by and appears to observe or assist, creating a collaborative atmosphere. Their presence adds a quiet dynamic to the scene, as if they are ready to provide input or guidance. Sticky notes with handwritten notes are attached to the monitor’s stand, adding a touch of personal organization amidst the digital workspace. The focus on the keyboard and mouse emphasizes a streamlined workflow, indicative of a productive work environment. The overall ambiance is calm and focuses on teamwork, technology, and efficient workspace management.",
                    #     "A determined climber is scaling a massive rock face, showcasing exceptional strength and skill. The person, clad in a teal shirt and dark pants, climbs with precision, their movements measured and deliberate. They are secured by climbing gear, which includes ropes and a harness, emphasizing their commitment to safety. The rugged texture of the sandy-colored rock provides an imposing backdrop, adding drama and scale to the climb. In the distance, other large rock formations and sparse vegetation can be seen under a bright, overcast sky, contributing to the natural and adventurous atmosphere. The scene captures a moment of focus and challenge, highlighting the climber's tenacity and the breathtaking environment.",
                    #     "In the video, a lone musician stands gracefully in front of a grand cathedral, playing an accordion while surrounded by the lively water display of a central fountain. Dressed in a casual ensemble, he wears a light-colored shirt, dark pants, and a flat cap that gives him a vintage charm. His posture is relaxed, yet engaged, as he sways gently in rhythm with the music, casting soft shadows on the cobblestone steps beneath him. The backdrop features the cathedral's towering twin spires, with intricate stonework that casts a rich, historical aura around the scene. Sunlight bathes the entire setting, enhancing the golden hues of the cathedral facade and creating a halo-like effect around the musician. The fountain's water jets splash playfully, catching glimmers of light and adding a dynamic element to the tranquil atmosphere. The scene captures a harmonious blend of architectural majesty and human creativity, framed by the clear, azure sky that extends infinitely above. It's a vivid depiction of solitude and artistry, set against a timeless urban landscape.",
                    #     "In the video, a fluffy dog with brown patches is intently engaged with a bright red toy shaped like a fire hydrant, which has a yellow and orange rope attached. The dog's body is relaxed as it lies on a plain white background, concentrating on nudging and playfully biting the toy. Its ears perk up slightly with curiosity, and its eyes are fixated on the toy, suggesting a scene of focused playfulness. The neutral tones of the dog's fur contrast starkly against the vivid red of the toy, creating a visually striking moment.",
                    # ]
                    bsz_per_gpu = len(self.val_prompts) // self.world_size
                    local_prompts = self.val_prompts[self.global_rank * bsz_per_gpu: (self.global_rank + 1) * bsz_per_gpu]
                    validation = self.generate_video(
                        self.val_pipeline,
                        prompts=local_prompts,
                    )
                    if self.is_main_process:
                        all_prompts = local_prompts
                        all_videos = [validation]

                        for rank in range(1, self.world_size):
                            recv_prompts = self.recv_object(src=rank)
                            recv_videos = self.recv_object(src=rank)
                            all_prompts.extend(recv_prompts)
                            all_videos.append(recv_videos)

                        all_videos = np.concatenate(all_videos, axis=0)

                        log_videos = []
                        for i, prompt in enumerate(all_prompts):
                            filename = f"/workspace/temp_{self.step}_{i}.mp4"
                            write_video(filename, all_videos[i], fps=16)
                            log_videos.append(wandb.Video(filename, caption=prompt))
                        logs = { "validation_videos": log_videos }
                        wandb.log(logs, step=self.step)
                    else:
                        self.send_object(local_prompts, dst=0)
                        self.send_object(validation, dst=0)

                self.model.generator.train()
                barrier()

            self.train_one_step()

            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

        torch.distributed.destroy_process_group(self.cpu_group)
        self.cpu_group = None
