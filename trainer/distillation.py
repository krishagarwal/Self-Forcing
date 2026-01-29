from contextlib import contextmanager
import copy
import gc
import glob
import logging
import pickle
import numpy as np

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import wandb
import time
import os
from torchvision.io import write_video
from pipeline import CausalInferencePipeline, CausalDiffusionInferencePipeline, BidirectionalInferencePipeline, BidirectionalDiffusionInferencePipeline
from torch.distributed.fsdp import ShardingStrategy
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.device_mesh import init_device_mesh

from safetensors.torch import load_file

# TODO: dirty fix for consistency in eval results
rand_seeds = [49, 32, 16, 7, 90, 81, 55, 77, 16, 86, 64, 1, 72, 47, 56, 75, 22, 21, 16, 38, 61, 5, 17, 89, 83, 32, 30, 51, 19, 3, 34, 38, 38, 87, 91, 94, 9, 9, 66, 23, 99, 83, 21, 51, 67, 57, 10, 94, 51, 64, 19, 38, 64, 54, 67, 66, 82, 71, 96, 33, 66, 68, 34, 14, 13, 62, 45, 52, 18, 26, 38, 52, 81, 68, 83, 95, 96, 97, 73, 62, 2, 21, 0, 72, 58, 95, 91, 45, 70, 20, 47, 12, 55, 2, 84, 49, 78, 21, 99, 31, 33, 46, 59, 71, 29, 24, 35, 45, 56, 87, 48, 94, 82, 49, 42, 15, 13, 98, 50, 14, 22, 54, 38, 8, 64, 85, 89, 22, 30, 13, 52, 29, 94, 39, 89, 54, 7, 89, 58, 68, 25, 40, 21, 2, 73, 73, 65, 12, 51, 50, 46, 9, 17, 64, 48, 84, 95, 78, 87, 55, 14, 32, 32, 71, 80, 47, 87, 66, 55, 38, 0, 89, 48, 0, 42, 49, 74, 96, 35, 53, 7, 37, 78, 67, 52, 37, 33, 88, 32, 79, 74, 2, 46, 42, 11, 74, 20, 49, 26, 1, 77, 0, 45, 21, 93, 32, 76, 1, 0, 53, 80, 31, 34, 81, 28, 43, 10, 14, 29, 16, 3, 1, 26, 96, 90, 78, 48, 35, 40, 90, 24, 22, 18, 76, 19, 32, 26, 26, 36, 0, 22, 66, 71, 11, 6, 28, 12, 44, 58, 77, 22, 33, 10, 29, 42, 91, 82, 18, 94, 34, 96, 39, 55, 48, 66, 35, 26, 74, 80, 33, 0, 65, 87, 6, 27, 6, 47, 93, 44, 74, 24, 82, 50, 35, 97, 21, 32, 34, 54, 60, 42, 12, 81, 60, 88, 49, 78, 56, 60, 69, 66, 61, 85, 10, 72, 64, 82, 81, 41, 39, 59, 32, 37, 73, 63, 68, 59, 71, 57, 30, 56, 32, 22, 94, 66, 36, 93, 14, 30, 61, 54, 83, 48, 19, 16, 24, 18, 97, 30, 17, 4, 63, 75, 3, 76, 52, 6, 8, 32, 81, 91, 16, 69, 34, 2, 3, 71, 57, 23, 81, 39, 16, 9, 50, 79, 41, 19, 30, 43, 13, 93, 43, 54, 57, 70, 47, 82, 45, 16, 72, 95, 11, 36, 32, 72, 49, 53, 25, 50, 23, 54, 35, 19, 49, 43, 0, 43, 16, 15, 56, 89, 3, 51, 45, 87, 25, 59, 93, 45, 89, 87, 60, 93, 78, 46, 90, 80, 63, 93, 72, 86, 52, 72, 51, 67, 44, 8, 87, 16, 1, 49, 76, 17, 71, 57, 71, 67, 95, 31, 10, 89, 60, 94, 99, 92, 87, 31, 52, 93, 79, 2, 48, 36, 38, 23, 80, 8, 80, 98, 74, 10, 71, 23, 22, 83, 59, 1, 79, 26, 82, 43, 47, 47, 40, 86, 55, 35, 78, 78, 69, 91, 11, 51, 68, 30, 59, 52, 49, 99, 87, 81, 73, 28, 43, 18, 27, 52, 42, 86, 59, 24, 73, 47, 33, 80, 15, 97, 54, 9, 69, 87, 8, 87, 81, 7, 4, 23, 36, 9, 89, 26, 8, 63, 17, 88, 57, 95, 3, 37, 72, 94, 64, 37, 86, 27, 45, 35, 93, 1, 80, 49, 36, 38, 87, 71, 4, 25, 11, 6, 61, 56, 96, 87, 89, 6, 12, 91, 63, 29, 50, 53, 37, 48, 52, 47, 92, 14, 60, 60, 31, 52, 98, 84, 77, 0, 74, 8, 66, 44, 99, 11, 42, 43, 42, 33, 48, 7, 47, 24, 22, 38, 94, 69, 47, 97, 86, 61, 7, 15, 90, 57, 76, 71, 29, 29, 56, 97, 41, 2, 15, 56, 91, 27, 24, 97, 58, 84, 78, 71, 97, 67, 72, 42, 87, 67, 73, 11, 32, 98, 35, 96, 42, 80, 15, 96, 59, 5, 21, 69, 84, 30, 92, 50, 21, 38, 39, 7, 62, 76, 40, 51, 58, 60, 31, 63, 25, 80, 22, 16, 87, 91, 40, 52, 13, 60, 48, 78, 35, 24, 66, 70, 29, 55, 9, 25, 16, 83, 18, 92, 51, 60, 23, 13, 97, 97, 35, 92, 86, 12, 50, 93, 53, 24, 5, 85, 42, 69, 19, 84, 51, 8, 51, 55, 31, 21, 36, 24, 62, 73, 86, 91, 72, 94, 3, 52, 22, 72, 71, 95, 18, 53, 47, 72, 32, 1, 74, 0, 42, 57, 8, 42, 11, 85, 25, 91, 12, 16, 68, 39, 63, 78, 41, 95, 66, 81, 30, 67, 0, 5, 5, 40, 46, 1, 40, 13, 64, 72, 85, 85, 34, 58, 3, 39, 74, 93, 54, 11, 22, 23, 28, 21, 90, 25, 91, 94, 3, 21, 30, 34, 84, 4, 61, 5, 39, 41, 65, 46, 4, 98, 80, 5, 62, 21, 76, 75, 66, 66, 17, 16, 91, 33, 26, 39, 61, 58, 40, 7, 5, 92, 15, 62, 75, 68, 25, 25, 58, 57, 23, 68, 83, 89, 87, 46, 57, 95, 61, 54, 70, 25, 92, 86, 69, 86, 54, 51, 3, 61, 13, 21, 54, 67, 99, 36, 78, 87, 57, 49, 59, 15, 61, 2, 0, 28, 93, 25, 5, 41, 86, 18, 77, 54, 97, 48, 34, 35, 18, 1, 71, 35, 62, 34, 16, 19, 5, 52, 70, 27, 6, 46, 25, 67, 14, 17, 79, 67, 14, 32, 93, 11, 7, 60, 12, 10, 19, 76, 48, 99, 18, 25, 39, 21, 79, 78, 76, 2, 60, 66, 67, 72, 24, 48, 10, 83, 84, 23, 75, 88, 5, 76, 95, 43, 22, 95, 11, 36, 11, 11, 67, 29, 79, 69, 92, 87, 43, 53, 10, 90, 22, 2, 37, 74, 60, 35, 76, 78, 77, 54, 78, 92, 82, 38, 27, 11, 13, 35, 90, 71, 79, 86, 95, 31, 67, 65, 86, 46, 69, 39, 58, 51, 71, 13, 79, 5, 71, 67, 88, 13, 6, 44, 29, 33, 51, 39, 42, 41, 18, 46, 87, 98, 47, 43, 89, 63, 16, 83, 48, 73, 30, 26, 93]
seed_count = 0

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
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")
        
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.device_mesh = init_device_mesh(
            "cuda",
            (self.world_size // local_world_size, local_world_size),
            mesh_dim_names=("replicate", "shard"),
        )

        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            ema_model = copy.deepcopy(self.model.generator)
            ema_model = fsdp_wrap(
                ema_model,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy,
                device_mesh=self.device_mesh,
                cpu_offload=getattr(config, "cpu_offload_all", False),
            ) # requires same exact FSDP config as generator
            self.generator_ema = EMA_FSDP(ema_model, decay=ema_weight)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            device_mesh=self.device_mesh,
            cpu_offload=getattr(config, "cpu_offload_all", False),
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            device_mesh=self.device_mesh,
            cpu_offload=getattr(config, "cpu_offload_all", False),
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            device_mesh=self.device_mesh,
            cpu_offload=getattr(config, "cpu_offload_all", False),
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            device_mesh=self.device_mesh,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False) or getattr(config, "cpu_offload_all", False),
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

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
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
        if hasattr(config, "convert_dist_cp") and config.convert_dist_cp is not None and not os.path.exists(checkpoint_path):
            if self.is_main_process:
                print(f"Converting DCP checkpoint from {config.convert_dist_cp}...")
                dcp_to_torch_save(config.convert_dist_cp, checkpoint_path)
            barrier()

        if checkpoint_path is not None:
            print(f"Loading pretrained generator from {checkpoint_path}")
            if os.path.isdir(checkpoint_path):
                shard_paths = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
                if not shard_paths:
                    raise ValueError(
                        f"Checkpoint directory {checkpoint_path} contains no .safetensors files."
                    )
                state_dict = {}
                for shard_path in shard_paths:
                    print(f"  Loading shard: {os.path.basename(shard_path)}")
                    shard_state = load_file(shard_path, device="cpu")
                    state_dict.update(shard_state)
                state_dict = {f"model.{k}" : v for k, v in state_dict.items()}
            elif checkpoint_path.endswith(".safetensors"):
                state_dict = load_file(checkpoint_path, device="cpu")
                state_dict = {f"model.{k}" : v for k, v in state_dict.items()}
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu")

            # if "generator_ema" in state_dict and self.generator_ema is not None:
            #     self.generator_ema.load_state_dict(
            #         state_dict["generator_ema"], strict=True
            #     )
            if "critic" in state_dict:
                self.model.fake_score.load_state_dict(
                    state_dict["critic"], strict=True
                )
            init_ema = getattr(config, "init_ema", False)
            if init_ema and "generator_ema" in state_dict:
                state_dict = state_dict["generator_ema"]
            elif "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {rename_param(k): v for k, v in state_dict.items()}
            self.model.generator.load_state_dict(
                state_dict, strict=False # TODO: temporarily set strict=False to support VSA
            )

        ##############################################################################################################

        # TODO: disabling this for now
        # Let's delete EMA params for early steps to save some computes at training and inference
        # if self.step < config.ema_start_step:
        #     self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
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
        
        if hasattr(config, 'benchmark_prompts_file') and config.benchmark_prompts_file is not None:
            with open(config.benchmark_prompts_file, "r") as f:
                all_prompts = [line.strip() for line in f.readlines()]
            self.benchmark_prompts = all_prompts
            self.benchmark_samples = config.benchmark_samples if hasattr(config, 'benchmark_samples') else 1
            self.rename_prompts = (not config.disable_benchmark_rename) if hasattr(config, 'disable_benchmark_rename') else True
            self.benchmark_output_prefix = config.benchmark_output_prefix if hasattr(config, 'benchmark_output_prefix') else None
        else:
            self.benchmark_prompts = None
            self.benchmark_samples = 0

        self.inference_only = getattr(config, "inference_only", False)

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

    # def save(self):
    #     print("Start gathering distributed model states...")
    #     generator_state_dict = fsdp_state_dict(
    #         self.model.generator)
    #     critic_state_dict = fsdp_state_dict(
    #         self.model.fake_score)

    #     if self.config.ema_start_step < self.step:
    #         state_dict = {
    #             "generator": generator_state_dict,
    #             "critic": critic_state_dict,
    #             "generator_ema": self.generator_ema.state_dict(),
    #         }
    #     else:
    #         state_dict = {
    #             "generator": generator_state_dict,
    #             "critic": critic_state_dict,
    #         }

    #     if self.is_main_process:
    #         os.makedirs(os.path.join(self.output_path,
    #                     f"checkpoint_model_{self.step:06d}"), exist_ok=True)
    #         torch.save(state_dict, os.path.join(self.output_path,
    #                    f"checkpoint_model_{self.step:06d}", "model.pt"))
    #         print("Model saved to", os.path.join(self.output_path,
    #               f"checkpoint_model_{self.step:06d}", "model.pt"))

    def save(self):
        ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        def _get_state_dict():
            opts = StateDictOptions(full_state_dict=False)
            gen_state = get_model_state_dict(self.model.generator, options=opts)
            critic_state = get_model_state_dict(self.model.fake_score, options=opts)
            state = {"generator": gen_state, "critic": critic_state}
            if self.generator_ema is not None:
                state["generator_ema"] = get_model_state_dict(self.generator_ema.ema_model, options=opts)
            return state

        sharding_strategy = getattr(self.model.generator, "sharding_strategy", None)
        is_hybrid = sharding_strategy == ShardingStrategy.HYBRID_SHARD

        if is_hybrid:
            shard_pg = self.device_mesh["shard"].get_group()
            shard_ranks = dist.get_process_group_ranks(shard_pg)
            is_primary_shard_group = 0 in shard_ranks
            if not is_primary_shard_group:
                return
            state = _get_state_dict()
            print(f"[rank {self.global_rank}] Saving HYBRID_SHARD checkpoint via DCP to {ckpt_dir}")
            dist_cp.save(
                state_dict=state,
                checkpoint_id=ckpt_dir,
                process_group=shard_pg,
            )
        else:
            print(f"[rank {self.global_rank}] Saving checkpoint via DCP to {ckpt_dir}")
            state = _get_state_dict()
            dist_cp.save(state_dict=state, checkpoint_id=ckpt_dir)

        if self.is_main_process:
            print(f"[DCP] Saved sharded generator checkpoint to {ckpt_dir}")

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

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

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        global seed_count, rand_seeds
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            torch.manual_seed(rand_seeds[seed_count])
            seed_count = (seed_count + 1) % len(rand_seeds)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            torch.manual_seed(rand_seeds[seed_count])
            seed_count = (seed_count + 1) % len(rand_seeds)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
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

        backup = self.model.generator
        self.model.generator = self.generator_ema.ema_model

        # Backup current (non-EMA) params on CPU
        # backup = {}
        # with FSDP.summon_full_params(self.model.generator, writeback=False):
        #     for n, p in self.model.generator.module.named_parameters():
        #         backup[n] = p.detach().clone().cpu()

        # # Load EMA params into the live generator
        # self.generator_ema.copy_to(self.model.generator)

        try:
            yield True
        finally:
            # Restore training params
            # with FSDP.summon_full_params(self.model.generator, writeback=True):
            #     for n, p in self.model.generator.module.named_parameters():
            #         p.data.copy_(backup[n].to(device=p.device, dtype=p.dtype))
            self.model.generator = backup

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

        while self.step <= 700 and not self.inference_only:
            if self.step % 100 == 0 and not (self.disable_wandb or self.val_prompts is None):
                self.run_validation()
                if self.generator_ema is not None and self.step > self.config.ema_start_step:
                    with self.use_generator_ema():
                        self.run_validation("validation_videos_ema")

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None and self.step >= self.config.ema_start_step:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # TODO: disabling this for now, should already be instantiated during init
            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and (self.generator_ema is not None):
                self.generator_ema.copy_(self.model.generator)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                if self.step == 700:
                    self.save()
                    torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

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

        if self.benchmark_prompts is not None:
            if self.generator_ema is not None and self.step > self.config.ema_start_step:
                os.makedirs("/workspace/vbench_videos_ema", exist_ok=True)
                with self.use_generator_ema():
                    filename_fn = lambda step, i, sample_num, prompt: f"/workspace/vbench_videos_ema/{prompt}-{sample_num}.mp4"
                    self.run_final_validation(prompts=self.benchmark_prompts, samples=self.benchmark_samples, filename_fn=filename_fn)
                    if self.local_rank == 0:
                        if self.benchmark_output_prefix is None:
                            path = f"s3://agi-mm-training-shared-us-east-2/beidchen/data/{self.run_name}_ema_vbench_videos/"
                        else:
                            path = f"s3://agi-mm-training-shared-us-east-2/beidchen/data/{self.benchmark_output_prefix}/{self.run_name}_ema_vbench_videos/"
                        os.system(f"aws s3 cp /workspace/vbench_videos_ema {path} --region us-east-2 --recursive")

            os.makedirs("/workspace/vbench_videos", exist_ok=True)
            filename_fn = lambda step, i, sample_num, prompt: f"/workspace/vbench_videos/{prompt}-{sample_num}.mp4"
            self.run_final_validation(prompts=self.benchmark_prompts, samples=self.benchmark_samples, filename_fn=filename_fn)
            if self.local_rank == 0:
                print(f"uploading data from rank {self.global_rank}")
                os.system("ls -l /workspace/vbench_videos | wc -l")
                if self.benchmark_output_prefix is None:
                    path = f"s3://agi-mm-training-shared-us-east-2/beidchen/data/{self.run_name}_vbench_videos/"
                else:
                    path = f"s3://agi-mm-training-shared-us-east-2/beidchen/data/{self.benchmark_output_prefix}/{self.run_name}_vbench_videos/"
                os.system(f"aws s3 cp /workspace/vbench_videos {path} --region us-east-2 --recursive")
            barrier()

        torch.distributed.destroy_process_group(self.cpu_group)
        self.cpu_group = None
