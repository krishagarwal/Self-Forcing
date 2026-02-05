from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from utils.resolution import total_seq_len, frame_height, frame_width

# TODO: dirty fix for consistency in eval results
rand_seeds = [49, 32, 16, 7, 90, 81, 55, 77, 16, 86, 64, 1, 72, 47, 56, 75, 22, 21, 16, 38, 61, 5, 17, 89, 83, 32, 30, 51, 19, 3, 34, 38, 38, 87, 91, 94, 9, 9, 66, 23, 99, 83, 21, 51, 67, 57, 10, 94, 51, 64, 19, 38, 64, 54, 67, 66, 82, 71, 96, 33, 66, 68, 34, 14, 13, 62, 45, 52, 18, 26, 38, 52, 81, 68, 83, 95, 96, 97, 73, 62, 2, 21, 0, 72, 58, 95, 91, 45, 70, 20, 47, 12, 55, 2, 84, 49, 78, 21, 99, 31, 33, 46, 59, 71, 29, 24, 35, 45, 56, 87, 48, 94, 82, 49, 42, 15, 13, 98, 50, 14, 22, 54, 38, 8, 64, 85, 89, 22, 30, 13, 52, 29, 94, 39, 89, 54, 7, 89, 58, 68, 25, 40, 21, 2, 73, 73, 65, 12, 51, 50, 46, 9, 17, 64, 48, 84, 95, 78, 87, 55, 14, 32, 32, 71, 80, 47, 87, 66, 55, 38, 0, 89, 48, 0, 42, 49, 74, 96, 35, 53, 7, 37, 78, 67, 52, 37, 33, 88, 32, 79, 74, 2, 46, 42, 11, 74, 20, 49, 26, 1, 77, 0, 45, 21, 93, 32, 76, 1, 0, 53, 80, 31, 34, 81, 28, 43, 10, 14, 29, 16, 3, 1, 26, 96, 90, 78, 48, 35, 40, 90, 24, 22, 18, 76, 19, 32, 26, 26, 36, 0, 22, 66, 71, 11, 6, 28, 12, 44, 58, 77, 22, 33, 10, 29, 42, 91, 82, 18, 94, 34, 96, 39, 55, 48, 66, 35, 26, 74, 80, 33, 0, 65, 87, 6, 27, 6, 47, 93, 44, 74, 24, 82, 50, 35, 97, 21, 32, 34, 54, 60, 42, 12, 81, 60, 88, 49, 78, 56, 60, 69, 66, 61, 85, 10, 72, 64, 82, 81, 41, 39, 59, 32, 37, 73, 63, 68, 59, 71, 57, 30, 56, 32, 22, 94, 66, 36, 93, 14, 30, 61, 54, 83, 48, 19, 16, 24, 18, 97, 30, 17, 4, 63, 75, 3, 76, 52, 6, 8, 32, 81, 91, 16, 69, 34, 2, 3, 71, 57, 23, 81, 39, 16, 9, 50, 79, 41, 19, 30, 43, 13, 93, 43, 54, 57, 70, 47, 82, 45, 16, 72, 95, 11, 36, 32, 72, 49, 53, 25, 50, 23, 54, 35, 19, 49, 43, 0, 43, 16, 15, 56, 89, 3, 51, 45, 87, 25, 59, 93, 45, 89, 87, 60, 93, 78, 46, 90, 80, 63, 93, 72, 86, 52, 72, 51, 67, 44, 8, 87, 16, 1, 49, 76, 17, 71, 57, 71, 67, 95, 31, 10, 89, 60, 94, 99, 92, 87, 31, 52, 93, 79, 2, 48, 36, 38, 23, 80, 8, 80, 98, 74, 10, 71, 23, 22, 83, 59, 1, 79, 26, 82, 43, 47, 47, 40, 86, 55, 35, 78, 78, 69, 91, 11, 51, 68, 30, 59, 52, 49, 99, 87, 81, 73, 28, 43, 18, 27, 52, 42, 86, 59, 24, 73, 47, 33, 80, 15, 97, 54, 9, 69, 87, 8, 87, 81, 7, 4, 23, 36, 9, 89, 26, 8, 63, 17, 88, 57, 95, 3, 37, 72, 94, 64, 37, 86, 27, 45, 35, 93, 1, 80, 49, 36, 38, 87, 71, 4, 25, 11, 6, 61, 56, 96, 87, 89, 6, 12, 91, 63, 29, 50, 53, 37, 48, 52, 47, 92, 14, 60, 60, 31, 52, 98, 84, 77, 0, 74, 8, 66, 44, 99, 11, 42, 43, 42, 33, 48, 7, 47, 24, 22, 38, 94, 69, 47, 97, 86, 61, 7, 15, 90, 57, 76, 71, 29, 29, 56, 97, 41, 2, 15, 56, 91, 27, 24, 97, 58, 84, 78, 71, 97, 67, 72, 42, 87, 67, 73, 11, 32, 98, 35, 96, 42, 80, 15, 96, 59, 5, 21, 69, 84, 30, 92, 50, 21, 38, 39, 7, 62, 76, 40, 51, 58, 60, 31, 63, 25, 80, 22, 16, 87, 91, 40, 52, 13, 60, 48, 78, 35, 24, 66, 70, 29, 55, 9, 25, 16, 83, 18, 92, 51, 60, 23, 13, 97, 97, 35, 92, 86, 12, 50, 93, 53, 24, 5, 85, 42, 69, 19, 84, 51, 8, 51, 55, 31, 21, 36, 24, 62, 73, 86, 91, 72, 94, 3, 52, 22, 72, 71, 95, 18, 53, 47, 72, 32, 1, 74, 0, 42, 57, 8, 42, 11, 85, 25, 91, 12, 16, 68, 39, 63, 78, 41, 95, 66, 81, 30, 67, 0, 5, 5, 40, 46, 1, 40, 13, 64, 72, 85, 85, 34, 58, 3, 39, 74, 93, 54, 11, 22, 23, 28, 21, 90, 25, 91, 94, 3, 21, 30, 34, 84, 4, 61, 5, 39, 41, 65, 46, 4, 98, 80, 5, 62, 21, 76, 75, 66, 66, 17, 16, 91, 33, 26, 39, 61, 58, 40, 7, 5, 92, 15, 62, 75, 68, 25, 25, 58, 57, 23, 68, 83, 89, 87, 46, 57, 95, 61, 54, 70, 25, 92, 86, 69, 86, 54, 51, 3, 61, 13, 21, 54, 67, 99, 36, 78, 87, 57, 49, 59, 15, 61, 2, 0, 28, 93, 25, 5, 41, 86, 18, 77, 54, 97, 48, 34, 35, 18, 1, 71, 35, 62, 34, 16, 19, 5, 52, 70, 27, 6, 46, 25, 67, 14, 17, 79, 67, 14, 32, 93, 11, 7, 60, 12, 10, 19, 76, 48, 99, 18, 25, 39, 21, 79, 78, 76, 2, 60, 66, 67, 72, 24, 48, 10, 83, 84, 23, 75, 88, 5, 76, 95, 43, 22, 95, 11, 36, 11, 11, 67, 29, 79, 69, 92, 87, 43, 53, 10, 90, 22, 2, 37, 74, 60, 35, 76, 78, 77, 54, 78, 92, 82, 38, 27, 11, 13, 35, 90, 71, 79, 86, 95, 31, 67, 65, 86, 46, 69, 39, 58, 51, 71, 13, 79, 5, 71, 67, 88, 13, 6, 44, 29, 33, 51, 39, 42, 41, 18, 46, 87, 98, 47, 43, 89, 63, 16, 83, 48, 73, 30, 26, 93]
seed_count = 0

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = self.generator.model.num_layers
        self.frame_seq_length = frame_height * frame_width

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        global seed_count, rand_seeds

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]

                    torch.manual_seed(rand_seeds[seed_count])
                    seed_count = (seed_count + 1) % len(rand_seeds)

                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = total_seq_len

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, self.generator.model.num_heads, self.generator.model.dim // self.generator.model.num_heads], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, self.generator.model.num_heads, self.generator.model.dim // self.generator.model.num_heads], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
