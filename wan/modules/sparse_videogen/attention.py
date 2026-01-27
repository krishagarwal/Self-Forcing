# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import json
import sys
import warnings
from typing import Optional

import flashinfer
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import (
    flex_attention,
)
from .permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton
from .kmeans_utils import (
    batch_kmeans_Euclid,
    density_calculation,
    dynamic_block_sparse_fwd_flashinfer,
    identify_dynamic_map,
)
from .logger import logger
from .timer import time_logging_decorator
from .placement import (
    ref_wan_hidden_states_placement,
    ref_wan_sparse_head_placement,
    wan_hidden_states_placement,
    wan_sparse_head_placement,
)
from .utils import (
    create_block_mask_cached,
    flashinfer_sparse_attn_forward,
    gen_temporal_mask,
    generate_temporal_head_mask_mod,
)

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3


class WanAttn_SVGAttn_Processor2_0:
    version = None
    context_length = 0
    num_frame = 0
    frame_size = 0

    first_layers_fp = 0
    first_times_fp = 0

    num_sampled_rows = 32
    attention_masks = None
    sparsity = 0

    block_mask = None
    temporal_mask_metadata = None

    sample_mse_min_row = 0
    sample_mse_max_row = 10000

    curr_seq_len = 0

    @classmethod
    @time_logging_decorator("SVG - sample mse")
    def sample_mse(cls, query, key, value):
        assert len(cls.attention_masks) == 2

        cfg, num_heads, seq_len, dim = query.size()
        num_sampled_rows = min(cls.num_sampled_rows, seq_len)
        sampled_rows = torch.randint(low=max(0, cls.sample_mse_min_row), high=min(cls.sample_mse_max_row, seq_len), size=(num_sampled_rows,))
        sampled_q = query[:, :, sampled_rows, :]
        sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)

        sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
        sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

        sampled_mses = torch.zeros(len(cls.attention_masks), cfg, num_heads, device=query.device, dtype=query.dtype)

        # Only have Tri-diagonal and Striped
        for mask_idx, attn_mask in enumerate(cls.attention_masks):
            sampled_attention_mask = attn_mask[sampled_rows, :]
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float("-inf"))
            sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
            sampled_mses[mask_idx] = mse

        return sampled_mses

    @classmethod
    @time_logging_decorator("Level 3 - sparse flex attention")
    def sparse_flex_attention(self, query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    @classmethod
    @time_logging_decorator("Level 3 - sparse flashinfer attention")
    def sparse_flashinfer_attention(self, query, key, value, temporal_mask_metadata):
        return flashinfer_sparse_attn_forward(query, key, value, temporal_mask_metadata)

    @classmethod
    @time_logging_decorator("Level 3 - sparse head placement")
    def sparse_head_placement(
        self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
    ):
        query_out, key_out, value_out = ref_wan_sparse_head_placement(
            query, key, value, best_mask_idx, context_length, num_frame, frame_size
        )
        return query_out, key_out, value_out

    @classmethod
    @time_logging_decorator("Level 3 - fast sparse head placement")
    def fast_sparse_head_placement(
        self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
    ):
        wan_sparse_head_placement(
            query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
        )
        return query_out, key_out, value_out

    @classmethod
    @time_logging_decorator("Level 3 - hidden states placement")
    def hidden_states_placement(
        self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
    ):
        ref_wan_hidden_states_placement(
            hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
        )

    @classmethod
    @time_logging_decorator("Level 3 - fast hidden states placement")
    def fast_hidden_states_placement(
        self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
    ):
        wan_hidden_states_placement(
            hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
        )

    @classmethod
    @time_logging_decorator("Level 3 - Dense Flash Attention")
    def flash_attention(self, query, key, value):
        output_hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        return output_hidden_states

    @classmethod
    @time_logging_decorator("SVG - attention core logic")
    def attention_core_logic(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        timestep,
        layer_idx,
    ):
        cfg, num_heads, seq_len, dim = query.size()

        context_length, num_frame, frame_size = cls.context_length, cls.num_frame, cls.frame_size

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Determine if we use Full Attention to calculate
        full_attention_flag = False

        if layer_idx < cls.first_layers_fp:
            full_attention_flag = True
        if timestep > cls.first_times_fp:
            full_attention_flag = True

        if full_attention_flag:
            output_hidden_states = cls.flash_attention(query, key, value)
            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)
        else:
            sampled_mses = cls.sample_mse(query, key, value)
            best_mask_idx = torch.argmin(sampled_mses, dim=0)

            output_hidden_states = torch.zeros_like(query)
            query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

            query_out, key_out, value_out = cls.fast_sparse_head_placement(
                query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
            )

            hidden_states = cls.sparse_flex_attention(query_out, key_out, value_out, block_mask=cls.block_mask)
            # hidden_states = cls.sparse_flashinfer_attention(query_out, key_out, value_out, temporal_mask_metadata=cls.temporal_mask_metadata)

            cls.fast_hidden_states_placement(
                hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
            )

            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)


def prepare_flexattention(
    cfg_size,
    num_head,
    head_dim,
    dtype,
    device,
    context_length,
    prompt_length,
    num_frame,
    frame_size,
    diag_width=1,
    multiplier=2,
):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"

    seq_len = context_length + num_frame * frame_size
    query, key, value = [
        torch.zeros((cfg_size, num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)
    ]

    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)
    _ = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask


def prepare_flashinfer_attention(
    cfg_size,
    num_head,
    head_dim,
    dtype,
    device,
    context_length,
    prompt_length,
    num_frame,
    frame_size,
    diag_width=1,
    multiplier=2,
):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"

    temporal_mask_metadata = gen_temporal_mask(num_frame, frame_size, multiplier)

    return temporal_mask_metadata


# ---- Semantic Aware Permutation Processor ----
class WanAttn_SAPAttn_Processor(WanAttn_SVGAttn_Processor2_0):
    num_q_centroids = 0
    num_k_centroids = 0
    top_p_kmeans = 0
    min_kc_ratio = 0

    centroids_init = False
    q_centroids = None
    k_centroids = None

    kmeans_iter_init = 0
    kmeans_iter_step = 0
    zero_step_kmeans_init = False

    logging_file = None

    def __init__(
        self,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx

    @time_logging_decorator("Level 3.7 - kmeans init")
    def kmeans_init(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_q_centroids, max_iters=self.kmeans_iter_init
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_k_centroids, max_iters=self.kmeans_iter_init
        )

        self.q_centroids = qcentroids
        self.k_centroids = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @time_logging_decorator("Level 3.7 - kmeans step")
    def kmeans_step(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_q_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_k_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.k_centroids,
        )

        self.q_centroids = qcentroids
        self.k_centroids = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @time_logging_decorator("Level 3.5 - kmeans clustering")
    def kmeans_clustering(self, query, key, layer_idx):
        if not self.centroids_init:
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_init(
                query, key, layer_idx
            )
            self.centroids_init = True
            print(f"Centroids initialized at layer {layer_idx}. Init step: {self.kmeans_iter_init}")
        else:
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_step(
                query, key, layer_idx
            )

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @time_logging_decorator("Level 3 - semantic aware permutation")
    def semantic_aware_permutation(self, query, key, value):
        cfg, num_heads, seq_len, dim = query.size()

        # 1. Kmeans clustering
        qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_clustering(
            query, key, self.layer_idx
        )

        # 2. Identify dynamic map
        q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
        k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)

        dynamic_map = identify_dynamic_map(
            qcentroids.view(cfg, num_heads, self.num_q_centroids, dim),
            kcentroids.view(cfg, num_heads, self.num_k_centroids, dim),
            q_cluster_sizes,
            k_cluster_sizes,
            self.top_p_kmeans,
            self.min_kc_ratio,
        )

        # 3. Permute the query, key, value
        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, qlabels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
        v_permuted, v_sorted_indices = permute_tensor_by_labels_triton(
            value, klabels, dim=2, sorted_indices=k_sorted_indices
        )

        return q_permuted, k_permuted, v_permuted, dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices

    @time_logging_decorator("Level 3 - Dense Flashinfer Attention")
    def flashinfer_attention(self, query, key, value):

        cfg, num_heads, seq_len, dim = query.size()

        query = query.flatten(0, 1).permute(1, 0, 2)
        key = key.flatten(0, 1).permute(1, 0, 2)
        value = value.flatten(0, 1).permute(1, 0, 2)

        o, o_lse = flashinfer.single_prefill_with_kv_cache(
            query,
            key,
            value,
            causal=False,
            return_lse=True,
        )

        o = o.permute(1, 0, 2).reshape(cfg, num_heads, seq_len, dim)

        return o

    @time_logging_decorator("Level 2 - attention core logic")
    def attention_core_logic(self, query, key, value, timestep):
        cfg, num_heads, seq_len, dim = query.size()
        assert cfg == 1, "Batch size must be 1 for kmeans block sparse attention"

        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Determine if we use Full Attention to calculate
        full_attention_flag = False

        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        if timestep < self.first_times_fp:
            full_attention_flag = True

        if full_attention_flag:
            if self.zero_step_kmeans_init:
                video_length = self.num_frame * self.frame_size
                query_video = query[:, :, :video_length, :].contiguous()
                key_video = key[:, :, :video_length, :].contiguous()
                self.kmeans_clustering(query_video, key_video, self.layer_idx)

            output_hidden_states = self.flash_attention(query, key, value)
            # output_hidden_states = self.flashinfer_attention(query, key, value)
            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)

        else:
            q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices = self.semantic_aware_permutation(
                query, key, value
            )

            output_permuted = dynamic_block_sparse_fwd_flashinfer(
                q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, is_cpu=False
            )

            attn_output = apply_inverse_permutation_triton(output_permuted, q_sorted_indices, dim=2)

            # Save time, layer, density information to logging file
            if self.logging_file is not None:
                with time_logging_decorator("Level 3 - density calculation and logging"):
                    # 4. Calculate density
                    densities = density_calculation(dyn_map, qc_sz_s, kc_sz_s)

                    avg_density = densities.mean().item()
                    log_entry = {
                        "timestep": timestep,
                        "layer": self.layer_idx,
                        "avg_density": avg_density,
                        "density": densities.tolist(),
                    }

                    # print(f"Time Step: {timestep} Layer: {self.layer_idx} Density: {avg_density}")

                    with open(self.logging_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

            return attn_output.reshape(cfg, num_heads, seq_len, dim)
