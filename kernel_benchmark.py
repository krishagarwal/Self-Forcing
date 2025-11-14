# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)

from wan.modules.monarch_attn import DEVICE, monarch_video_attn

# benchmark flash_attention, ref, and forward
import time
N_WARMUP = 10
N_ITER = 100

dtype = torch.bfloat16

for causal in [False, True]:
    print(f"=== CAUSAL: {causal} ===")

    for F in range(21, 22, 3):
        print(f" --- F = {F} --- ")
        Z, block_b1, block_b2, H, HEAD_DIM = 1, 30, 52, 12, 128
        S = F * block_b1 * block_b2
        sm_scale = HEAD_DIM ** -0.5

        Q = torch.randn((Z, S if not causal else 3 * block_b1 * block_b2, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        K = torch.randn((Z, S, H, HEAD_DIM), dtype=dtype, device=DEVICE)
        V = torch.randn((Z, S, H, HEAD_DIM), dtype=dtype, device=DEVICE)

        causal_configs = [(1, 1, 1), (1, 2, 1)]
        noncausal_configs = [(3, 1, 1), (3, 2, 1)]

        configs = causal_configs if causal else noncausal_configs

        for config in configs:
            ft, hr, wr = config
            for _ in range(N_WARMUP):
                out = monarch_video_attn(Q, K, V, ft, hr, wr, 30, 52)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(N_ITER):
                out = monarch_video_attn(Q, K, V, ft, hr, wr, 30, 52)
            torch.cuda.synchronize()
            end = time.time()
            print(f"monarch attn ({ft}, {hr}, {wr}): {(end - start) / N_ITER * 1000:.6f} ms")

        for _ in range(N_WARMUP):
            out = flash_attention(Q, K, V)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(N_ITER):
            out = flash_attention(Q, K, V)
        torch.cuda.synchronize()
        end = time.time()
        print(f"flash_attention 3: {(end - start) / N_ITER * 1000:.6f} ms")


        for _ in range(N_WARMUP):
            out = flash_attention(Q, K, V, version=2)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(N_ITER):
            out = flash_attention(Q, K, V, version=2)
        torch.cuda.synchronize()
        end = time.time()
        print(f"flash_attention 2: {(end - start) / N_ITER * 1000:.6f} ms")
