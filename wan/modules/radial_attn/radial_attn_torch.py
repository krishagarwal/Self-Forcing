import torch
import math
from torch.nn.functional import scaled_dot_product_attention

def _get_diagonal_split_ok(dist: int, token_per_frame: int, block_size: int) -> bool:
    """
    Matches your get_diagonal_split_mask() logic (note: no decay_factor there).
    Returns whether the (frame_i, frame_j) pair is allowed at all.
    """
    if dist == 0:
        return True
    group = dist.bit_length()
    threshold = block_size
    decay_length = (2 ** token_per_frame.bit_length()) / (2 ** group)

    if decay_length >= threshold:
        return True

    split_factor = int(threshold / decay_length)
    return (dist % split_factor) == 0

def _get_window_width(dist: int, token_per_frame: int, *, decay_factor: float, block_size: int, model_type: str):
    """
    Matches your get_window_width() logic.
    """
    if model_type == "wan":
        if dist < 1:
            return token_per_frame
        if dist == 1:
            return token_per_frame // 2
    elif model_type == "hunyuan":
        if dist <= 1:
            return token_per_frame
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    group = dist.bit_length()
    decay_length = (2 ** token_per_frame.bit_length()) / (2 ** group) * decay_factor
    threshold = block_size
    return decay_length if decay_length >= threshold else threshold

@torch.no_grad()
def build_radial_dense_allow_mask(
    video_token_num: int,
    num_frame: int,
    *,
    block_size: int = 128,
    decay_factor: float = 1.0,
    model_type: str = "hunyuan",
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """
    Returns allow_mask: [video_token_num, video_token_num] bool
    True  = attention allowed
    False = attention disallowed

    This is the dense token-level version of your RadialAttn mask (video-video only).
    """
    assert video_token_num % num_frame == 0, "video_token_num must be divisible by num_frame"
    tp = video_token_num // num_frame  # tokens per frame

    device = torch.device(device)
    # Within-frame token index grids
    row = torch.arange(tp, device=device).view(-1, 1)
    col = torch.arange(tp, device=device).view(1, -1)

    # Precompute per-distance local masks (tp x tp) once
    dist2local = {}
    for dist in range(num_frame):
        split_ok = _get_diagonal_split_ok(dist, tp, block_size)

        # window width (token band) for this distance
        window = _get_window_width(
            dist, tp, decay_factor=decay_factor, block_size=block_size, model_type=model_type
        )

        if split_ok:
            # band mask: abs(i-j) <= window
            dist2local[dist] = (col - row).abs() <= window
        else:
            dist2local[dist] = torch.zeros((tp, tp), device=device, dtype=torch.bool)

    # Assemble full allow mask
    allow = torch.zeros((video_token_num, video_token_num), device=device, dtype=torch.bool)

    for i in range(num_frame):
        qi0, qi1 = i * tp, (i + 1) * tp

        # "wan attention sink" behavior from your code: if j == 0, full ones
        if model_type == "wan":
            allow[qi0:qi1, 0:tp] = True

        for j in range(num_frame):
            if model_type == "wan" and j == 0:
                continue
            kj0, kj1 = j * tp, (j + 1) * tp
            dist = abs(i - j)
            allow[qi0:qi1, kj0:kj1] = dist2local[dist]

    return allow

def allow_to_sdpa_bool_mask(allow_mask: torch.Tensor) -> torch.Tensor:
    """
    SDPA bool mask convention: True means "mask out / disallow".
    """
    return ~allow_mask

def radial_sdpa_video_only(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    sdpa_bool_mask_ss: torch.Tensor,
):
    """
    q/k/v: [B, S, H, D] (diffusers-style) with S == video_token_num
    sdpa_bool_mask_ss: [S, S] bool, True=masked-out
    returns: [B, S, H, D]
    """
    # SDPA expects [B, H, S, D] (fastest path)
    q = q_bshd.transpose(1, 2).contiguous()
    k = k_bshd.transpose(1, 2).contiguous()
    v = v_bshd.transpose(1, 2).contiguous()

    # This supports backward through q/k/v (mask is non-grad)
    out = scaled_dot_product_attention(q, k, v, attn_mask=sdpa_bool_mask_ss)
    return out.transpose(1, 2).contiguous()

# --- example ---
if __name__ == "__main__":
    device = "cuda"
    video_token_num = 25440
    num_frame = 16
    B, H, D = 1, 8, 64

    q = torch.randn(B, video_token_num, H, D, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    allow = build_radial_dense_allow_mask(
        video_token_num,
        num_frame,
        block_size=128,
        decay_factor=1.0,
        model_type="hunyuan",
        device=device,
    )
    sdpa_mask = allow_to_sdpa_bool_mask(allow)  # [S,S], True = disallow

    # (Optional) if you want to force a backward-friendly kernel even when masking:
    # torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

    out = radial_sdpa_video_only(q, k, v, sdpa_mask)
    loss = out.square().mean()
    loss.backward()
    print("backward ok:", q.grad is not None, k.grad is not None, v.grad is not None)
