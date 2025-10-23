from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from einops import rearrange
import os

# class MonarchAttnImplicitFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, Q, K, V, sm_scale, num_iters, eps):
#         b, a, i, j, h, _ = Q.shape
#         block_b1, block_b2 = i, j
#         f = K.size(-5)

#         sm_scale_sqrt = sm_scale ** 0.5
#         Q = Q * sm_scale_sqrt
#         K = K * sm_scale_sqrt

#         L = torch.eye(block_b1, device=Q.device, dtype=Q.dtype).view(1, 1, 1, 1, 1, block_b1, block_b1).expand(b, h, a, f, block_b2, block_b1, block_b1) # (b, h, a, f, j, k, i)

#         with torch.no_grad():
#             for _ in range(num_iters):
#                 aR = torch.einsum("bhafjki,baijhd->bafkjhd", L, Q)
#                 bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
#                 # cR = torch.einsum("bhjki->bhkj", L_star).unsqueeze(-1)
#                 # R = torch.softmax(bR / (cR + eps), dim=-1)
#                 cR = L.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).clamp_min(eps).transpose(-2, -3) # (b, h, a, f, k, j, 1)
#                 z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#                 z = z - z.amax(dim=-1, keepdim=True)
#                 R = torch.softmax(z, dim=-1).to(Q.dtype)

#                 aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
#                 bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
#                 # cL = torch.einsum("bhkjl->bhjk", torch.xlogy(R, R)).unsqueeze(-1)
#                 logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
#                 cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
#                 L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
#                 L = torch.softmax(L, dim=-1).to(Q.dtype)
#                 L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)

#         ctx.save_for_backward(L, R, Q, K, V)
#         ctx.eps = eps
#         ctx.sm_scale_sqrt = sm_scale_sqrt

#         out = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V)
#         out = torch.einsum("bhafjki,bafjkhd->baijhd", L, out)
#         return out

#     @staticmethod
#     def backward(ctx, grad_out):
#         L_star, R_star, Q, K, V = ctx.saved_tensors
#         eps = ctx.eps

#         b, a, i, j, h, _ = Q.shape
#         block_b1, block_b2 = i, j
#         f = K.size(-5)

#         grad_tmp = torch.einsum("baijhd,bhafjki->bafjkhd", grad_out, L_star)
#         grad_V = torch.einsum("bhafkjl,bafjkhd->bfklhd", R_star, grad_tmp)
#         grad_R = torch.einsum("bafjkhd,bfklhd->bhafkjl", grad_tmp, V)

#         def R_from_QK(Q_in, K_in):
#             aR = torch.einsum("bhafjki,baijhd->bafkjhd", L_star, Q_in)
#             bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K_in)
#             # cR = torch.einsum("bhjki->bhkj", L_star).unsqueeze(-1)
#             # R = torch.softmax(bR / (cR + eps), dim=-1)
#             cR = L_star.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).clamp_min(eps).transpose(-2, -3) # (b, h, a, f, k, j, 1)
#             z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=-1, keepdim=True)
#             R = torch.softmax(z, dim=-1)
#             return R.to(Q_in.dtype)

#         _, (grad_Q, grad_K) = torch.autograd.functional.vjp(R_from_QK, (Q, K), v=grad_R, create_graph=False, strict=True)

#         def O_from_L(L_in):
#             aR = torch.einsum("bhafjki,baijhd->bafkjhd", L_in, Q)
#             bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
#             # cR = torch.einsum("bhjki->bhkj", L_in).unsqueeze(-1)
#             # R = torch.softmax(bR / (cR + eps), dim=-1)
#             cR = L_in.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).clamp_min(eps).transpose(-2, -3) # (b, h, a, f, k, j, 1)
#             z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=-1, keepdim=True)
#             R = torch.softmax(z, dim=-1).to(L_in.dtype)

#             out = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V)
#             out = torch.einsum("bhafjki,bafjkhd->baijhd", L_in, out)
#             return out
        
#         _, grad_L = torch.autograd.functional.vjp(O_from_L, L_star, v=grad_out, create_graph=False)

#         def L_from_L(L_in):
#             aR = torch.einsum("bhafjki,baijhd->bafkjhd", L_in, Q)
#             bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
#             # cR = torch.einsum("bhjki->bhkj", L_in).unsqueeze(-1)
#             # R_out = torch.softmax(bR / (cR + eps), dim=-1)
#             cR = L_in.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).clamp_min(eps).transpose(-2, -3) # (b, h, a, f, k, j, 1)
#             z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=-1, keepdim=True)
#             R_out = torch.softmax(z, dim=-1).to(L_in.dtype)

#             aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R_out, K)
#             bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
#             # cL = torch.einsum("bhkjl->bhjk", torch.xlogy(R_out, R_out)).unsqueeze(-1)
#             logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, k, j, 1)
#             cL = (R_out * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
#             # L_out = torch.softmax(bL - cL, dim=-2).to(L_in.dtype)
#             L_out = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
#             L_out = torch.softmax(L_out, dim=-1).to(Q.dtype)
#             L_out = rearrange(L_out, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)
#             return L_out

#         u = torch.zeros_like(grad_L)
#         r = grad_L.clone()
#         for _ in range(1):
#             u = u + r
#             _, r = torch.autograd.functional.vjp(L_from_L, L_star, v=r, create_graph=False)
        
#         def L_from_QK(Q, K):
#             aR = torch.einsum("bhafjki,baijhd->bafkjhd", L_star, Q)
#             bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
#             # cR = torch.einsum("bhjki->bhkj", L_star).unsqueeze(-1)
#             # R_out = torch.softmax(bR / (cR + eps), dim=-1)
#             cR = L_star.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).clamp_min(eps).transpose(-2, -3) # (b, h, a, f, k, j, 1)
#             z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=-1, keepdim=True)
#             R_out = torch.softmax(z, dim=-1).to(Q.dtype)

#             aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R_out, K)
#             bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
#             # cL = torch.einsum("bhkjl->bhjk", torch.xlogy(R_out, R_out)).unsqueeze(-1)
#             logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, k, j, 1)
#             cL = (R_out * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
#             # L_out = torch.softmax(bL - cL, dim=-2).to(Q.dtype)
#             L_out = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
#             L_out = torch.softmax(L_out, dim=-1).to(Q.dtype)
#             L_out = rearrange(L_out, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)
#             return L_out

#         _, (grad_Q_L, grad_K_L) = torch.autograd.functional.vjp(
#             L_from_QK, (Q, K), v=u, create_graph=False, strict=True
#         )

#         grad_Q += grad_Q_L
#         grad_K += grad_K_L

#         grad_Q = grad_Q * ctx.sm_scale_sqrt
#         grad_K = grad_K * ctx.sm_scale_sqrt

#         return grad_Q, grad_K, grad_V, None, None, None


class MonarchAttnImplicitFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, sm_scale, num_iters, eps):
        b, a, i, j, h, d = Q.shape
        block_b1, block_b2 = i, j
        k, l = block_b1, block_b2
        f = K.size(-5)

        sm_scale_sqrt = sm_scale ** 0.5
        Q = Q * sm_scale_sqrt
        K = K * sm_scale_sqrt

        aR = Q.sum(-5)
        cR = torch.full((b, h, 1, k, j, 1), fill_value=a, device=Q.device, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(num_iters - 1):
                bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR, K)
                z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
                z = z - z.amax(dim=(-1, -4), keepdim=True)
                R = rearrange(z, 'b h f k j l -> b h k j (f l)')
                R = torch.softmax(R, dim=-1).to(Q.dtype)
                R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
                aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K)
                logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
                cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)
                
                bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q)
                L = torch.softmax(bL - cL, dim=-2).to(Q.dtype)
                aR = torch.einsum("bhajki,baijhd->bkjhd", L, Q)
                cR = L.sum(dim=(-1, -4), dtype=torch.float32, keepdim=True).transpose(-2, -3) # (b, h, 1, k, j, 1)
            
        bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR, K)
        z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
        z = z - z.amax(dim=(-1, -4), keepdim=True)
        R = rearrange(z, 'b h f k j l -> b h k j (f l)')
        R = torch.softmax(R, dim=-1).to(Q.dtype)
        R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
        aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K)
        logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
        cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)
        Y = torch.einsum("bhfkjl,bfklhd->bkjhd", R, V)

        bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q)
        L = torch.softmax(bL - cL, dim=-2).to(Q.dtype)
        out = torch.einsum("bhajki,bkjhd->baijhd", L, Y)

        ctx.save_for_backward(aR, cR, Q, K, V)
        ctx.eps = eps
        ctx.sm_scale_sqrt = sm_scale_sqrt
        
        return out

    @staticmethod
    def backward(ctx, grad_out):
        aR_star, cR_star, Q, K, V = ctx.saved_tensors
        eps = ctx.eps

        b, a, i, j, h, d = Q.shape
        block_b1, block_b2 = i, j
        k, l = block_b1, block_b2
        f = K.shape[-5]

        def O_from_QKV_aR_cR(Q_in, K_in, V_in, aR_in, cR_in):
            bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR_in, K_in)
            z = bR.to(torch.float32) * (1.0 / (cR_in + eps)).clamp_max(1e4)
            z = z - z.amax(dim=(-1, -4), keepdim=True)
            R = rearrange(z, 'b h f k j l -> b h k j (f l)')
            R = torch.softmax(R, dim=-1).to(Q_in.dtype)
            R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
            aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K_in)
            logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
            cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)
            Y = torch.einsum("bhfkjl,bfklhd->bkjhd", R, V_in)

            bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q_in)
            L = torch.softmax(bL - cL, dim=-2).to(Q_in.dtype)
            out = torch.einsum("bhajki,bkjhd->baijhd", L, Y)
            return out

        _, (grad_Q, grad_K, grad_V, grad_aR, grad_cR) = torch.autograd.functional.vjp(O_from_QKV_aR_cR, (Q, K, V, aR_star, cR_star), v=grad_out, create_graph=False, strict=True)

        def aR_cR_from_aR_cR(aRcR_in):
            aR_in = aRcR_in[:b*k*j*h*d].view(b, k, j, h, d)
            cR_in = aRcR_in[b*k*j*h*d:].view(b, h, 1, k, j, 1)

            bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR_in, K)
            z = bR.to(torch.float32) * (1.0 / (cR_in + eps)).clamp_max(1e4)
            z = z - z.amax(dim=(-1, -4), keepdim=True)
            R = rearrange(z, 'b h f k j l -> b h k j (f l)')
            R = torch.softmax(R, dim=-1).to(Q.dtype)
            R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
            aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K)
            logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
            cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)

            bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q)
            L = torch.softmax(bL - cL, dim=-2).to(Q.dtype)

            aR_out = torch.einsum("bhajki,baijhd->bkjhd", L, Q)
            cR_out = L.sum(dim=(-1, -4), dtype=torch.float32, keepdim=True).transpose(-2, -3) # (b, h, 1, k, j, 1)
            aRcR_out = torch.cat([aR_out.flatten(), cR_out.flatten()])
            return aRcR_out
        
        def solve_adj_gmres2(v, lam=0.0):
            v = torch.cat([v_i.flatten() for v_i in v])
            aRcR_star = torch.cat([aR_star.flatten(), cR_star.flatten()])

            b0 = v
            _, Ab0 = torch.autograd.functional.vjp(aR_cR_from_aR_cR, aRcR_star, v=b0, create_graph=False)
            b1 = Ab0

            Sb0 = (1.0 + lam) * b0 - Ab0 # (I - A + λI) b0
            _, Ab1 = torch.autograd.functional.vjp(aR_cR_from_aR_cR, aRcR_star, v=b1, create_graph=False)
            Sb1 = (1.0 + lam) * b1 - Ab1 # (I - A + λI) b1

            # Build 2×2 normal equations G x = rhs
            g00 = (Sb0.flatten() @ Sb0.flatten())
            g01 = (Sb0.flatten() @ Sb1.flatten())
            g11 = (Sb1.flatten() @ Sb1.flatten())
            r0  = (v.flatten()  @ Sb0.flatten())
            r1  = (v.flatten()  @ Sb1.flatten())

            # Solve small SPD system safely
            det = (g00 * g11 - g01 * g01).clamp_min(1e-20)
            x0 = ( r0 * g11 - r1 * g01) / det
            x1 = (-r0 * g01 + r1 * g00) / det

            u = x0 * b0 + x1 * b1
            u1 = u[:b*k*j*h*d].view(b, k, j, h, d)
            u2 = u[b*k*j*h*d:].view(b, h, 1, k, j, 1)
            return (u1, u2)
        
        u = solve_adj_gmres2((grad_aR, grad_cR), lam=0.00)

        # r = (grad_aR.clone(), grad_cR.clone())
        # u = r
        # for _ in range(1, 1):
        #     _, r = torch.autograd.functional.vjp(aR_cR_from_aR_cR, (aR_star, cR_star), v=r, create_graph=False)
        #     u = tuple(a + b for a, b in zip(u, r))

        def aR_cR_from_QK(Q_in, K_in):
            bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR_star, K_in)
            z = bR.to(torch.float32) * (1.0 / (cR_star + eps)).clamp_max(1e4)
            z = z - z.amax(dim=(-1, -4), keepdim=True)
            R = rearrange(z, 'b h f k j l -> b h k j (f l)')
            R = torch.softmax(R, dim=-1).to(Q_in.dtype)
            R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
            aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K_in)
            logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
            cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)

            bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q_in)
            L = torch.softmax(bL - cL, dim=-2).to(Q_in.dtype)

            aR_out = torch.einsum("bhajki,baijhd->bkjhd", L, Q_in)
            cR_out = L.sum(dim=(-1, -4), dtype=torch.float32, keepdim=True).transpose(-2, -3) # (b, h, 1, k, j, 1)
            return aR_out, cR_out

        _, (grad_Q_arcr, grad_K_arcr) = torch.autograd.functional.vjp(
            aR_cR_from_QK, (Q, K), v=u, create_graph=False, strict=True
        )

        grad_Q += grad_Q_arcr
        grad_K += grad_K_arcr

        grad_Q = grad_Q * ctx.sm_scale_sqrt
        grad_K = grad_K * ctx.sm_scale_sqrt

        return grad_Q, grad_K, grad_V, None, None, None

monarch_attn = MonarchAttnImplicitFn.apply

monarch_attn = MonarchAttnImplicitFn.apply

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        self.num_iters = int(os.getenv("MONARCH_ATTN_NUM_ITERS", "1"))
        self.target_sparsity = float(os.getenv("MONARCH_ATTN_TARGET_SPARSITY", "1.0"))
        self.disable_monarch = bool(int(os.getenv("DISABLE_MONARCH_ATTN", "0")))

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def get_block_sizes(self, h, w):
        factors = [i for i in range(1, h + 1) if h % i == 0]
        sparsities = [1 - (f*f*w + w*w*f)/(f*f*w*w) for f in factors]
        dists = [abs(s - self.target_sparsity) for s in sparsities]
        min_idx = dists.index(min(dists))
        return (factors[min_idx], w)

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            assert False
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            # ptr = kv_cache["k"].data_ptr()
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # prev = kv_cache["k"].requires_grad
                # curr = roped_key.requires_grad
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
                # now = kv_cache["k"].requires_grad
                # print("kv_cache_ptr", kv_cache["k"].data_ptr(), max(0, local_end_index - self.max_attention_size), local_start_index, local_end_index)
                # if prev != now and dist.get_rank() == 0:
                #     print(f"requires grad changed from {prev} to {now} (curr grad enabled is {curr}) global grad enabled is {torch.is_grad_enabled()}, kv_cache is at ptr {ptr} and now is at {kv_cache['k'].data_ptr()}")
            # if kv_cache["k"].data_ptr() != ptr and dist.get_rank() == 0:
            #     print("Warning: kv_cache has been reallocated, data_ptr changed from",
            #           ptr, "to", kv_cache["k"].data_ptr())
            if self.disable_monarch:
                x = attention(
                    roped_query,
                    kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                    kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
                )
            else:
                block_b1, block_b2 = self.get_block_sizes(grid_sizes[0, 1].item(), grid_sizes[0, 2].item())
                # block_b1 = grid_sizes[0, 1]
                # block_b2 = grid_sizes[0, 2]
                b, s, h, d = roped_query.shape
                curr_q = roped_query.view(b, -1, block_b1, block_b2, h, d)
                curr_k = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index].view(b, -1, block_b1, block_b2, h, d)
                curr_v = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index].view(b, -1, block_b1, block_b2, h, d)
                x = monarch_attn(curr_q, curr_k, curr_v, d ** -0.5, self.num_iters, self.eps).reshape(b, s, h, d)

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # if block_index == 0 and dist.get_rank() == 0:
                #     print("gradient checkpointing, kv grad enabled", kv_cache[block_index]["k"].requires_grad)
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                # if block_index == 0 and dist.get_rank() == 0:
                #     print("no gradient checkpointing, kv grad enabled", kv_cache[block_index]["k"].requires_grad)
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        # if dist.get_rank() == 0:
        #     print("In forward")
        if kwargs.get('kv_cache', None) is not None:
            # if dist.get_rank() == 0:
            #     print("Using kv cache, grad enabled is", kwargs['kv_cache'][0]["k"].requires_grad)
            return self._forward_inference(*args, **kwargs)
        else:
            # if dist.get_rank() == 0:
            #     print("Not using kv cache")
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
