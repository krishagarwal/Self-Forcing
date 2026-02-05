from wan.modules.attention import attention
from wan.modules.attn_patch import full_attention_with_kv_cache
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
from .sparse_videogen.attention import WanAttn_SVGAttn_Processor2_0, prepare_flexattention
from .sparse_videogen.utils import get_attention_mask, sparsity_to_width
from .radial_attn.attn_mask import MaskMap, RadialAttention
from .monarch_attn import monarch_attn, monarch_attn_with_kv_cache

from utils.resolution import frame_height, frame_width, total_seq_len

# class MonarchAttnImplicitFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, Q, K, V, sm_scale, num_iters, eps):
#         b, a, i, j, h, _ = Q.shape
#         block_b1, block_b2 = i, j
#         k, l = block_b1, block_b2
#         f = K.shape[-5]

#         sm_scale_sqrt = sm_scale ** 0.5
#         Q = Q * sm_scale_sqrt
#         K = K * sm_scale_sqrt

#         aR = Q.clone().unsqueeze(-5).expand(-1, -1, f, -1, -1, -1, -1) # (b, a, f, k, j, h, d)
#         cR = torch.ones((b, h, a, f, k, j, 1), device=Q.device, dtype=Q.dtype) # (b, h, a, f, k, j, 1)

#         with torch.no_grad():
#             for _ in range(num_iters - 1):
#                 bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
#                 z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#                 z = z - z.amax(dim=-1, keepdim=True)
#                 R = torch.softmax(z, dim=-1).to(Q.dtype)
#                 aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
#                 logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
#                 cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)

#                 bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
#                 L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
#                 L = torch.softmax(L, dim=-1).to(Q.dtype)
#                 L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=k)

#                 aR = torch.einsum("bhafjki,baijhd->bafkjhd", L, Q) 
#                 cR = L.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).transpose(-2, -3) # (b, h, a, f, k, j, 1)
        
#         bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
#         z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#         z = z - z.amax(dim=-1, keepdim=True)
#         R = torch.softmax(z, dim=-1).to(Q.dtype)
#         aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
#         logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
#         cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
#         Y = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V)

#         bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
#         L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
#         L = torch.softmax(L, dim=-1).to(Q.dtype)
#         L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=k)

#         out = torch.einsum("bhafjki,bafjkhd->baijhd", L, Y)

#         ctx.save_for_backward(aR, cR, Q, K, V)
#         ctx.eps = eps
#         ctx.sm_scale_sqrt = sm_scale_sqrt

#         return out

#     @staticmethod
#     def backward(ctx, grad_out):
#         aR_star, cR_star, Q, K, V = ctx.saved_tensors
#         eps = ctx.eps

#         b, a, i, j, h, _ = Q.shape
#         block_b1, block_b2 = i, j
#         k, l = block_b1, block_b2
#         f = K.shape[-5]

#         def O_from_QKV_aR_cR(Q_in, K_in, V_in, aR_in, cR_in):
#             bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR_in, K_in)
#             z = bR.to(torch.float32) * (1.0 / (cR_in + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=-1, keepdim=True)
#             R = torch.softmax(z, dim=-1).to(Q_in.dtype)
#             aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K_in)
#             logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
#             cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
#             Y = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V_in)

#             bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q_in)
#             L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
#             L = torch.softmax(L, dim=-1).to(Q_in.dtype)
#             L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=k)
#             out = torch.einsum("bhafjki,bafjkhd->baijhd", L, Y)
#             return out

#         _, (grad_Q, grad_K, grad_V, grad_aR, grad_cR) = torch.autograd.functional.vjp(O_from_QKV_aR_cR, (Q, K, V, aR_star, cR_star), v=grad_out, create_graph=False, strict=True)

#         def aR_cR_from_aR_cR(aR_in, cR_in):
#             bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR_in, K)
#             z = bR.to(torch.float32) * (1.0 / (cR_in + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=-1, keepdim=True)
#             R = torch.softmax(z, dim=-1).to(Q.dtype)
#             aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
#             logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
#             cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)

#             bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
#             L = torch.softmax(bL - cL, dim=-2).to(Q.dtype)
#             L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
#             L = torch.softmax(L, dim=-1).to(Q.dtype)
#             L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=k)

#             aR_out = torch.einsum("bhafjki,baijhd->bafkjhd", L, Q)
#             cR_out = L.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).transpose(-2, -3) # (b, h, a, f, k, j, 1)
#             return aR_out, cR_out
        
#         def solve_adj_gmres2(v, lam=0.0):
#             b0 = v
#             _, Ab0 = torch.autograd.functional.vjp(aR_cR_from_aR_cR, (aR_star, cR_star), v=b0, create_graph=False)
#             b1 = Ab0

#             Sb0 = tuple((1.0 + lam) * b0_i - Ab0_i for b0_i, Ab0_i in zip(b0, Ab0)) # (I - A + λI) b0
#             _, Ab1 = torch.autograd.functional.vjp(aR_cR_from_aR_cR, (aR_star, cR_star), v=b1, create_graph=False)
#             Sb1 = tuple((1.0 + lam) * b1_i - Ab1_i for b1_i, Ab1_i in zip(b1, Ab1)) # (I - A + λI) b1

#             # Build 2×2 normal equations G x = rhs
#             g00 = sum((Sb0_i.flatten() @ Sb0_i.flatten()).to(torch.float32) for Sb0_i in Sb0)
#             g01 = sum((Sb0_i.flatten() @ Sb1_i.flatten()).to(torch.float32) for Sb0_i, Sb1_i in zip(Sb0, Sb1))
#             g11 = sum((Sb1_i.flatten() @ Sb1_i.flatten()).to(torch.float32) for Sb1_i in Sb1)
#             r0  = sum((v_i.flatten() @ Sb0_i.flatten()).to(torch.float32) for v_i, Sb0_i in zip(v, Sb0))
#             r1  = sum((v_i.flatten() @ Sb1_i.flatten()).to(torch.float32) for v_i, Sb1_i in zip(v, Sb1))

#             # Solve small SPD system safely
#             det = (g00 * g11 - g01 * g01).clamp_min(1e-20)
#             x0 = ( r0 * g11 - r1 * g01) / det
#             x1 = (-r0 * g01 + r1 * g00) / det

#             uaR, ucR = tuple(x0 * b0_i + x1 * b1_i for b0_i, b1_i in zip(b0, b1))
#             return (uaR.to(Q.dtype), ucR)
        
#         u = solve_adj_gmres2((grad_aR, grad_cR), lam=0.00)

#         # r = (grad_aR.clone(), grad_cR.clone())
#         # u = (torch.zeros_like(grad_aR), torch.zeros_like(grad_cR))
#         # for _ in range(1, 10):
#         #     _, r = torch.autograd.functional.vjp(aR_cR_from_aR_cR, (aR_star, cR_star), v=r, create_graph=False)
#         #     u = tuple(a + b for a, b in zip(u, r))

#         def aR_cR_from_QK(Q_in, K_in):
#             bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR_star, K_in)
#             z = bR.to(torch.float32) * (1.0 / (cR_star + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=-1, keepdim=True)
#             R = torch.softmax(z, dim=-1).to(Q_in.dtype)
#             aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K_in)
#             logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
#             cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)

#             bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q_in)
#             L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
#             L = torch.softmax(L, dim=-1).to(Q_in.dtype)
#             L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=k)

#             aR_out = torch.einsum("bhafjki,baijhd->bafkjhd", L, Q_in)
#             cR_out = L.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).transpose(-2, -3) # (b, h, a, f, k, j, 1)
#             return aR_out, cR_out

#         _, (grad_Q_arcr, grad_K_arcr) = torch.autograd.functional.vjp(
#             aR_cR_from_QK, (Q, K), v=u, create_graph=False, strict=True
#         )

#         grad_Q += grad_Q_arcr
#         grad_K += grad_K_arcr

#         grad_Q = grad_Q * ctx.sm_scale_sqrt
#         grad_K = grad_K * ctx.sm_scale_sqrt

#         return grad_Q, grad_K, grad_V, None, None, None

class MonarchAttnImplicitFn(torch.autograd.Function):
    def forward(ctx, Q, K, V, sm_scale, num_iters, eps):
        b, a, i, j, h, _ = Q.shape
        block_b1, block_b2 = i, j
        f = K.shape[-5]

        sm_scale_sqrt = sm_scale ** 0.5
        Q = Q * sm_scale_sqrt
        K = K * sm_scale_sqrt

        aR = Q.clone().unsqueeze(-5).expand(-1, -1, f, -1, -1, -1, -1) # (b, a, f, k, j, h, d)
        cR = torch.ones((b, h, a, f, block_b1, j, 1), device=Q.device, dtype=Q.dtype) # (b, h, a, f, k, j, 1)

        with torch.no_grad():
            for _ in range(num_iters - 1):
                bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
                z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
                del bR
                z = z - z.amax(dim=-1, keepdim=True)
                R = torch.softmax(z, dim=-1).to(Q.dtype)
                aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
                logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
                cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
                del z, R, logz

                bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
                del aL
                L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
                del bL, cL
                L = torch.softmax(L, dim=-1).to(Q.dtype)
                L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)

                aR = torch.einsum("bhafjki,baijhd->bafkjhd", L, Q) 
                cR = L.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).transpose(-2, -3) # (b, h, a, f, k, j, 1)
                del L
        
        bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K)
        del aR
        z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
        del bR, cR
        z = z - z.amax(dim=-1, keepdim=True)
        R = torch.softmax(z, dim=-1).to(Q.dtype)
        aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
        logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
        cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
        del z, logz
        Y = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V)

        bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
        del aL
        L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
        del bL, cL
        L = torch.softmax(L, dim=-1).to(Q.dtype)
        L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)

        out = torch.einsum("bhafjki,bafjkhd->baijhd", L, Y)
        del Y

        # ctx.save_for_backward(L, R, Q, K, V)
        ctx.save_for_backward(Q, K, V)
        ctx.eps = eps
        ctx.sm_scale_sqrt = sm_scale_sqrt

        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V = ctx.saved_tensors
        eps = ctx.eps
        b, a, i, j, h, _ = Q.shape
        block_b1, block_b2 = i, j
        f = K.size(-5)

        def O_from_QKV(Q_in, K_in, V_in):
            aR = Q_in.clone().unsqueeze(-5).expand(-1, -1, f, -1, -1, -1, -1) # (b, a, f, k, j, h, d)
            cR = torch.ones((b, h, a, f, block_b1, j, 1), device=Q_in.device, dtype=Q_in.dtype) # (b, h, a, f, k, j, 1)

            bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR, K_in)
            z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
            z = z - z.amax(dim=-1, keepdim=True)
            R = torch.softmax(z, dim=-1).to(Q_in.dtype)
            aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K_in)
            logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
            cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
            Y = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V_in)

            bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q_in)
            L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
            L = torch.softmax(L, dim=-1).to(Q_in.dtype)
            L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)

            out = torch.einsum("bhafjki,bafjkhd->baijhd", L, Y)
            return out

        _, (grad_Q, grad_K, grad_V) = torch.autograd.functional.vjp(O_from_QKV, (Q, K, V), v=grad_out, create_graph=False, strict=True)
        grad_Q = grad_Q * ctx.sm_scale_sqrt
        grad_K = grad_K * ctx.sm_scale_sqrt
        return grad_Q, grad_K, grad_V, None, None, None

def low_rank_project(M, rank):
    U, S, Vt = torch.linalg.svd(M.float())
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    return U, Vt

def low_rank_project_faster(M, rank):
    # batched svd is very slow, flatten the first n-2 dims and run it per slice
    orig_shape = M.shape[:-2]
    M_flat = M.reshape(-1, M.size(-2), M.size(-1))
    all_U = []
    all_Vt = []
    batch_size = 32
    for i in range(0, M_flat.size(0), batch_size):
        U, Vt = low_rank_project(M_flat[i:i+batch_size], rank)
        all_U.append(U)
        all_Vt.append(Vt)
    U = torch.cat(all_U, dim=0).reshape(*orig_shape, M.size(-2), rank)
    Vt = torch.cat(all_Vt, dim=0).reshape(*orig_shape, rank, M.size(-1))
    return U, Vt

def monarch_attn_exact_decomp(Q, K, V, sm_scale):
    attn = torch.einsum("baijhd,bfklhd->bhaijfkl", Q, K)
    attn = (attn * sm_scale).flatten(-3).softmax(dim=-1)
    attn = rearrange(attn, "b h a i j (f k l) -> b h a f j k i l", f=K.size(-5), k=K.size(-4), l=K.size(-3))
    L, R = low_rank_project_faster(attn, rank=1)
    L = L.squeeze(-1)
    R = R.squeeze(-2)
    attn = torch.einsum("bhafjki,bhafjkl->bhafijkl", L, R)
    out = torch.einsum("bhafijkl,bfklhd->baijhd", attn.to(V.dtype), V)
    return out

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


# class MonarchAttnImplicitFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, Q, K, V, sm_scale, num_iters, eps):
#         b, a, i, j, h, d = Q.shape
#         block_b1, block_b2 = i, j
#         k, l = block_b1, block_b2
#         f = K.size(-5)

#         sm_scale_sqrt = sm_scale ** 0.5
#         Q = Q * sm_scale_sqrt
#         K = K * sm_scale_sqrt

#         aR = Q.sum(-5)
#         cR = torch.full((b, h, 1, k, j, 1), fill_value=a, device=Q.device, dtype=torch.float32)

#         with torch.no_grad():
#             for _ in range(num_iters - 1):
#                 bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR, K)
#                 z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#                 z = z - z.amax(dim=(-1, -4), keepdim=True)
#                 R = rearrange(z, 'b h f k j l -> b h k j (f l)')
#                 R = torch.softmax(R, dim=-1).to(Q.dtype)
#                 R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
#                 aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K)
#                 logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
#                 cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)
                
#                 bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q)
#                 L = torch.softmax(bL - cL, dim=-2).to(Q.dtype)
#                 aR = torch.einsum("bhajki,baijhd->bkjhd", L, Q)
#                 cR = L.sum(dim=(-1, -4), dtype=torch.float32, keepdim=True).transpose(-2, -3) # (b, h, 1, k, j, 1)
            
#         bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR, K)
#         z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#         z = z - z.amax(dim=(-1, -4), keepdim=True)
#         R = rearrange(z, 'b h f k j l -> b h k j (f l)')
#         R = torch.softmax(R, dim=-1).to(Q.dtype)
#         R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
#         aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K)
#         logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
#         cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)
#         Y = torch.einsum("bhfkjl,bfklhd->bkjhd", R, V)

#         bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q)
#         L = torch.softmax(bL - cL, dim=-2).to(Q.dtype)
#         out = torch.einsum("bhajki,bkjhd->baijhd", L, Y)

#         ctx.save_for_backward(aR, cR, Q, K, V)
#         ctx.eps = eps
#         ctx.sm_scale_sqrt = sm_scale_sqrt
        
#         return out

#     @staticmethod
#     def backward(ctx, grad_out):
#         aR_star, cR_star, Q, K, V = ctx.saved_tensors
#         eps = ctx.eps

#         b, a, i, j, h, d = Q.shape
#         block_b1, block_b2 = i, j
#         k, l = block_b1, block_b2
#         f = K.shape[-5]

#         def O_from_QKV_aR_cR(Q_in, K_in, V_in, aR_in, cR_in):
#             bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR_in, K_in)
#             z = bR.to(torch.float32) * (1.0 / (cR_in + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=(-1, -4), keepdim=True)
#             R = rearrange(z, 'b h f k j l -> b h k j (f l)')
#             R = torch.softmax(R, dim=-1).to(Q_in.dtype)
#             R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
#             aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K_in)
#             logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
#             cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)
#             Y = torch.einsum("bhfkjl,bfklhd->bkjhd", R, V_in)

#             bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q_in)
#             L = torch.softmax(bL - cL, dim=-2).to(Q_in.dtype)
#             out = torch.einsum("bhajki,bkjhd->baijhd", L, Y)
#             return out

#         _, (grad_Q, grad_K, grad_V, grad_aR, grad_cR) = torch.autograd.functional.vjp(O_from_QKV_aR_cR, (Q, K, V, aR_star, cR_star), v=grad_out, create_graph=False, strict=True)

#         def aR_cR_from_aR_cR(aR_in, cR_in):
#             bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR_in, K)
#             z = bR.to(torch.float32) * (1.0 / (cR_in + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=(-1, -4), keepdim=True)
#             R = rearrange(z, 'b h f k j l -> b h k j (f l)')
#             R = torch.softmax(R, dim=-1).to(Q.dtype)
#             R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
#             aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K)
#             logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
#             cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)

#             bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q)
#             L = torch.softmax(bL - cL, dim=-2).to(Q.dtype)

#             aR_out = torch.einsum("bhajki,baijhd->bkjhd", L, Q)
#             cR_out = L.sum(dim=(-1, -4), dtype=torch.float32, keepdim=True).transpose(-2, -3) # (b, h, 1, k, j, 1)
#             return aR_out, cR_out
        
#         def solve_adj_gmres2(v, lam=0.0):
#             b0 = v
#             _, Ab0 = torch.autograd.functional.vjp(aR_cR_from_aR_cR, (aR_star, cR_star), v=b0, create_graph=False)
#             b1 = Ab0

#             Sb0 = tuple((1.0 + lam) * b0_i - Ab0_i for b0_i, Ab0_i in zip(b0, Ab0)) # (I - A + λI) b0
#             _, Ab1 = torch.autograd.functional.vjp(aR_cR_from_aR_cR, (aR_star, cR_star), v=b1, create_graph=False)
#             Sb1 = tuple((1.0 + lam) * b1_i - Ab1_i for b1_i, Ab1_i in zip(b1, Ab1)) # (I - A + λI) b1

#             # Build 2×2 normal equations G x = rhs
#             g00 = sum((Sb0_i.flatten() @ Sb0_i.flatten()).to(torch.float32) for Sb0_i in Sb0)
#             g01 = sum((Sb0_i.flatten() @ Sb1_i.flatten()).to(torch.float32) for Sb0_i, Sb1_i in zip(Sb0, Sb1))
#             g11 = sum((Sb1_i.flatten() @ Sb1_i.flatten()).to(torch.float32) for Sb1_i in Sb1)
#             r0  = sum((v_i.flatten() @ Sb0_i.flatten()).to(torch.float32) for v_i, Sb0_i in zip(v, Sb0))
#             r1  = sum((v_i.flatten() @ Sb1_i.flatten()).to(torch.float32) for v_i, Sb1_i in zip(v, Sb1))

#             # Solve small SPD system safely
#             det = (g00 * g11 - g01 * g01).clamp_min(1e-20)
#             x0 = ( r0 * g11 - r1 * g01) / det
#             x1 = (-r0 * g01 + r1 * g00) / det

#             uaR, ucR = tuple(x0 * b0_i + x1 * b1_i for b0_i, b1_i in zip(b0, b1))
#             return (uaR.to(Q.dtype), ucR)
        
#         u = solve_adj_gmres2((grad_aR, grad_cR), lam=0.00)

#         # r = (grad_aR.clone(), grad_cR.clone())
#         # u = r
#         # for _ in range(1, 1):
#         #     _, r = torch.autograd.functional.vjp(aR_cR_from_aR_cR, (aR_star, cR_star), v=r, create_graph=False)
#         #     u = tuple(a + b for a, b in zip(u, r))

#         def aR_cR_from_QK(Q_in, K_in):
#             bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR_star, K_in)
#             z = bR.to(torch.float32) * (1.0 / (cR_star + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=(-1, -4), keepdim=True)
#             R = rearrange(z, 'b h f k j l -> b h k j (f l)')
#             R = torch.softmax(R, dim=-1).to(Q_in.dtype)
#             R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
#             aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K_in)
#             logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
#             cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)

#             bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q_in)
#             L = torch.softmax(bL - cL, dim=-2).to(Q_in.dtype)

#             aR_out = torch.einsum("bhajki,baijhd->bkjhd", L, Q_in)
#             cR_out = L.sum(dim=(-1, -4), dtype=torch.float32, keepdim=True).transpose(-2, -3) # (b, h, 1, k, j, 1)
#             return aR_out, cR_out

#         _, (grad_Q_arcr, grad_K_arcr) = torch.autograd.functional.vjp(
#             aR_cR_from_QK, (Q, K), v=u, create_graph=False, strict=True
#         )

#         grad_Q += grad_Q_arcr
#         grad_K += grad_K_arcr

#         grad_Q = grad_Q * ctx.sm_scale_sqrt
#         grad_K = grad_K * ctx.sm_scale_sqrt

#         return grad_Q, grad_K, grad_V, None, None, None

# monarch_attn = MonarchAttnImplicitFn.apply

# def monarch_attn(Q, K, V, sm_scale, num_iters, eps):
#     b, a, i, j, h, d = Q.shape
#     block_b1, block_b2 = i, j
#     k, l = block_b1, block_b2
#     f = K.size(-5)

#     sm_scale_sqrt = sm_scale ** 0.5
#     Q = Q * sm_scale_sqrt
#     K = K * sm_scale_sqrt

#     aR = Q.sum(-5)
#     cR = torch.full((b, h, 1, k, j, 1), fill_value=a, device=Q.device, dtype=torch.float32)

#     with torch.no_grad():
#         for _ in range(num_iters - 1):
#             bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR, K)
#             z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#             z = z - z.amax(dim=(-1, -4), keepdim=True)
#             R = rearrange(z, 'b h f k j l -> b h k j (f l)')
#             R = torch.softmax(R, dim=-1).to(Q.dtype)
#             R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
#             aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K)
#             logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
#             cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)
            
#             bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q)
#             L = torch.softmax(bL - cL, dim=-2).to(Q.dtype)
#             aR = torch.einsum("bhajki,baijhd->bkjhd", L, Q)
#             cR = L.sum(dim=(-1, -4), dtype=torch.float32, keepdim=True).transpose(-2, -3) # (b, h, 1, k, j, 1)
        
#     bR = torch.einsum("bkjhd,bfklhd->bhfkjl", aR, K)
#     z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
#     z = z - z.amax(dim=(-1, -4), keepdim=True)
#     R = rearrange(z, 'b h f k j l -> b h k j (f l)')
#     R = torch.softmax(R, dim=-1).to(Q.dtype)
#     R = rearrange(R, 'b h k j (f l) -> b h f k j l', f=f, l=block_b2)
#     aL = torch.einsum("bhfkjl,bfklhd->bjkhd", R, K)
#     logz = torch.logsumexp(z, dim=(-1, -4), keepdim=True) # (b, h, 1, k, j, 1)
#     cL = (R * (z - logz)).sum(dim=(-1, -4), keepdim=True).transpose(-2, -3) # (b, h, 1, j, k, 1)
#     Y = torch.einsum("bhfkjl,bfklhd->bkjhd", R, V)

#     bL = torch.einsum("bjkhd,baijhd->bhajki", aL, Q)
#     L = torch.softmax(bL - cL, dim=-2).to(Q.dtype)
#     out = torch.einsum("bhajki,bkjhd->baijhd", L, Y)
#     return out

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
                 eps=1e-6,
                 block_num=None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = total_seq_len if local_attn_size == -1 else local_attn_size * (frame_height * frame_width)

        self.num_iters = int(os.getenv("MONARCH_ATTN_NUM_ITERS", "1"))
        self.use_dense_init = bool(int(os.getenv("USE_DENSE_INIT", "0")))
        self.h_reduce = int(os.getenv("MONARCH_ATTN_H_REDUCE", "1"))
        self.w_reduce = int(os.getenv("MONARCH_ATTN_W_REDUCE", "1"))
        self.init_h_reduce = os.getenv("MONARCH_ATTN_INIT_H_REDUCE")
        if self.init_h_reduce is not None:
            self.init_h_reduce = int(self.init_h_reduce)
        self.init_w_reduce = os.getenv("MONARCH_ATTN_INIT_W_REDUCE")
        if self.init_w_reduce is not None:
            self.init_w_reduce = int(self.init_w_reduce)
        self.use_framewise = bool(int(os.getenv("MONARCH_ATTN_FRAMEWISE", "0")))
        self.disable_monarch = bool(int(os.getenv("DISABLE_MONARCH_ATTN", "0")))
        self.exact_monarch = bool(int(os.getenv("MONARCH_ATTN_EXACT", "0")))
        layer_disable_list = os.getenv("MONARCH_ATTN_DISABLE_LAYERS")
        self.topk = os.getenv("ATTN_TOPK_PCT")
        self.use_hacks = bool(int(os.getenv("USE_HACKS", "0")))
        if self.topk is not None:
            self.topk = float(self.topk)
        if layer_disable_list is not None:
            assert block_num is not None
            layer_disable_list = [int(x) for x in layer_disable_list.split(",")]
            if block_num in layer_disable_list:
                self.disable_monarch = True
        self.use_svg = bool(int(os.getenv("USE_SVG", "0")))
        self.use_radial_attn = bool(int(os.getenv("USE_RADIAL_ATTN", "0")))
        if self.use_radial_attn:
            self.mask_map = None
        self.block_num = block_num

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        timestep=None,
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
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                assert False, "teacher forcing unsupported for now"
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

                if self.disable_monarch:
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
                    b, _, h, d = roped_query.shape
                    x = monarch_attn(
                        roped_query,
                        roped_key, v,
                        1 if self.use_framewise else 3,
                        self.h_reduce,
                        self.w_reduce,
                        frame_height,
                        frame_width,
                        self.num_iters,
                        block_causal_size=3*frame_height*frame_width,
                    )
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
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
            # kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            # kv_cache["v"][:, local_start_index:local_end_index] = v
            
            cache_start = max(0, local_end_index - self.max_attention_size)
            curr_k = kv_cache["k"][:, cache_start:local_end_index]
            curr_v = kv_cache["v"][:, cache_start:local_end_index]
            local_start_index -= cache_start
            local_end_index -= cache_start

            is_init = (self.use_dense_init and curr_k.size(1) == (3 * grid_sizes[0, 1].item() * grid_sizes[0, 2].item()))
            if (self.disable_monarch and not self.use_svg and not self.use_radial_attn) or is_init:
                if not is_init and self.topk is not None:
                    curr_k[:, local_start_index:local_end_index] = roped_key
                    curr_v[:, local_start_index:local_end_index] = v
                    qk = torch.einsum('bihd,bjhd->bhij', roped_query, curr_k) * (d ** -0.5)
                    _, bottomk = qk.topk(dim=-1, k=int((1 - self.topk) * qk.size(-1)), largest=False)
                    qk.scatter_(-1, bottomk, -torch.inf)
                    attn = torch.softmax(qk, dim=-1)
                    x = torch.einsum('bhij,bjhd->bihd', attn, curr_v)
                else:
                    # x = full_attention_with_kv_cache(
                    #     roped_query,
                    #     curr_k,
                    #     curr_v,
                    #     roped_key,
                    #     v,
                    #     local_start_index,
                    #     local_end_index,
                    #     grad_only_new_kv=not (kv_cache["k"].requires_grad or kv_cache["v"].requires_grad)
                    # )
                    curr_k[:, local_start_index:local_end_index] = roped_key
                    curr_v[:, local_start_index:local_end_index] = v
                    x = attention(
                        roped_query,
                        curr_k,
                        curr_v,
                    )
            elif self.use_svg:
                curr_k[:, local_start_index:local_end_index] = roped_key
                curr_v[:, local_start_index:local_end_index] = v
                target_seq_len = total_seq_len # curr_k.size(1)
                if WanAttn_SVGAttn_Processor2_0.curr_seq_len != target_seq_len:
                    sample_mse_max_row = target_seq_len
                    WanAttn_SVGAttn_Processor2_0.num_sampled_rows = 64
                    WanAttn_SVGAttn_Processor2_0.sample_mse_max_row = sample_mse_max_row
                    num_frame_patches = target_seq_len // (frame_height * frame_width)
                    frame_patches_one_frame = frame_height * frame_width
                    masks = ["spatial", "temporal"]
                    WanAttn_SVGAttn_Processor2_0.attention_masks = [
                        get_attention_mask(
                            mask_name, sample_mse_max_row, 0, num_frame_patches, frame_patches_one_frame
                        )
                        for mask_name in masks
                    ]
                    WanAttn_SVGAttn_Processor2_0.first_layers_fp = 0.025 if self.use_hacks else 0
                    WanAttn_SVGAttn_Processor2_0.first_times_fp = 0.075 if self.use_hacks else 0

                    multiplier = diag_width = sparsity_to_width(
                        0.15, 0, num_frame_patches, frame_patches_one_frame
                    )
                    WanAttn_SVGAttn_Processor2_0.context_length = 0
                    WanAttn_SVGAttn_Processor2_0.num_frame = num_frame_patches
                    WanAttn_SVGAttn_Processor2_0.frame_size = frame_patches_one_frame
                    WanAttn_SVGAttn_Processor2_0.block_mask = prepare_flexattention(
                        1,
                        12,
                        128,
                        q.dtype,
                        q.device,
                        0,
                        0,
                        num_frame_patches,
                        frame_patches_one_frame,
                        diag_width,
                        multiplier
                    )
                    WanAttn_SVGAttn_Processor2_0.curr_seq_len = target_seq_len

                # padded_q = torch.nn.functional.pad(
                #     roped_query, (0, 0, 0, 0, curr_k.size(1) - roped_query.size(1), 0), value=0.0
                # )
                # assert padded_q.shape == curr_k.shape

                roped_query = roped_query.transpose(1, 2).contiguous()
                curr_k = curr_k.transpose(1, 2).contiguous()
                curr_v = curr_v.transpose(1, 2).contiguous()
                
                padded_q = torch.nn.functional.pad(
                    roped_query, (0, 0, curr_k.size(2) - roped_query.size(2), total_seq_len - curr_k.size(2)), value=0.0
                )
                padded_k = torch.nn.functional.pad(
                    curr_k, (0, 0, 0, total_seq_len - curr_k.size(2)), value=0.0
                )
                padded_v = torch.nn.functional.pad(
                    curr_v, (0, 0, 0, total_seq_len - curr_v.size(2)), value=0.0
                )
                assert padded_q.shape == padded_k.shape
                WanAttn_SVGAttn_Processor2_0.sample_mse_min_row = curr_k.size(2) - roped_query.size(2)
                WanAttn_SVGAttn_Processor2_0.sample_mse_max_row = curr_k.size(2)

                # padded_q = padded_q.transpose(1, 2).contiguous()
                # curr_k = curr_k.transpose(1, 2).contiguous()
                # curr_v = curr_v.transpose(1, 2).contiguous()
                x = WanAttn_SVGAttn_Processor2_0.attention_core_logic(
                    padded_q,
                    padded_k,
                    padded_v,
                    layer_idx=self.block_num,
                    timestep=timestep,
                ).transpose(1, 2)
                # x = x[:, -roped_query.size(1):, :, :]
                x = x[:, curr_k.size(2) - roped_query.size(2) : curr_k.size(2), :, :]
                # assert x.shape == roped_query.shape
            elif self.use_radial_attn:
                curr_k[:, local_start_index:local_end_index] = roped_key
                curr_v[:, local_start_index:local_end_index] = v
                # self.mask_map = MaskMap(video_token_num=curr_k.size(1), num_frame=(curr_k.size(1) // (frame_height * frame_width)))
                # padded_q = torch.nn.functional.pad(
                #     roped_query, (0, 0, 0, 0, curr_k.size(1) - roped_query.size(1), 0), value=0.0
                # )
                # assert padded_q.shape == curr_k.shape
                # x = RadialAttention(
                #     padded_q, curr_k, curr_v, self.mask_map, sparsity_type="radial", block_size=1, decay_factor=0.0, model_type="wan", pre_defined_mask=None, use_sage_attention=False
                # )
                # x = rearrange(x, 'b s (h d) -> b s h d', d=d)
                # x = x[:, -roped_query.size(1):, :, :]
                # assert x.shape == roped_query.shape

                if (timestep == 1000 or self.block_num < 1) and self.use_hacks:
                    x = attention(
                        roped_query,
                        curr_k,
                        curr_v
                    )
                else:
                    if self.mask_map is None:
                        self.mask_map = MaskMap(video_token_num=total_seq_len, num_frame=21)
                    padded_q = torch.nn.functional.pad(
                        roped_query, (0, 0, 0, 0, curr_k.size(1) - roped_query.size(1), total_seq_len - curr_k.size(1)), value=0.0
                    )
                    padded_k = torch.nn.functional.pad(
                        curr_k, (0, 0, 0, 0, 0, total_seq_len - curr_k.size(1)), value=0.0
                    )
                    padded_v = torch.nn.functional.pad(
                        curr_v, (0, 0, 0, 0, 0, total_seq_len - curr_v.size(1)), value=0.0
                    )
                    assert padded_q.shape == padded_k.shape
                    x = RadialAttention(
                        padded_q, padded_k, padded_v, self.mask_map, sparsity_type="radial", block_size=1, decay_factor=0.0, model_type="wan", pre_defined_mask=None, use_sage_attention=False
                    )
                    x = rearrange(x, 'b s (h d) -> b s h d', d=d)
                    x = x[:, curr_k.size(1) - roped_query.size(1) : curr_k.size(1), :, :]
                    assert x.shape == roped_query.shape
            else:
                x = monarch_attn_with_kv_cache(
                    roped_query,
                    curr_k,
                    curr_v,
                    roped_key,
                    v,
                    local_start_index,
                    local_end_index,
                    1 if self.use_framewise else 3,
                    self.h_reduce,
                    self.w_reduce,
                    frame_height,
                    frame_width,
                    num_iters=self.num_iters,
                    grad_only_new_kv=not (kv_cache["k"].requires_grad or kv_cache["v"].requires_grad)
                )

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
                 eps=1e-6,
                 block_num=None):
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
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps, block_num)
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
        cache_start=None,
        timestep=None,
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
            freqs, block_mask, kv_cache, current_start, cache_start, timestep)

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
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps, i)
            for i in range(num_layers)
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
        
        timestep = t.flatten()[0].item()
        assert (t == timestep).all()

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # if block_index == 0 and dist.get_rank() == 0:
                #     print("gradient checkpointing, kv grad enabled", kv_cache[block_index]["k"].requires_grad)
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "timestep": timestep,
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
                        "cache_start": cache_start,
                        "timestep": timestep,
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
