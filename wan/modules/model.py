# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat, rearrange

from .attention import flash_attention

__all__ = ['WanModel']

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
        z = bR.to(torch.float32) * (1.0 / (cR + eps)).clamp_max(1e4)
        del bR
        z = z - z.amax(dim=-1, keepdim=True)
        R = torch.softmax(z, dim=-1).to(Q.dtype)
        aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
        logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
        cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)
        del z, logz
        Y = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, V)
        del R

        bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
        del aL
        L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
        del bL, cL
        L = torch.softmax(L, dim=-1).to(Q.dtype)
        L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)

        out = torch.einsum("bhafjki,bafjkhd->baijhd", L, Y)
        del L, Y

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
            bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR_in, K_in)
            z = bR.to(torch.float32) * (1.0 / (cR_in + eps)).clamp_max(1e4)
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

        _, (grad_Q, grad_K, grad_V, grad_aR, grad_cR) = torch.autograd.functional.vjp(O_from_QKV_aR_cR, (Q, K, V, aR_star, cR_star), v=grad_out, create_graph=False, strict=True)

        def aR_cR_from_aR_cR(aR_in, cR_in):
            bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR_in, K)
            z = bR.to(torch.float32) * (1.0 / (cR_in + eps)).clamp_max(1e4)
            z = z - z.amax(dim=-1, keepdim=True)
            R = torch.softmax(z, dim=-1).to(Q.dtype)
            aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K)
            logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
            cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)

            bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q)
            L = torch.softmax(bL - cL, dim=-2).to(Q.dtype)
            L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
            L = torch.softmax(L, dim=-1).to(Q.dtype)
            L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)

            aR_out = torch.einsum("bhafjki,baijhd->bafkjhd", L, Q)
            cR_out = L.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).transpose(-2, -3) # (b, h, a, f, k, j, 1)
            return aR_out, cR_out
        
        # def solve_adj_gmres2(v, lam=0.0):
        #     v = torch.cat([v_i.flatten() for v_i in v])
        #     aRcR_star = torch.cat([aR_star.flatten(), cR_star.flatten()])

        #     b0 = v
        #     _, Ab0 = torch.autograd.functional.vjp(aR_cR_from_aR_cR, aRcR_star, v=b0, create_graph=False)
        #     b1 = Ab0

        #     Sb0 = (1.0 + lam) * b0 - Ab0 # (I - A + λI) b0
        #     _, Ab1 = torch.autograd.functional.vjp(aR_cR_from_aR_cR, aRcR_star, v=b1, create_graph=False)
        #     Sb1 = (1.0 + lam) * b1 - Ab1 # (I - A + λI) b1

        #     # Build 2×2 normal equations G x = rhs
        #     g00 = (Sb0.flatten() @ Sb0.flatten())
        #     g01 = (Sb0.flatten() @ Sb1.flatten())
        #     g11 = (Sb1.flatten() @ Sb1.flatten())
        #     r0  = (v.flatten()  @ Sb0.flatten())
        #     r1  = (v.flatten()  @ Sb1.flatten())

        #     # Solve small SPD system safely
        #     det = (g00 * g11 - g01 * g01).clamp_min(1e-20)
        #     x0 = ( r0 * g11 - r1 * g01) / det
        #     x1 = (-r0 * g01 + r1 * g00) / det

        #     u = x0 * b0 + x1 * b1
        #     u1 = u[:b*a*f*k*j*h*d].view(b, a, f, k, j, h, d)
        #     u2 = u[b*a*f*k*j*h*d:].view(b, h, a, f, k, j, 1)
        #     return (u1, u2)
        
        # u = solve_adj_gmres2((grad_aR, grad_cR), lam=0.00)

        r = (grad_aR.clone(), grad_cR.clone())
        u = (torch.zeros_like(grad_aR), torch.zeros_like(grad_cR))
        for _ in range(1):
            _, r = torch.autograd.functional.vjp(aR_cR_from_aR_cR, (aR_star, cR_star), v=r, create_graph=False)
            u = tuple(a + b for a, b in zip(u, r))

        def aR_cR_from_QK(Q_in, K_in):
            bR = torch.einsum("bafkjhd,bfklhd->bhafkjl", aR_star, K_in)
            z = bR.to(torch.float32) * (1.0 / (cR_star + eps)).clamp_max(1e4)
            z = z - z.amax(dim=-1, keepdim=True)
            R = torch.softmax(z, dim=-1).to(Q_in.dtype)
            aL = torch.einsum("bhafkjl,bfklhd->bafjkhd", R, K_in)
            logz = torch.logsumexp(z, dim=-1, keepdim=True) # (b, h, a, f, k, j, 1)
            cL = (R * (z - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3) # (b, h, a, f, j, k, 1)

            bL = torch.einsum("bafjkhd,baijhd->bhafjki", aL, Q_in)
            L = rearrange(bL - cL, "b h a f j k i -> b h a j i (f k)")
            L = torch.softmax(L, dim=-1).to(Q_in.dtype)
            L = rearrange(L, "b h a j i (f k) -> b h a f j k i", f=f, k=block_b1)

            aR_out = torch.einsum("bhafjki,baijhd->bafkjhd", L, Q_in)
            cR_out = L.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).transpose(-2, -3) # (b, h, a, f, k, j, 1)
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

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
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
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
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


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.min_block_size = int(os.getenv("MONARCH_ATTN_MIN_BLOCK_SIZE", "1"))
        self.target_sparsity = os.getenv("MONARCH_ATTN_TARGET_SPARSITY")
        if self.target_sparsity is not None:
            self.target_sparsity = float(self.target_sparsity)
        self.num_iters = int(os.getenv("MONARCH_ATTN_NUM_ITERS", "1"))
        self.disable_monarch = bool(int(os.getenv("DISABLE_MONARCH_ATTN", "0")))

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def get_block_sizes(self, seq_len, h, w):
        # return (h, w)
        if self.target_sparsity is None:
            return (h, w) # max sparsity
        factors = [i for i in range(self.min_block_size, h + 1) if h % i == 0]
        assert len(factors) > 0, f"Cannot find usable block sizes with min block size {self.min_block_size}"
        sparsities = [1 - (f*f*w + w*w*f)/(f*f*w*w) for f in factors]
        dists = [abs(s - self.target_sparsity) for s in sparsities]
        min_idx = dists.index(min(dists))
        return (factors[min_idx], w)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if self.disable_monarch:
            x = flash_attention(
                q=rope_apply(q, grid_sizes, freqs),
                k=rope_apply(k, grid_sizes, freqs),
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)
        else:
            block_b1, block_b2 = self.get_block_sizes(q.size(1), grid_sizes[0, 1].item(), grid_sizes[0, 2].item())
            b, s, h, d = q.shape
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
            q = q.view(b, -1, block_b1, block_b2, h, d)
            k = k.view(b, -1, block_b1, block_b2, h, d)
            v = v.view(b, -1, block_b1, block_b2, h, d)
            x = monarch_attn(q, k, v, d ** -0.5, self.num_iters, self.eps).reshape(b, s, h, d)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanGanCrossAttention(WanSelfAttention):

    def forward(self, x, context, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)

        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)

        # compute attention
        x = flash_attention(qq, kk, vv)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(
            dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
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
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class GanAttentionBlock(nn.Module):

    def __init__(self,
                 dim=1536,
                 ffn_dim=8192,
                 num_heads=12,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        # self.norm1 = WanLayerNorm(dim, eps)
        # self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
        #   eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.cross_attn = WanGanCrossAttention(dim, num_heads,
                                               (-1, -1),
                                               qk_norm,
                                               eps)

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        context,
        # seq_lens,
        # grid_sizes,
        # freqs,
        # context,
        # context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        # e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # # self-attention
        # y = self.self_attn(
        #     self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
        #     freqs)
        # # with amp.autocast(dtype=torch.float32):
        # x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context):
            token = context + self.cross_attn(self.norm3(x), context)
            y = self.ffn(self.norm2(token)) + token  # * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            # x = x + y * e[5]
            return y

        x = cross_attn_ffn(x, context)
        return x


class Head(nn.Module):

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
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class RegisterTokens(nn.Module):
    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim, eps=1e-6)

    def forward(self):
        return self.rms_norm(self.register_tokens)

    def reset_parameters(self):
        nn.init.normal_(self.register_tokens, std=0.02)


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
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
                 window_size=(-1, -1),
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
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
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
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = 21

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
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

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

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        *args,
        **kwargs
    ):
        # if kwargs.get('classify_mode', False) is True:
        # kwargs.pop('classify_mode')
        # return self._forward_classify(*args, **kwargs)
        # else:
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
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

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert torch.all(seq_lens == seq_lens[0])
        assert seq_lens.max() <= seq_len
        # x = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
        #               dim=1) for u in x
        # ])
        x = torch.cat(x)

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
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
            context_lens=context_lens)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        final_x = None
        if classify_mode:
            assert register_tokens is not None
            assert gan_ca_blocks is not None
            assert cls_pred_branch is not None

            final_x = []
            registers = repeat(register_tokens(), "n d -> b n d", b=x.shape[0])
            # x = torch.cat([registers, x], dim=1)

        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

            if classify_mode and ii in [13, 21, 29]:
                gan_token = registers[:, gan_idx: gan_idx + 1]
                final_x.append(gan_ca_blocks[gan_idx](x, gan_token))
                gan_idx += 1

        if classify_mode:
            final_x = torch.cat(final_x, dim=1)
            if concat_time_embeddings:
                final_x = cls_pred_branch(torch.cat([final_x, 10 * e[:, None, :]], dim=1).view(final_x.shape[0], -1))
            else:
                final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if classify_mode:
            return torch.stack(x), final_x

        return torch.stack(x)

    def _forward_classify(
        self,
        x,
        t,
        context,
        seq_len,
        register_tokens,
        cls_pred_branch,
        clip_fea=None,
        y=None,
    ):
        r"""
        Feature extraction through the diffusion model

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
                List of video features with original input shapes [C_block, F, H / 8, W / 8]
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
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
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
            context_lens=context_lens)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        for block in self.blocks[:16]:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # unpatchify
        x = self.unpatchify(x, grid_sizes, c=self.dim // 4)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes, c=None):
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

        c = self.out_dim if c is None else c
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
