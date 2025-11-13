from types import SimpleNamespace
from einops import rearrange
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def _is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def _supports_host_descriptor():
    return _is_cuda() and torch.cuda.get_device_capability()[0] >= 9

DEVICE = triton.runtime.driver.active.get_active_torch_device()
assert triton.runtime.driver.active.get_current_target().backend == "cuda"
supports_host_descriptor = _supports_host_descriptor()

@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

def _init_al_cl_fwd_descs(Z, H, A, F, HEAD_DIM, block_b1, block_b2, aR, k, aL, cR):
    ZAF = Z * A * F
    ZFK = Z * F * block_b1
    # F has stride 0 in first iter
    ZAFK = Z * A * block_b1
    hd_stride = H * HEAD_DIM
    jhd_stride = block_b2 * hd_stride
    khd_stride = block_b1 * hd_stride
    jkhd_stride = block_b2 * khd_stride
    ZHAF = ZAF * H
    kj_stride = block_b1 * block_b2

    descs = SimpleNamespace()
    if supports_host_descriptor:
        descs.aR = TensorDescriptor(
            aR,
            shape=[ZAFK, block_b2, H, HEAD_DIM],
            strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.k = TensorDescriptor(
            k,
            shape=[ZFK, block_b2, H, HEAD_DIM],
            strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.aL = TensorDescriptor(
            aL,
            shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
            strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.cR = TensorDescriptor(
            cR,
            shape=[ZHAF, block_b1, block_b2],
            strides=[kj_stride, block_b2, 1],
            block_shape=[1, 1, 1]
        ) if (block_b2 * cR.element_size()) % 16 == 0 else cR
    else:
        descs.aR = aR
        descs.k = k
        descs.aL = aL
        descs.cR = cR

    return descs

def _modify_al_cl_fwd_descs(Z, H, A, F, HEAD_DIM, block_b1, block_b2, aR, descs):
    ZAF = Z * A * F
    ZAFK = ZAF * block_b1    
    if supports_host_descriptor:
        descs.aR.base = aR
        descs.aR.shape = [ZAFK, block_b2, H, HEAD_DIM]
    else:
        descs.aR = aR

def _al_cl_fwd_pre_hook(nargs):
    BLOCK_J = nargs["BLOCK_J"]
    BLOCK_L = nargs["BLOCK_L"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["ar_ptr"], TensorDescriptor):
        return
    nargs["ar_ptr"].block_shape = [1, BLOCK_J, 1, HEAD_DIM]
    nargs["k_ptr"].block_shape = [1, BLOCK_L, 1, HEAD_DIM]
    nargs["al_ptr"].block_shape = [1, BLOCK_J, 1, 1, HEAD_DIM]
    if isinstance(nargs["cr_ptr"], TensorDescriptor):
        nargs["cr_ptr"].block_shape = [1, 1, BLOCK_J]

configs = [
    triton.Config({'BLOCK_J': BJ, 'BLOCK_L': BL}, num_stages=s, num_warps=w, pre_hook=_al_cl_fwd_pre_hook) \
    for BJ in [64, 128]\
    for BL in [32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_J = conf.kwargs["BLOCK_J"]
    BLOCK_L = conf.kwargs["BLOCK_L"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_J * BLOCK_L < 128 * 128
                and conf.num_warps == 8)

@triton.autotune(configs=list(filter(keep, configs)), key=["block_b1", "block_b2", "HEAD_DIM", "IS_FIRST_ITER"], cache_results=True)
@triton.jit
def _al_cl_fwd(Z, H, A, F,
              ar_ptr, cr_ptr, k_ptr,
              al_ptr, cl_ptr,
              eps, max_clamp,
              sm_scale_sqrt,
              block_b1: tl.constexpr,
              block_b2: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              IS_FIRST_ITER: tl.constexpr,
              BLOCK_J: tl.constexpr,
              BLOCK_L: tl.constexpr,
              ):
    start_j = tl.program_id(0) * BLOCK_J
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_afk = tl.program_id(2)
    off_k = off_afk % block_b1
    off_f = (off_afk // block_b1) % F
    off_a = off_afk // (block_b1 * F)

    off_zaf = (off_z * A + off_a) * F + off_f
    ZAF = Z * A * F
    off_zfk = (off_z * F + off_f) * block_b1 + off_k
    ZFK = Z * F * block_b1

    if IS_FIRST_ITER:
        # F has stride 0
        off_zafk = (off_z * A + off_a) * block_b1 + off_k
        ZAFK = Z * A * block_b1
    else:
        off_zafk = off_zaf * block_b1 + off_k
        ZAFK = ZAF * block_b1

    hd_stride = H * HEAD_DIM
    jhd_stride = block_b2 * hd_stride

    desc_ar = _maybe_make_tensor_desc(
        ar_ptr,
        shape=[ZAFK, block_b2, H, HEAD_DIM],
        strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_J, 1, HEAD_DIM]
    )
    desc_k = _maybe_make_tensor_desc(
        k_ptr,
        shape=[ZFK, block_b2, H, HEAD_DIM],
        strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_L, 1, HEAD_DIM]
    )
    khd_stride = block_b1 * hd_stride
    jkhd_stride = block_b2 * khd_stride
    desc_al = _maybe_make_tensor_desc(
        al_ptr,
        shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
        strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_J, 1, 1, HEAD_DIM]
    )
    ZHAF = ZAF * H
    off_zhaf = ((off_z * H + off_h) * A + off_a) * F + off_f
    kj_stride = block_b1 * block_b2
    if not IS_FIRST_ITER:
        desc_cr = tl.make_block_ptr(
            cr_ptr,
            shape=[ZHAF, block_b1, block_b2],
            strides=[kj_stride, block_b2, 1],
            block_shape=[1, 1, BLOCK_J],
            order=[2, 1, 0],
            offsets=[0, 0, 0],
        ) if not isinstance(cr_ptr, tl.tensor_descriptor) else cr_ptr
    cl_ptrs = cl_ptr + (off_zhaf * kj_stride
                        + off_k) + (start_j + tl.arange(0, BLOCK_J)) * block_b1

    dtype = desc_ar.dtype

    sm_scale_sqrt = sm_scale_sqrt.to(dtype)
    al_acc = tl.zeros([BLOCK_J, HEAD_DIM], dtype=tl.float32)
    cl_acc = tl.zeros([BLOCK_J], dtype=tl.float32)
    l_j = tl.zeros([BLOCK_J], dtype=tl.float32)
    m_j = tl.full([BLOCK_J], dtype=tl.float32, value=float("-inf"))

    ar_j = desc_ar.load([off_zafk, start_j, off_h, 0]).reshape(BLOCK_J, HEAD_DIM)
    if IS_FIRST_ITER:
        ar_j = ar_j * sm_scale_sqrt
        cr_j_scale = tl.full([BLOCK_J], dtype=tl.float32, value=1.44269504) # log2(e)
    else:
        if isinstance(desc_cr, tl.tensor_descriptor):
            cr_j = desc_cr.load([off_zhaf, off_k, start_j]).reshape(BLOCK_J)
        else:
            cr_j = tl.load(desc_cr.advance([off_zhaf, off_k, start_j]), boundary_check=(2,)).reshape(BLOCK_J)
        cr_j_scale = tl.clamp(1.0 / (cr_j + eps), float("-inf"), max_clamp) * 1.44269504 # log2(e)
    l_range = tl.arange(0, BLOCK_L)

    for l in tl.static_range(0, block_b2, BLOCK_L):
        l_mask = l_range + l < block_b2

        k_l = desc_k.load([off_zfk, l, off_h, 0]).reshape(BLOCK_L, HEAD_DIM) * sm_scale_sqrt
        k_l = tl.where(l_mask[:, None], k_l, 0.0)
        br_jl = tl.dot(ar_j, k_l.T) * cr_j_scale[:, None]
        z_jl = tl.where(l_mask[None, :], br_jl, float("-inf"))

        m_jl = tl.maximum(m_j, tl.max(z_jl, 1))
        z_jl = z_jl - m_jl[:, None]
        p = tl.math.exp2(z_jl)
        l_jl = tl.sum(p, 1)

        alpha = tl.math.exp2(m_j - m_jl)
        al_acc = al_acc * alpha[:, None]
        cl_acc = cl_acc * alpha + tl.sum(br_jl * p, 1)

        p = p.to(dtype)
        al_acc = tl.dot(p, k_l, al_acc)

        l_j = l_j * alpha + l_jl
        m_j = m_jl

    al_acc = al_acc / l_j[:, None]
    desc_al.store([off_zaf, start_j, off_k, off_h, 0], al_acc.to(dtype).reshape(1, BLOCK_J, 1, 1, HEAD_DIM))

    lse = m_j + tl.log2(l_j)
    cl_acc = cl_acc / l_j - lse # keep cL in log2 space
    tl.store(cl_ptrs, cl_acc, start_j + tl.arange(0, BLOCK_J) < block_b2)


def _init_ar_cr_fwd_preprocess_descs(Z, H, A, F, HEAD_DIM, block_b1, block_b2, aL, q, cL, lse):
    ZA = Z * A
    HD = H * HEAD_DIM
    khd_stride = block_b1 * HD
    jkhd_stride = block_b2 * khd_stride
    fjkhd_stride = F * jkhd_stride
    jhd_stride = block_b2 * HD
    ijhd_stride = jkhd_stride
    ZHA = ZA * H
    jk_stride = block_b2 * block_b1
    fjk_stride = F * jk_stride
    ji_stride = jk_stride
    
    descs = SimpleNamespace()
    if supports_host_descriptor:
        descs.aL = TensorDescriptor(
            aL,
            shape=[ZA, F, block_b2, block_b1, HD],
            strides=[fjkhd_stride, jkhd_stride, khd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.q = TensorDescriptor(
            q,
            shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
            strides=[ijhd_stride, jhd_stride, HD, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        # TMA requires 16-byte alignment for leading strides
        descs.cL = TensorDescriptor(
            cL,
            shape=[ZHA, F, block_b2, block_b1],
            strides=[fjk_stride, jk_stride, block_b1, 1],
            block_shape=[1, 1, 1, 1]
        ) if (block_b1 * cL.element_size()) % 16 == 0 else cL
        descs.lse = TensorDescriptor(
            lse,
            shape=[ZHA, block_b2, block_b1],
            strides=[ji_stride, block_b1, 1],
            block_shape=[1, 1, 1]
        ) if (block_b1 * lse.element_size()) % 16 == 0 else lse
    else:
        descs.aL = aL
        descs.q = q
        descs.cL = cL
        descs.lse = lse

    return descs

def _ar_cr_fwd_preprocess_pre_hook(nargs):
    BLOCK_K = nargs["BLOCK_K"]
    BLOCK_I = nargs["BLOCK_I"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["al_ptr"], TensorDescriptor):
        return
    nargs["al_ptr"].block_shape = [1, 1, 1, BLOCK_K, HEAD_DIM]
    nargs["q_ptr"].block_shape = [1, BLOCK_I, 1, 1, HEAD_DIM]
    if isinstance(nargs["cl_ptr"], TensorDescriptor):
        nargs["cl_ptr"].block_shape = [1, 1, 1, BLOCK_K]
    if isinstance(nargs["lse_ptr"], TensorDescriptor):
        nargs["lse_ptr"].block_shape = [1, 1, BLOCK_I]

configs = [
    triton.Config({'BLOCK_I': BI, 'BLOCK_K': BK}, num_stages=s, num_warps=w, pre_hook=_ar_cr_fwd_preprocess_pre_hook) \
    for BI in [64, 128]\
    for BK in [32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_I = conf.kwargs["BLOCK_I"]
    BLOCK_K = conf.kwargs["BLOCK_K"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_I * BLOCK_K < 128 * 128
                and conf.num_warps == 8)

@triton.autotune(configs=list(filter(keep, configs)), key=["block_b1", "block_b2", "HEAD_DIM"], cache_results=True)
@triton.jit
def _ar_cr_fwd_preprocess(Z, H, A, F,
              al_ptr, cl_ptr, q_ptr,
              lse_ptr,
              sm_scale_sqrt,
              block_b1: tl.constexpr,
              block_b2: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_K: tl.constexpr,
              BLOCK_I: tl.constexpr,
              ):
    start_i = tl.program_id(0) * BLOCK_I
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hd = off_h * HEAD_DIM
    off_aj = tl.program_id(2)
    off_a = off_aj // block_b2
    off_j = off_aj % block_b2
    off_za = off_z * A + off_a
    off_zha = off_hz * A + off_a

    ZA = Z * A
    HD = H * HEAD_DIM
    khd_stride = block_b1 * HD
    jkhd_stride = block_b2 * khd_stride
    fjkhd_stride = F * jkhd_stride
    desc_al = _maybe_make_tensor_desc(
        al_ptr,
        shape=[ZA, F, block_b2, block_b1, HD],
        strides=[fjkhd_stride, jkhd_stride, khd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_K, HEAD_DIM]
    )
    jhd_stride = block_b2 * HD
    ijhd_stride = jkhd_stride
    desc_q = _maybe_make_tensor_desc(
        q_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, HD, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    ZHA = ZA * H
    jk_stride = block_b2 * block_b1
    fjk_stride = F * jk_stride
    desc_cl = tl.make_block_ptr(
        cl_ptr,
        shape=[ZHA, F, block_b2, block_b1],
        strides=[fjk_stride, jk_stride, block_b1, 1],
        block_shape=[1, 1, 1, BLOCK_K],
        order=[3, 2, 1, 0],
        offsets=[0, 0, 0, 0],
    ) if not isinstance(cl_ptr, tl.tensor_descriptor) else cl_ptr
    ji_stride = jk_stride
    desc_lse = tl.make_block_ptr(
        lse_ptr,
        shape=[ZHA, block_b2, block_b1],
        strides=[ji_stride, block_b1, 1],
        block_shape=[1, 1, BLOCK_I],
        order=[2, 1, 0],
        offsets=[0, 0, 0],
    ) if not isinstance(lse_ptr, tl.tensor_descriptor) else lse_ptr

    dtype = desc_al.dtype
    sm_scale_sqrt = sm_scale_sqrt.to(dtype)

    l_i = tl.zeros([BLOCK_I], dtype=tl.float32)
    m_i = tl.full([BLOCK_I], dtype=tl.float32, value=float("-inf"))

    q_i = desc_q.load([off_za, start_i, off_j, off_h, 0]).reshape(BLOCK_I, HEAD_DIM) * sm_scale_sqrt
    k_range = tl.arange(0, BLOCK_K)

    for f in tl.range(F):
        for k in tl.static_range(0, block_b1, BLOCK_K):
            k_mask = k_range + k < block_b1
            al_k = desc_al.load([off_za, f, off_j, k, off_hd]).reshape(BLOCK_K, HEAD_DIM)
            if isinstance(desc_cl, tl.tensor_descriptor):
                cl_k = desc_cl.load([off_zha, f, off_j, k]).reshape(BLOCK_K) # already in log2 space
            else:
                cl_k = tl.load(desc_cl.advance([off_zha, f, off_j, k]), boundary_check=(3,)).reshape(BLOCK_K)

            bl_ik = tl.dot(q_i, al_k.T) * 1.44269504 # log2(e)
            z_ik = tl.where(k_mask[None, :], bl_ik - cl_k[None, :], float("-inf"))

            m_ik = tl.maximum(m_i, tl.max(z_ik, 1))
            z_ik = z_ik - m_ik[:, None]
            p = tl.math.exp2(z_ik)
            l_ik = tl.sum(p, 1)

            alpha = tl.math.exp2(m_i - m_ik)
            l_i = l_i * alpha + l_ik
            m_i = m_ik

    lse = m_i + tl.log2(l_i)
    if isinstance(desc_lse, tl.tensor_descriptor):
        desc_lse.store([off_zha, off_j, start_i], lse.reshape(1, 1, BLOCK_I))
    else:
        tl.store(desc_lse.advance([off_zha, off_j, start_i]), lse.reshape(1, 1, BLOCK_I), boundary_check=(2,))


def _init_ar_cr_fwd_descs(Z, H, A, F, HEAD_DIM, block_b1, block_b2, aL, q, aR, cL, lse):
    ZA = Z * A
    ZAF = ZA * F
    hd_stride = H * HEAD_DIM
    khd_stride = block_b1 * hd_stride
    jkhd_stride = block_b2 * khd_stride
    jhd_stride = block_b2 * hd_stride
    ijhd_stride = jkhd_stride
    ZHAF = ZAF * H
    jk_stride = block_b2 * block_b1
    ji_stride = jk_stride
    ZHA = ZA * H

    descs = SimpleNamespace()
    if supports_host_descriptor:
        descs.aL = TensorDescriptor(
            aL,
            shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
            strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.q = TensorDescriptor(
            q,
            shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
            strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.aR = TensorDescriptor(
            aR,
            shape=[ZAF, block_b1, block_b2, H, HEAD_DIM],
            strides=[jkhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        # TMA requires 16-byte alignment for leading strides
        descs.cL = TensorDescriptor(
            cL,
            shape=[ZHAF, block_b2, block_b1],
            strides=[jk_stride, block_b1, 1],
            block_shape=[1, 1, 1]
        ) if (block_b1 * cL.element_size()) % 16 == 0 else cL
        descs.lse = TensorDescriptor(
            lse,
            shape=[ZHA, block_b2, block_b1],
            strides=[ji_stride, block_b1, 1],
            block_shape=[1, 1, 1]
        ) if (block_b1 * lse.element_size()) % 16 == 0 else lse
    else:
        descs.aL = aL
        descs.q = q
        descs.aR = aR
        descs.cL = cL
        descs.lse = lse

    return descs

def _ar_cr_fwd_pre_hook(nargs):
    BLOCK_K = nargs["BLOCK_K"]
    BLOCK_I = nargs["BLOCK_I"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["al_ptr"], TensorDescriptor):
        return
    nargs["al_ptr"].block_shape = [1, 1, BLOCK_K, 1, HEAD_DIM]
    nargs["q_ptr"].block_shape = [1, BLOCK_I, 1, 1, HEAD_DIM]
    nargs["ar_ptr"].block_shape = [1, BLOCK_K, 1, 1, HEAD_DIM]
    if isinstance(nargs["cl_ptr"], TensorDescriptor):
        nargs["cl_ptr"].block_shape = [1, 1, BLOCK_K]
    if isinstance(nargs["lse_ptr"], TensorDescriptor):
        nargs["lse_ptr"].block_shape = [1, 1, BLOCK_I]

configs = [
    triton.Config({'BLOCK_I': BI, 'BLOCK_K': BK}, num_stages=s, num_warps=w, pre_hook=_ar_cr_fwd_pre_hook) \
    for BI in [64, 128]\
    for BK in [32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_I = conf.kwargs["BLOCK_I"]
    BLOCK_K = conf.kwargs["BLOCK_K"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_I * BLOCK_K < 128 * 128
                and conf.num_warps == 8)

@triton.autotune(configs=list(filter(keep, configs)), key=["block_b1", "block_b2", "HEAD_DIM"], cache_results=True)
@triton.jit
def _ar_cr_fwd(Z, H, A, F,
              al_ptr, cl_ptr, q_ptr,
              lse_ptr, ar_ptr, cr_ptr,
              sm_scale_sqrt,
              block_b1: tl.constexpr,
              block_b2: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_K: tl.constexpr,
              BLOCK_I: tl.constexpr,
              ):
    start_k = tl.program_id(0) * BLOCK_K
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_afj = tl.program_id(2)
    off_j = off_afj % block_b2
    off_f = off_afj // block_b2 % F
    off_a = off_afj // block_b2 // F
    off_za = off_z * A + off_a
    off_zaf = off_za * F + off_f
    off_zha = off_hz * A + off_a
    off_zhaf = off_zha * F + off_f

    ZA = Z * A
    ZAF = ZA * F
    hd_stride = H * HEAD_DIM
    khd_stride = block_b1 * hd_stride
    jkhd_stride = block_b2 * khd_stride
    desc_al = _maybe_make_tensor_desc(
        al_ptr,
        shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
        strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, 1, BLOCK_K, 1, HEAD_DIM]
    )
    jhd_stride = block_b2 * hd_stride
    ijhd_stride = jkhd_stride
    desc_q = _maybe_make_tensor_desc(
        q_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    desc_ar = _maybe_make_tensor_desc(
        ar_ptr,
        shape=[ZAF, block_b1, block_b2, H, HEAD_DIM],
        strides=[jkhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_K, 1, 1, HEAD_DIM]
    )
    ZHAF = ZAF * H
    jk_stride = block_b2 * block_b1
    desc_cl = tl.make_block_ptr(
        cl_ptr,
        shape=[ZHAF, block_b2, block_b1],
        strides=[jk_stride, block_b1, 1],
        block_shape=[1, 1, BLOCK_K],
        order=[2, 1, 0],
        offsets=[0, 0, 0],
    ) if not isinstance(cl_ptr, tl.tensor_descriptor) else cl_ptr
    cr_ptrs = cr_ptr + (off_zhaf * jk_stride
                         + off_j) + (start_k + tl.arange(0, BLOCK_K)) * block_b2
    ji_stride = jk_stride
    ZHA = ZA * H
    desc_lse = tl.make_block_ptr(
        lse_ptr,
        shape=[ZHA, block_b2, block_b1],
        strides=[ji_stride, block_b1, 1],
        block_shape=[1, 1, BLOCK_I],
        order=[2, 1, 0],
        offsets=[0, 0, 0],
    ) if not isinstance(lse_ptr, tl.tensor_descriptor) else lse_ptr

    dtype = desc_al.dtype

    sm_scale_sqrt = sm_scale_sqrt.to(dtype)
    ar_acc = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
    cr_acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    al_k = desc_al.load([off_zaf, off_j, start_k, off_h, 0]).reshape(BLOCK_K, HEAD_DIM)
    if isinstance(desc_cl, tl.tensor_descriptor):
        cl_k = desc_cl.load([off_zhaf, off_j, start_k]).reshape(BLOCK_K) # already in log2 space
    else:
        cl_k = tl.load(desc_cl.advance([off_zhaf, off_j, start_k]), boundary_check=(2,)).reshape(BLOCK_K)
    i_range = tl.arange(0, BLOCK_I)

    for i in tl.static_range(0, block_b1, BLOCK_I):
        i_mask = i_range + i < block_b1

        q_i = desc_q.load([off_za, i, off_j, off_h, 0]).reshape(BLOCK_I, HEAD_DIM) * sm_scale_sqrt
        q_i = tl.where(i_mask[:, None], q_i, 0.0)
        if isinstance(desc_lse, tl.tensor_descriptor):
            lse_i = desc_lse.load([off_zha, off_j, i]).reshape(BLOCK_I)
        else:
            lse_i = tl.load(desc_lse.advance([off_zha, off_j, i]), boundary_check=(2,)).reshape(BLOCK_I)
        bl_ki = tl.dot(al_k, q_i.T) * 1.44269504 # log2(e)
        z_ki = tl.where(i_mask[None, :], bl_ki - cl_k[:, None] - lse_i[None, :], float("-inf"))

        p = tl.math.exp2(z_ki)
        cr_acc += tl.sum(p, 1)
        p = p.to(dtype)
        ar_acc = tl.dot(p, q_i, ar_acc)

    # cr_acc = tl.clamp(cr_acc, eps, float("inf"))

    desc_ar.store([off_zaf, start_k, off_j, off_h, 0], ar_acc.to(dtype).reshape(1, BLOCK_K, 1, 1, HEAD_DIM))
    tl.store(cr_ptrs, cr_acc, start_k + tl.arange(0, BLOCK_K) < block_b1)


def _init_al_cl_y_fwd_descs(Z, H, A, F, HEAD_DIM, block_b1, block_b2, IS_FIRST_ITER, aR, k, v, aL, y, cR):
    ZAF = Z * A * F
    ZFK = Z * F * block_b1
    if IS_FIRST_ITER:
        # F has stride 0
        ZAFK = Z * A * block_b1
    else:
        ZAFK = ZAF * block_b1
    hd_stride = H * HEAD_DIM
    jhd_stride = block_b2 * hd_stride
    khd_stride = block_b1 * hd_stride
    jkhd_stride = block_b2 * khd_stride
    ZHAF = ZAF * H
    kj_stride = block_b1 * block_b2

    descs = SimpleNamespace()
    if supports_host_descriptor:
        descs.aR = TensorDescriptor(
            aR,
            shape=[ZAFK, block_b2, H, HEAD_DIM],
            strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.k = TensorDescriptor(
            k,
            shape=[ZFK, block_b2, H, HEAD_DIM],
            strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.v = TensorDescriptor(
            v,
            shape=[ZFK, block_b2, H, HEAD_DIM],
            strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1]
        )
        descs.aL = TensorDescriptor(
            aL,
            shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
            strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.y = TensorDescriptor(
            y,
            shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
            strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        # TMA requires 16-byte alignment for leading strides
        descs.cR = TensorDescriptor(
            cR,
            shape=[ZHAF, block_b1, block_b2],
            strides=[kj_stride, block_b2, 1],
            block_shape=[1, 1, 1]
        ) if (block_b2 * cR.element_size()) % 16 == 0 else cR
    else:
        descs.aR = aR
        descs.k = k
        descs.v = v
        descs.aL = aL
        descs.y = y
        descs.cR = cR

    return descs

def _al_cl_y_fwd_pre_hook(nargs):
    BLOCK_J = nargs["BLOCK_J"]
    BLOCK_L = nargs["BLOCK_L"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["ar_ptr"], TensorDescriptor):
        return
    nargs["ar_ptr"].block_shape = [1, BLOCK_J, 1, HEAD_DIM]
    nargs["k_ptr"].block_shape = [1, BLOCK_L, 1, HEAD_DIM]
    nargs["v_ptr"].block_shape = [1, BLOCK_L, 1, HEAD_DIM]
    nargs["al_ptr"].block_shape = [1, BLOCK_J, 1, 1, HEAD_DIM]
    nargs["y_ptr"].block_shape = [1, BLOCK_J, 1, 1, HEAD_DIM]
    if isinstance(nargs["cr_ptr"], TensorDescriptor):
        nargs["cr_ptr"].block_shape = [1, 1, BLOCK_J]

configs = [
    triton.Config({'BLOCK_J': BJ, 'BLOCK_L': BL}, num_stages=s, num_warps=w, pre_hook=_al_cl_y_fwd_pre_hook) \
    for BJ in [16, 32, 64, 128]\
    for BL in [16, 32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_J = conf.kwargs["BLOCK_J"]
    BLOCK_L = conf.kwargs["BLOCK_L"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_J * BLOCK_L < 128 * 128
                and conf.num_warps == 8)

# configs = [triton.Config({'BLOCK_J': 64, 'BLOCK_L': 64}, num_stages=2, num_warps=4, pre_hook=_al_cl_y_fwd_pre_hook)]

@triton.autotune(configs=list(filter(keep, configs)), key=["block_b1", "block_b2", "HEAD_DIM", "IS_FIRST_ITER"], cache_results=True)
@triton.jit
def _al_cl_y_fwd(Z, H, A, F,
                 ar_ptr, cr_ptr,
                 k_ptr, v_ptr,
                 al_ptr, cl_ptr,
                 y_ptr, out_lse_ptr,
                 eps, max_clamp,
                 sm_scale_sqrt,
                 block_b1: tl.constexpr,
                 block_b2: tl.constexpr,
                 HEAD_DIM: tl.constexpr,
                 IS_FIRST_ITER: tl.constexpr,
                 OUTPUT_LSE: tl.constexpr,
                 BLOCK_J: tl.constexpr,
                 BLOCK_L: tl.constexpr,
                 ):
    start_j = tl.program_id(0) * BLOCK_J
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_afk = tl.program_id(2)
    off_k = off_afk % block_b1
    off_f = (off_afk // block_b1) % F
    off_a = off_afk // (block_b1 * F)

    off_zaf = (off_z * A + off_a) * F + off_f
    ZAF = Z * A * F
    off_zfk = (off_z * F + off_f) * block_b1 + off_k
    ZFK = Z * F * block_b1

    if IS_FIRST_ITER:
        # F has stride 0
        off_zafk = (off_z * A + off_a) * block_b1 + off_k
        ZAFK = Z * A * block_b1
    else:
        off_zafk = off_zaf * block_b1 + off_k
        ZAFK = ZAF * block_b1

    hd_stride = H * HEAD_DIM
    jhd_stride = block_b2 * hd_stride

    desc_ar = _maybe_make_tensor_desc(
        ar_ptr,
        shape=[ZAFK, block_b2, H, HEAD_DIM],
        strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_J, 1, HEAD_DIM]
    )
    desc_k = _maybe_make_tensor_desc(
        k_ptr,
        shape=[ZFK, block_b2, H, HEAD_DIM],
        strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_L, 1, HEAD_DIM]
    )
    desc_v = _maybe_make_tensor_desc(
        v_ptr,
        shape=[ZFK, block_b2, H, HEAD_DIM],
        strides=[jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_L, 1, HEAD_DIM]
    )
    khd_stride = block_b1 * hd_stride
    jkhd_stride = block_b2 * khd_stride
    desc_al = _maybe_make_tensor_desc(
        al_ptr,
        shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
        strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_J, 1, 1, HEAD_DIM]
    )
    desc_y = _maybe_make_tensor_desc(
        y_ptr,
        shape=[ZAF, block_b2, block_b1, H, HEAD_DIM],
        strides=[jkhd_stride, khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_J, 1, 1, HEAD_DIM]
    )
    ZHAF = ZAF * H
    off_zhaf = ((off_z * H + off_h) * A + off_a) * F + off_f
    kj_stride = block_b1 * block_b2
    if not IS_FIRST_ITER:
        desc_cr = tl.make_block_ptr(
            cr_ptr,
            shape=[ZHAF, block_b1, block_b2],
            strides=[kj_stride, block_b2, 1],
            block_shape=[1, 1, BLOCK_J],
            order=[2, 1, 0],
            offsets=[0, 0, 0],
        ) if not isinstance(cr_ptr, tl.tensor_descriptor) else cr_ptr
    cl_ptrs = cl_ptr + (off_zhaf * kj_stride
                        + off_k) + (start_j + tl.arange(0, BLOCK_J)) * block_b1
    if OUTPUT_LSE:
        desc_lse = tl.make_block_ptr(
            out_lse_ptr,
            shape=[ZHAF, block_b1, block_b2],
            strides=[kj_stride, block_b2, 1],
            block_shape=[1, 1, BLOCK_J],
            order=[2, 1, 0],
            offsets=[0, 0, 0],
        )

    dtype = desc_ar.dtype

    sm_scale_sqrt = sm_scale_sqrt.to(dtype)
    al_acc = tl.zeros([BLOCK_J, HEAD_DIM], dtype=tl.float32)
    y_acc = tl.zeros([BLOCK_J, HEAD_DIM], dtype=tl.float32)
    cl_acc = tl.zeros([BLOCK_J], dtype=tl.float32)
    l_j = tl.zeros([BLOCK_J], dtype=tl.float32)
    m_j = tl.full([BLOCK_J], dtype=tl.float32, value=float("-inf"))

    ar_j = desc_ar.load([off_zafk, start_j, off_h, 0]).reshape(BLOCK_J, HEAD_DIM)
    if IS_FIRST_ITER:
        cr_j_scale = tl.full([BLOCK_J], dtype=tl.float32, value=1.44269504) # log2(e)
        ar_j = ar_j * sm_scale_sqrt
    else:
        if isinstance(desc_cr, tl.tensor_descriptor):
            cr_j = desc_cr.load([off_zhaf, off_k, start_j]).reshape(BLOCK_J)
        else:
            cr_j = tl.load(desc_cr.advance([off_zhaf, off_k, start_j]), boundary_check=(2,)).reshape(BLOCK_J)
        cr_j_scale = tl.clamp(1.0 / (cr_j + eps), float("-inf"), max_clamp) * 1.44269504 # log2(e)
    l_range = tl.arange(0, BLOCK_L)

    for l in tl.static_range(0, block_b2, BLOCK_L):
        l_mask = l_range + l < block_b2

        k_l = desc_k.load([off_zfk, l, off_h, 0]).reshape(BLOCK_L, HEAD_DIM) * sm_scale_sqrt
        v_l = desc_v.load([off_zfk, l, off_h, 0]).reshape(BLOCK_L, HEAD_DIM)
        k_l = tl.where(l_mask[:, None], k_l, 0.0)
        v_l = tl.where(l_mask[:, None], v_l, 0.0)

        br_jl = tl.dot(ar_j, k_l.T) * cr_j_scale[:, None]
        z_jl = tl.where(l_mask[None, :], br_jl, float("-inf"))

        m_jl = tl.maximum(m_j, tl.max(z_jl, 1))
        z_jl = z_jl - m_jl[:, None]
        p = tl.math.exp2(z_jl)
        l_jl = tl.sum(p, 1)

        alpha = tl.math.exp2(m_j - m_jl)
        al_acc = al_acc * alpha[:, None]
        cl_acc = cl_acc * alpha + tl.sum(br_jl * p, 1)
        y_acc = y_acc * alpha[:, None]

        p = p.to(dtype)
        al_acc = tl.dot(p, k_l, al_acc)
        y_acc = tl.dot(p, v_l, y_acc)

        l_j = l_j * alpha + l_jl
        m_j = m_jl

    al_acc = al_acc / l_j[:, None]
    desc_al.store([off_zaf, start_j, off_k, off_h, 0], al_acc.to(dtype).reshape(1, BLOCK_J, 1, 1, HEAD_DIM))

    y_acc = y_acc / l_j[:, None]
    desc_y.store([off_zaf, start_j, off_k, off_h, 0], y_acc.to(dtype).reshape(1, BLOCK_J, 1, 1, HEAD_DIM))

    lse = m_j + tl.log2(l_j)
    cl_acc = cl_acc / l_j - lse # keep cL in log2 space
    tl.store(cl_ptrs, cl_acc, start_j + tl.arange(0, BLOCK_J) < block_b2)

    if OUTPUT_LSE:
        tl.store(desc_lse.advance([off_zhaf, off_k, start_j]), lse.reshape(1, 1, BLOCK_J), boundary_check=(2,))


def _init_z_fwd_descs(Z, H, A, F, HEAD_DIM, block_b1, block_b2, aL, y, q, z, cL):
    ZA = Z * A
    HD = H * HEAD_DIM
    khd_stride = block_b1 * HD
    jkhd_stride = block_b2 * khd_stride
    fjkhd_stride = F * jkhd_stride
    jhd_stride = block_b2 * HD
    ijhd_stride = jkhd_stride
    ZHA = ZA * H
    jk_stride = block_b2 * block_b1
    fjk_stride = F * jk_stride

    descs = SimpleNamespace()
    if supports_host_descriptor:
        descs.aL = TensorDescriptor(
            aL,
            shape=[ZA, F, block_b2, block_b1, HD],
            strides=[fjkhd_stride, jkhd_stride, khd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.y = TensorDescriptor(
            y,
            shape=[ZA, F, block_b2, block_b1, HD],
            strides=[fjkhd_stride, jkhd_stride, khd_stride, HD, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.q = TensorDescriptor(
            q,
            shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
            strides=[ijhd_stride, jhd_stride, HD, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        descs.z = TensorDescriptor(
            z,
            shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
            strides=[ijhd_stride, jhd_stride, HD, HEAD_DIM, 1],
            block_shape=[1, 1, 1, 1, 1]
        )
        # TMA requires 16-byte alignment for leading strides
        descs.cL = TensorDescriptor(
            cL,
            shape=[ZHA, F, block_b2, block_b1],
            strides=[fjk_stride, jk_stride, block_b1, 1],
            block_shape=[1, 1, 1, 1]
        ) if (block_b1 * cL.element_size()) % 16 == 0 else cL
    else:
        descs.aL = aL
        descs.y = y
        descs.q = q
        descs.z = z
        descs.cL = cL

    return descs

def _z_fwd_pre_hook(nargs):
    BLOCK_K = nargs["BLOCK_K"]
    BLOCK_I = nargs["BLOCK_I"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["al_ptr"], TensorDescriptor):
        return
    nargs["al_ptr"].block_shape = [1, 1, 1, BLOCK_K, HEAD_DIM]
    nargs["y_ptr"].block_shape = [1, 1, 1, BLOCK_K, HEAD_DIM]
    nargs["q_ptr"].block_shape = [1, BLOCK_I, 1, 1, HEAD_DIM]
    nargs["z_ptr"].block_shape = [1, BLOCK_I, 1, 1, HEAD_DIM]
    if isinstance(nargs["cl_ptr"], TensorDescriptor):
        nargs["cl_ptr"].block_shape = [1, 1, 1, BLOCK_K]

configs = [
    triton.Config({'BLOCK_I': BI, 'BLOCK_K': BK}, num_stages=s, num_warps=w, pre_hook=_z_fwd_pre_hook) \
    for BI in [16, 32, 64, 128]\
    for BK in [32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_I = conf.kwargs["BLOCK_I"]
    BLOCK_K = conf.kwargs["BLOCK_K"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_I * BLOCK_K < 128 * 128
                and conf.num_warps == 8)

# configs = [triton.Config({'BLOCK_I': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4, pre_hook=_z_fwd_pre_hook)]
@triton.autotune(configs=list(filter(keep, configs)), key=["block_b1", "block_b2", "HEAD_DIM", "OUTPUT_LSE"], cache_results=True)
@triton.jit
def _z_fwd(Z, H, A, F,
           al_ptr, cl_ptr,
           q_ptr, y_ptr,
           z_ptr, out_lse_ptr,
           sm_scale_sqrt,
           block_b1: tl.constexpr,
           block_b2: tl.constexpr,
           HEAD_DIM: tl.constexpr,
           OUTPUT_LSE: tl.constexpr,
           BLOCK_K: tl.constexpr,
           BLOCK_I: tl.constexpr,
           ):
    start_i = tl.program_id(0) * BLOCK_I
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hd = off_h * HEAD_DIM
    off_aj = tl.program_id(2)
    off_a = off_aj // block_b2
    off_j = off_aj % block_b2
    off_za = off_z * A + off_a
    off_zha = off_hz * A + off_a

    ZA = Z * A
    HD = H * HEAD_DIM
    khd_stride = block_b1 * HD
    jkhd_stride = block_b2 * khd_stride
    fjkhd_stride = F * jkhd_stride
    desc_al = _maybe_make_tensor_desc(
        al_ptr,
        shape=[ZA, F, block_b2, block_b1, HD],
        strides=[fjkhd_stride, jkhd_stride, khd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_K, HEAD_DIM]
    )
    desc_y = _maybe_make_tensor_desc(
        y_ptr,
        shape=[ZA, F, block_b2, block_b1, HD],
        strides=[fjkhd_stride, jkhd_stride, khd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_K, HEAD_DIM]
    )
    jhd_stride = block_b2 * HD
    ijhd_stride = jkhd_stride
    desc_q = _maybe_make_tensor_desc(
        q_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, HD, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    desc_z = _maybe_make_tensor_desc(
        z_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, HD, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    ZHA = ZA * H
    jk_stride = block_b2 * block_b1
    fjk_stride = F * jk_stride
    desc_cl = tl.make_block_ptr(
        cl_ptr,
        shape=[ZHA, F, block_b2, block_b1],
        strides=[fjk_stride, jk_stride, block_b1, 1],
        block_shape=[1, 1, 1, BLOCK_K],
        order=[3, 2, 1, 0],
        offsets=[0, 0, 0, 0],
    ) if not isinstance(cl_ptr, tl.tensor_descriptor) else cl_ptr
    ji_stride = jk_stride
    if OUTPUT_LSE:
        desc_lse = tl.make_block_ptr(
            out_lse_ptr,
            shape=[ZHA, block_b2, block_b1],
            strides=[ji_stride, block_b1, 1],
            block_shape=[1, 1, BLOCK_I],
            order=[2, 1, 0],
            offsets=[0, 0, 0],
        )


    dtype = desc_al.dtype

    z_acc = tl.zeros([BLOCK_I, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_I], dtype=tl.float32)
    m_i = tl.full([BLOCK_I], dtype=tl.float32, value=float("-inf"))

    sm_scale_sqrt = sm_scale_sqrt.to(dtype)
    q_i = desc_q.load([off_za, start_i, off_j, off_h, 0]).reshape(BLOCK_I, HEAD_DIM)
    q_i = q_i * sm_scale_sqrt
    k_range = tl.arange(0, BLOCK_K)

    for f in tl.range(F):
        for k in tl.static_range(0, block_b1, BLOCK_K):
            k_mask = k_range + k < block_b1
            al_k = desc_al.load([off_za, f, off_j, k, off_hd]).reshape(BLOCK_K, HEAD_DIM)
            if isinstance(desc_cl, tl.tensor_descriptor):
                cl_k = desc_cl.load([off_zha, f, off_j, k]).reshape(BLOCK_K) # already in log2 space
            else:
                cl_k = tl.load(desc_cl.advance([off_zha, f, off_j, k]), boundary_check=(2,)).reshape(BLOCK_K)
            y_k = desc_y.load([off_za, f, off_j, k, off_hd]).reshape(BLOCK_K, HEAD_DIM)
            y_k = tl.where(k_mask[:, None], y_k, 0.0)

            bl_ik = tl.dot(q_i, al_k.T) * 1.44269504 # log2(e)
            z_ik = tl.where(k_mask[None, :], bl_ik - cl_k[None, :], float("-inf"))

            m_ik = tl.maximum(m_i, tl.max(z_ik, 1))
            z_ik = z_ik - m_ik[:, None]
            p = tl.math.exp2(z_ik)
            l_ik = tl.sum(p, 1)

            alpha = tl.math.exp2(m_i - m_ik)
            z_acc = z_acc * alpha[:, None]

            p = p.to(dtype)
            z_acc = tl.dot(p, y_k, z_acc)

            l_i = l_i * alpha + l_ik
            m_i = m_ik

    z_acc = z_acc / l_i[:, None]
    desc_z.store([off_za, start_i, off_j, off_h, 0], z_acc.to(dtype).reshape(1, BLOCK_I, 1, 1, HEAD_DIM))

    if OUTPUT_LSE:
        lse = m_i + tl.math.log2(l_i)
        tl.store(desc_lse.advance([off_zha, off_j, start_i]), lse.reshape(1, 1, BLOCK_I), boundary_check=(2,))

configs = [
    triton.Config({'BLOCK_Q': BQ, 'BLOCK_HEAD_DIM': BHD}, num_stages=s, num_warps=w) \
    for BQ in [16, 32, 64, 128]\
    for BHD in [16, 32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

@triton.autotune(configs=configs, key=["HEAD_DIM", "block_b1", "block_b2"], cache_results=True)
@triton.jit
def _al_cl_y_bwd_preprocess(A, F, H,
                            al_ptr, dal_ptr,
                            y_ptr, dy_ptr,
                            d_ptr,
                            block_b1: tl.constexpr,
                            block_b2: tl.constexpr,
                            HEAD_DIM: tl.constexpr,
                            BLOCK_Q: tl.constexpr,
                            BLOCK_HEAD_DIM: tl.constexpr
                            ):
    off_z = tl.program_id(0)
    start_q = tl.program_id(1) * BLOCK_Q
    off_h = tl.program_id(2)

    q_range = tl.arange(0, BLOCK_Q) + start_q
    j_idxs = q_range % block_b2 # make j contiguous for the d store
    k_idxs = (q_range // block_b2) % block_b1
    f_idxs = (q_range // (block_b2 * block_b1)) % F
    a_idxs = q_range // (block_b2 * block_b1 * F)
    q_mask = a_idxs < A

    in_ptr_offset = ((((((off_z * A) + a_idxs) * F + f_idxs) * block_b2 + j_idxs) * block_b1 + k_idxs) * H + off_h) * HEAD_DIM
    al_ptrs = al_ptr + in_ptr_offset
    dal_ptrs = dal_ptr + in_ptr_offset

    d_ptrs = d_ptr + ((((off_z * H + off_h) * A + a_idxs) * F + f_idxs) * block_b1 + k_idxs) * block_b2 + j_idxs

    acc = tl.zeros([BLOCK_Q], dtype=tl.float32)
    tl.static_assert(HEAD_DIM % BLOCK_HEAD_DIM == 0, "HEAD_DIM must be multiple of BLOCK_HEAD_DIM")
    for d in tl.static_range(0, HEAD_DIM, BLOCK_HEAD_DIM):
        d_range = d + tl.arange(0, BLOCK_HEAD_DIM)
        z_i = tl.load(al_ptrs[:, None] + d_range[None, :], mask=q_mask[:, None])
        dz_i = tl.load(dal_ptrs[:, None] + d_range[None, :], mask=q_mask[:, None])
        acc += (z_i * dz_i).sum(1)

    y_ptrs = y_ptr + in_ptr_offset
    dy_ptrs = dy_ptr + in_ptr_offset

    for d in tl.static_range(0, HEAD_DIM, BLOCK_HEAD_DIM):
        d_range = d + tl.arange(0, BLOCK_HEAD_DIM)
        y_i = tl.load(y_ptrs[:, None] + d_range[None, :], mask=q_mask[:, None])
        dy_i = tl.load(dy_ptrs[:, None] + d_range[None, :], mask=q_mask[:, None])
        acc += (y_i * dy_i).sum(1)
    
    tl.store(d_ptrs, acc, mask=q_mask)

configs = [
    triton.Config({'BLOCK_J': BJ, 'BLOCK_L': BL}, num_stages=s, num_warps=w) \
    for BJ in [16, 32]#, 64, 128]\
    for BL in [16, 32]#, 64, 128]\
    for s in [1, 2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_J = conf.kwargs["BLOCK_J"]
    BLOCK_L = conf.kwargs["BLOCK_L"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_J * BLOCK_L < 128 * 128
                and conf.num_warps == 8)

@triton.autotune(configs=list(filter(keep, configs)), key=["block_b1", "block_b2", "HEAD_DIM", "IS_FIRST_ITER"], cache_results=True)
@triton.jit
def _al_cl_y_bwd(Z, H, A, F,
                 ar_ptr, cr_ptr,
                 k_ptr, v_ptr, cl_ptr,
                 dal_ptr, dcl_ptr, dy_ptr,
                 lse_ptr, d_ptr,
                 dar_ptr, dcr_ptr,
                 dk_ptr, dv_ptr,
                 eps, max_clamp,
                 block_b1: tl.constexpr,
                 block_b2: tl.constexpr,
                 HEAD_DIM: tl.constexpr,
                 IS_FIRST_ITER: tl.constexpr,
                 BLOCK_J: tl.constexpr,
                 BLOCK_L: tl.constexpr,
                 ):
    tl.static_assert(not IS_FIRST_ITER, "only supporting num_iters > 1 for now")
    start_l = tl.program_id(0) * BLOCK_L
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_fk = tl.program_id(2)
    off_f = off_fk // block_b1
    off_k = off_fk % block_b1

    HD = H * HEAD_DIM
    FK = F * block_b1
    KHD = block_b1 * HD
    ZH = Z * H

    off_hd = off_h * HEAD_DIM
    off_khd = off_k * HD + off_hd

    jhd_stride = block_b2 * HD
    kjhd_stride = block_b1 * jhd_stride
    fkjhd_stride = F * kjhd_stride
    afkjhd_stride = A * fkjhd_stride

    if IS_FIRST_ITER:
        desc_ar = tl.make_tensor_descriptor(
            ar_ptr,
            shape=[Z, A, block_b1, block_b2, HD],
            strides=[A * kjhd_stride, kjhd_stride, jhd_stride, HD, 1],
            block_shape=[1, 1, 1, BLOCK_J, HEAD_DIM]
        )
    else:
        desc_ar = tl.make_tensor_descriptor(
            ar_ptr,
            shape=[Z, A, FK, block_b2, HD],
            strides=[afkjhd_stride, fkjhd_stride, jhd_stride, HD, 1],
            block_shape=[1, 1, 1, BLOCK_J, HEAD_DIM]
        )
    desc_dar = tl.make_tensor_descriptor(
        dar_ptr,
        shape=[Z, A, FK, block_b2, HD],
        strides=[afkjhd_stride, fkjhd_stride, jhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_J, HEAD_DIM]
    )
    lhd_stride = jhd_stride
    klhd_stride = kjhd_stride
    fklhd_stride = fkjhd_stride
    desc_k = tl.make_tensor_descriptor(
        k_ptr,
        shape=[Z, F, block_b1, block_b2, HD],
        strides=[fklhd_stride, klhd_stride, lhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_L, HEAD_DIM]
    )
    desc_dk = tl.make_tensor_descriptor(
        dk_ptr,
        shape=[Z, F, block_b1, block_b2, HD],
        strides=[fklhd_stride, klhd_stride, lhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_L, HEAD_DIM]
    )
    desc_v = tl.make_tensor_descriptor(
        v_ptr,
        shape=[Z, F, block_b1, block_b2, HD],
        strides=[fklhd_stride, klhd_stride, lhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_L, HEAD_DIM]
    )
    desc_dv = tl.make_tensor_descriptor(
        dv_ptr,
        shape=[Z, F, block_b1, block_b2, HD],
        strides=[fklhd_stride, klhd_stride, lhd_stride, HD, 1],
        block_shape=[1, 1, 1, BLOCK_L, HEAD_DIM]
    )
    desc_dal = tl.make_tensor_descriptor(
        dal_ptr,
        shape=[Z, A, F, block_b2, KHD],
        strides=[afkjhd_stride, fkjhd_stride, kjhd_stride, KHD, 1],
        block_shape=[1, 1, 1, BLOCK_J, HEAD_DIM]
    )
    desc_dy = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[Z, A, F, block_b2, KHD],
        strides=[afkjhd_stride, fkjhd_stride, kjhd_stride, KHD, 1],
        block_shape=[1, 1, 1, BLOCK_J, HEAD_DIM]
    )
    kj_stride = block_b1 * block_b2
    fkj_stride = F * kj_stride
    afkj_stride = A * fkj_stride
    if IS_FIRST_ITER:
        cr_j_scale = tl.full([BLOCK_J], dtype=tl.float32, value=1)
        active_mask = tl.full([BLOCK_J], dtype=tl.int1, value=(1.0 < max_clamp))
    else:
        desc_cr = tl.make_tensor_descriptor(
            cr_ptr,
            shape=[ZH, A, F, block_b1, block_b2],
            strides=[afkj_stride, fkj_stride, kj_stride, block_b2, 1],
            block_shape=[1, 1, 1, 1, BLOCK_J]
        )
    desc_dcr = tl.make_tensor_descriptor(
        dcr_ptr,
        shape=[ZH, A, F, block_b1, block_b2],
        strides=[afkj_stride, fkj_stride, kj_stride, block_b2, 1],
        block_shape=[1, 1, 1, 1, BLOCK_J]
    )
    cl_off = (off_hz * afkj_stride + off_f * kj_stride + off_k) + (tl.arange(0, BLOCK_J) * block_b1)
    cl_ptrs = cl_ptr + cl_off
    dcl_ptrs = dcl_ptr + cl_off
    desc_lse = tl.make_tensor_descriptor(
        lse_ptr,
        shape=[ZH, A, F, block_b1, block_b2],
        strides=[afkj_stride, fkj_stride, kj_stride, block_b2, 1],
        block_shape=[1, 1, 1, 1, BLOCK_J]
    )
    desc_d = tl.make_tensor_descriptor(
        d_ptr,
        shape=[ZH, A, F, block_b1, block_b2],
        strides=[afkj_stride, fkj_stride, kj_stride, block_b2, 1],
        block_shape=[1, 1, 1, 1, BLOCK_J]
    )

    dtype = desc_ar.dtype

    j_range = tl.arange(0, BLOCK_J)
    l_mask = (start_l + tl.arange(0, BLOCK_L)) < block_b2

    dk_acc = tl.zeros([BLOCK_L, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_L, HEAD_DIM], dtype=tl.float32)

    k_l = desc_k.load([off_z, off_f, off_k, start_l, off_hd]).reshape(BLOCK_L, HEAD_DIM)
    k_l = tl.where(l_mask[:, None], k_l, 0.0)
    v_l = desc_v.load([off_z, off_f, off_k, start_l, off_hd]).reshape(BLOCK_L, HEAD_DIM)

    for a in tl.range(0, A):
        for j in tl.static_range(0, block_b2, BLOCK_J):
            j_mask = j_range + j < block_b2

            ar_j = desc_ar.load([off_z, a, off_k if IS_FIRST_ITER else off_fk, j, off_hd]).reshape(BLOCK_J, HEAD_DIM)
            if not IS_FIRST_ITER:
                cr_j = desc_cr.load([off_hz, a, off_f, off_k, j]).reshape(BLOCK_J)
                cr_j_scale = 1.0 / (cr_j + eps)
                active_mask = cr_j_scale < max_clamp
                cr_j_scale = tl.where(active_mask & j_mask, cr_j_scale, max_clamp)
                # cr_j_scale = tl.clamp(1.0 / (cr_j + eps), float("-inf"), max_clamp)
            lse_j = desc_lse.load([off_hz, a, off_f, off_k, j]).reshape(BLOCK_J)

            br_jl = tl.dot(ar_j, k_l.T) * cr_j_scale[:, None]
            s_jl = (br_jl * 1.44269504) - lse_j[:, None] # scale br by log2(e) (lse already scaled)
            p = tl.math.exp2(s_jl)

            dcl_j = tl.load(dcl_ptrs + j * block_b1, mask=j_mask)
            dp = dcl_j[:, None] * (1 + s_jl * 0.693147181)

            dy_j = desc_dy.load([off_z, a, off_f, j, off_khd]).reshape(BLOCK_J, HEAD_DIM)
            dp = tl.dot(dy_j, v_l.T, dp)

            dal_j = desc_dal.load([off_z, a, off_f, j, off_khd]).reshape(BLOCK_J, HEAD_DIM)
            dp = tl.dot(dal_j, k_l.T, dp)

            d_j = desc_d.load([off_hz, a, off_f, off_k, j]).reshape(BLOCK_J)
            cl_j = tl.load(cl_ptrs + j * block_b1, mask=j_mask)
            d_j += dcl_j * (1 + cl_j * 0.693147181)
            ds_jl = p * (dp - d_j[:, None])

            ds_jl = tl.where(l_mask[None, :], ds_jl, 0.0)
            br_jl = tl.where(l_mask[None, :], br_jl, 0.0)
            dcr_j = tl.sum(ds_jl * br_jl, 1) * (-cr_j_scale)
            dcr_j = tl.where(active_mask, dcr_j, 0.0)
            desc_dcr.atomic_add([off_hz, a, off_f, off_k, j], dcr_j.reshape(1, 1, 1, 1, BLOCK_J))

            dbr_jl = ds_jl * cr_j_scale[:, None]
            dbr_jl = dbr_jl.to(dtype)
            dar_j = tl.dot(dbr_jl, k_l)
            ar_j = tl.where(j_mask[:, None], ar_j, 0.0)
            dbr_jl = tl.where(j_mask[:, None], dbr_jl, 0.0)
            dk_acc = tl.dot(dbr_jl.T, ar_j, dk_acc)
            desc_dar.atomic_add([off_z, a, off_fk, j, off_hd], dar_j.reshape(1, 1, 1, BLOCK_J, HEAD_DIM))

            p = p.to(dtype)
            p = tl.where(j_mask[:, None], p, 0.0)
            dal_j = tl.where(j_mask[:, None], dal_j, 0.0)
            dk_acc = tl.dot(p.T, dal_j, dk_acc)
            dy_j = tl.where(j_mask[:, None], dy_j, 0.0)
            dv_acc = tl.dot(p.T, dy_j, dv_acc)

        cl_ptrs += fkj_stride
        dcl_ptrs += fkj_stride

    desc_dk.store([off_z, off_f, off_k, start_l, off_hd], dk_acc.to(dtype).reshape(1, 1, 1, BLOCK_L, HEAD_DIM))
    desc_dv.store([off_z, off_f, off_k, start_l, off_hd], dv_acc.to(dtype).reshape(1, 1, 1, BLOCK_L, HEAD_DIM))


configs = [
    triton.Config({'BLOCK_Q': BQ, 'BLOCK_HEAD_DIM': BHD}, num_stages=s, num_warps=w) \
    for BQ in [16, 32, 64, 128]\
    for BHD in [16, 32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

@triton.autotune(configs=configs, key=["HEAD_DIM", "block_b1", "block_b2"], cache_results=True)
@triton.jit
def _z_bwd_preprocess(A, H,
                      z_ptr, dz_ptr, d_ptr,
                      block_b1: tl.constexpr,
                      block_b2: tl.constexpr,
                      HEAD_DIM: tl.constexpr,
                      BLOCK_Q: tl.constexpr,
                      BLOCK_HEAD_DIM: tl.constexpr
                      ):
    off_z = tl.program_id(0)
    start_q = tl.program_id(1) * BLOCK_Q
    off_h = tl.program_id(2)

    q_range = tl.arange(0, BLOCK_Q) + start_q
    i_idxs = q_range % block_b1 # make i contiguous for the d store
    j_idxs = (q_range // block_b1) % block_b2
    a_idxs = q_range // (block_b2 * block_b1)
    q_mask = a_idxs < A

    z_ptr_offset = (((((off_z * A) + a_idxs) * block_b1 + i_idxs) * block_b2 + j_idxs) * H + off_h) * HEAD_DIM
    z_ptrs = z_ptr + z_ptr_offset
    dz_ptrs = dz_ptr + z_ptr_offset
    d_range = tl.arange(0, BLOCK_HEAD_DIM)

    d_ptrs = d_ptr + (((off_z * H + off_h) * A + a_idxs) * block_b2 + j_idxs) * block_b1 + i_idxs

    acc = tl.zeros([BLOCK_Q], dtype=tl.float32)
    tl.static_assert(HEAD_DIM % BLOCK_HEAD_DIM == 0, "HEAD_DIM must be multiple of BLOCK_HEAD_DIM")
    for _ in tl.static_range(0, HEAD_DIM, BLOCK_HEAD_DIM):
        z_i = tl.load(z_ptrs[:, None] + d_range[None, :], mask=q_mask[:, None])
        dz_i = tl.load(dz_ptrs[:, None] + d_range[None, :], mask=q_mask[:, None])
        acc += (z_i * dz_i).sum(1)
        d_range += BLOCK_HEAD_DIM
    tl.store(d_ptrs, acc, mask=q_mask)


configs = [
    triton.Config({'BLOCK_I': BI, 'BLOCK_K': BK}, num_stages=s, num_warps=w) \
    for BI in [64, 128]\
    for BK in [32, 64, 128]\
    for s in [2, 3, 4] \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_I = conf.kwargs["BLOCK_I"]
    BLOCK_K = conf.kwargs["BLOCK_K"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_I * BLOCK_K < 128 * 128
                and conf.num_warps == 8)

@triton.autotune(configs=list(filter(keep, configs)), key=["block_b1", "block_b2", "HEAD_DIM"], cache_results=True)
@triton.jit
def _z_bwd(Z, H, A, F,
           al_ptr, cl_ptr,
           q_ptr, y_ptr,
           dz_ptr, lse_ptr, d_ptr,
           dal_ptr, dcl_ptr,
           dq_ptr, dy_ptr,
           block_b1: tl.constexpr,
           block_b2: tl.constexpr,
           HEAD_DIM: tl.constexpr,
           BLOCK_K: tl.constexpr,
           BLOCK_I: tl.constexpr,
           ):
    start_k = tl.program_id(0) * BLOCK_K
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_afj = tl.program_id(2)
    off_j = off_afj % block_b2
    off_a = off_afj // (block_b2 * F)
    off_za = off_z * A + off_a

    AFJ = A * F * block_b2
    off_zafj = off_z * AFJ + off_afj
    off_zhaj = (off_hz * A + off_a) * block_b2 + off_j
    off_zhafj = off_hz * AFJ + off_afj

    ZAFJ = Z * AFJ
    hd_stride = H * HEAD_DIM
    khd_stride = block_b1 * hd_stride
    desc_al = tl.make_tensor_descriptor(
        al_ptr,
        shape=[ZAFJ, block_b1, H, HEAD_DIM],
        strides=[khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_dal = tl.make_tensor_descriptor(
        dal_ptr,
        shape=[ZAFJ, block_b1, H, HEAD_DIM],
        strides=[khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_y = tl.make_tensor_descriptor(
        y_ptr,
        shape=[ZAFJ, block_b1, H, HEAD_DIM],
        strides=[khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_K, 1, HEAD_DIM]
    )
    desc_dy = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[ZAFJ, block_b1, H, HEAD_DIM],
        strides=[khd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_K, 1, HEAD_DIM]
    )

    ZA = Z * A
    jhd_stride = block_b2 * hd_stride
    ijhd_stride = block_b1 * jhd_stride
    desc_q = tl.make_tensor_descriptor(
        q_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    desc_dq = tl.make_tensor_descriptor(
        dq_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    desc_dz = tl.make_tensor_descriptor(
        dz_ptr,
        shape=[ZA, block_b1, block_b2, H, HEAD_DIM],
        strides=[ijhd_stride, jhd_stride, hd_stride, HEAD_DIM, 1],
        block_shape=[1, BLOCK_I, 1, 1, HEAD_DIM]
    )
    ZHAJ = ZA * H * block_b2
    desc_lse = tl.make_tensor_descriptor(
        lse_ptr,
        shape=[ZHAJ, block_b1],
        strides=[block_b1, 1],
        block_shape=[1, BLOCK_I]
    )
    desc_d = tl.make_tensor_descriptor(
        d_ptr,
        shape=[ZHAJ, block_b1],
        strides=[block_b1, 1],
        block_shape=[1, BLOCK_I]
    )
    ZHAFJ = ZAFJ * H
    desc_cl = tl.make_tensor_descriptor(
        cl_ptr,
        shape=[ZHAFJ, block_b1],
        strides=[block_b1, 1],
        block_shape=[1, BLOCK_K]
    )
    desc_dcl = tl.make_tensor_descriptor(
        dcl_ptr,
        shape=[ZHAFJ, block_b1],
        strides=[block_b1, 1],
        block_shape=[1, BLOCK_K]
    )

    dtype = desc_al.dtype

    dal_k_acc = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
    dy_k_acc = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
    dcl_k_acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    al_k = desc_al.load([off_zafj, start_k, off_h, 0]).reshape(BLOCK_K, HEAD_DIM)
    y_kt = desc_y.load([off_zafj, start_k, off_h, 0]).reshape(BLOCK_K, HEAD_DIM).T
    cl_k = desc_cl.load([off_zhafj, start_k]).reshape(BLOCK_K) # already in log2 space

    k_mask = (start_k + tl.arange(0, BLOCK_K)) < block_b1
    al_k = tl.where(k_mask[:, None], al_k, 0.0)

    i_range = tl.arange(0, BLOCK_I)

    for i in tl.static_range(0, block_b1, BLOCK_I):
        i_mask = i_range + i < block_b1
        q_i = desc_q.load([off_za, i, off_j, off_h, 0]).reshape(BLOCK_I, HEAD_DIM)
        dz_i = desc_dz.load([off_za, i, off_j, off_h, 0]).reshape(BLOCK_I, HEAD_DIM)
        dz_i = tl.where(i_mask[:, None], dz_i, 0.0)
        lse_i = desc_lse.load([off_zhaj, i]).reshape(BLOCK_I)
        d_i = desc_d.load([off_zhaj, i]).reshape(BLOCK_I)
        d_i = tl.where(i_mask, d_i, 0.0)

        bl_ik = tl.dot(q_i, al_k.T) * 1.44269504 # log2(e)
        s_ik = bl_ik - cl_k[None, :] - lse_i[:, None]
        s_ik = tl.where(i_mask[:, None], s_ik, float("-inf"))
        p = tl.math.exp2(s_ik)

        dp = tl.dot(dz_i, y_kt)
        ds_ik = p * (dp - d_i[:, None])
        dcl_k_acc -= ds_ik.sum(0)

        p = p.to(dtype)
        dy_k_acc = tl.dot(p.T, dz_i, dy_k_acc)

        ds_ik = ds_ik.to(dtype)
        ds_ik = tl.where(k_mask[None, :], ds_ik, 0.0)
        dq_i = tl.dot(ds_ik, al_k)
        desc_dq.atomic_add([off_za, i, off_j, off_h, 0], dq_i.reshape(1, BLOCK_I, 1, 1, HEAD_DIM))

        q_i = tl.where(i_mask[:, None], q_i, 0.0)
        dal_k_acc = tl.dot(ds_ik.T, q_i, dal_k_acc)

    desc_dal.store([off_zafj, start_k, off_h, 0], dal_k_acc.reshape(1, BLOCK_K, 1, HEAD_DIM).to(dtype))
    desc_dy.store([off_zafj, start_k, off_h, 0], dy_k_acc.reshape(1, BLOCK_K, 1, HEAD_DIM).to(dtype))
    desc_dcl.store([off_zhafj, start_k], dcl_k_acc.reshape(1, BLOCK_K))


def forward(q, k, v, sm_scale, block_b1, block_b2, num_iters, eps=1e-6, max_clamp=1e4):
    b, s, h, d = q.shape

    sm_scale_sqrt = sm_scale ** 0.5
    # q = q * sm_scale_sqrt
    # k = k * sm_scale_sqrt

    q = q.view(b, -1, block_b1, block_b2, h, d) # (b, a, i, j, h, d)
    k = k.view(b, -1, block_b1, block_b2, h, d) # (b, f, k, l, h, d)
    v = v.view(b, -1, block_b1, block_b2, h, d) # (b, f, k, l, h, d)
    a, f = q.shape[1], k.shape[1]

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    triton.set_allocator(alloc_fn)

    def _al_cl_grid(META):
        return (triton.cdiv(block_b2, META["BLOCK_J"]), b * h, a * f * block_b1)
    def _ar_cr_preprocess_grid(META):
        return (triton.cdiv(block_b1, META["BLOCK_I"]), b * h, a * block_b2)
    def _ar_cr_grid(META):
        return (triton.cdiv(block_b1, META["BLOCK_K"]), b * h, a * f * block_b2)

    aR = q.unsqueeze(2).expand(-1, -1, f, -1, -1, -1, -1) # (b, a, f, k, j, h, d)
    cR = torch.empty((b, h, a, f, block_b1, block_b2), device=q.device, dtype=torch.float32)  # (b, h, a, f, k, j)
    aL = torch.empty((b, a, f, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
    cL = torch.empty((b, h, a, f, block_b2, block_b1), device=q.device, dtype=torch.float32)  # (b, h, a, f, j, k)
    
    assert num_iters == 1
    # lse = torch.empty((b, h, a, f, block_b2, block_b1), device=q.device, dtype=torch.float32)  # (b, h, a, f, j, i)

    # for t in range(num_iters - 1):
    #     if t == 0:
    #         al_cl_fwd_descs = _init_al_cl_fwd_descs(b, h, a, f, d, block_b1, block_b2, aR, k, aL, cR)
    #         ar_cr_preprocess_fwd_descs = _init_ar_cr_fwd_preprocess_descs(b, h, a, f, d, block_b1, block_b2, aL, q, cL, lse)
    #     if t == 1:
    #         _modify_al_cl_fwd_descs(b, h, a, f, d, block_b1, block_b2, aR, al_cl_fwd_descs)
    #     _al_cl_fwd[_al_cl_grid](
    #         b, h, a, f,
    #         al_cl_fwd_descs.aR, al_cl_fwd_descs.cR, al_cl_fwd_descs.k,
    #         al_cl_fwd_descs.aL, cL,
    #         eps, max_clamp,
    #         sm_scale_sqrt,
    #         block_b1,
    #         block_b2,
    #         d,
    #         IS_FIRST_ITER=(t == 0),
    #     )
    #     _ar_cr_fwd_preprocess[_ar_cr_preprocess_grid](
    #         b, h, a, f,
    #         ar_cr_preprocess_fwd_descs.aL,
    #         ar_cr_preprocess_fwd_descs.cL,
    #         ar_cr_preprocess_fwd_descs.q,
    #         ar_cr_preprocess_fwd_descs.lse,
    #         sm_scale_sqrt,
    #         block_b1,
    #         block_b2,
    #         d,
    #     )
    #     if t == 0:
    #         aR = torch.empty((b, a, f, block_b1, block_b2, h, d), device=q.device, dtype=q.dtype) # (b, a, f, k, j, h, d)
    #         ar_cr_fwd_descs = _init_ar_cr_fwd_descs(b, h, a, f, d, block_b1, block_b2, aL, q, aR, cL, lse)
    #     _ar_cr_fwd[_ar_cr_grid](
    #         b, h, a, f,
    #         ar_cr_fwd_descs.aL, ar_cr_fwd_descs.cL, ar_cr_fwd_descs.q,
    #         ar_cr_fwd_descs.lse, ar_cr_fwd_descs.aR, cR,
    #         sm_scale_sqrt,
    #         block_b1,
    #         block_b2,
    #         d,
    #     )

    al_cl_y_lse = torch.empty((b, a, f, block_b2, block_b1, h, d), device=q.device, dtype=torch.float32) # (b, a, f, j, k, h, d)
    # lse = torch.empty((b, h, a, f, block_b1, block_b2), device=q.device, dtype=torch.float32)  # (b, h, a, f, k, j)

    al_cl_y_fwd_descs = _init_al_cl_y_fwd_descs(b, h, a, f, d, block_b1, block_b2, (num_iters == 1), aR, k, v, aL, y, cR)
    _al_cl_y_fwd[_al_cl_grid](
        b, h, a, f,
        al_cl_y_fwd_descs.aR, al_cl_y_fwd_descs.cR,
        al_cl_y_fwd_descs.k, al_cl_y_fwd_descs.v,
        al_cl_y_fwd_descs.aL, cL,
        al_cl_y_fwd_descs.y, al_cl_y_lse,
        eps, max_clamp,
        sm_scale_sqrt,
        block_b1,
        block_b2,
        d,
        IS_FIRST_ITER=(num_iters == 1),
        OUTPUT_LSE=True,
    )
    z_lse = torch.empty((b, h, a, block_b2, block_b1), device=q.device, dtype=torch.float32) # (b, h, a, j, i)
    z = torch.empty((b, a, block_b1, block_b2, h, d), device=q.device, dtype=q.dtype) # (b, a, i, j, h, d)
    z_fwd_descs = _init_z_fwd_descs(b, h, a, f, d, block_b1, block_b2, aL, y, q, z, cL)
    _z_fwd[_ar_cr_preprocess_grid](
        b, h, a, f,
        z_fwd_descs.aL, z_fwd_descs.cL,
        z_fwd_descs.q, z_fwd_descs.y,
        z_fwd_descs.z, z_lse,
        sm_scale_sqrt,
        block_b1,
        block_b2,
        d,
        OUTPUT_LSE=True,
    )

    return z.view(b, s, h, d)

def backward(q, k, v, aR, cR, aL, cL, y, z, grad_z, z_lse, al_cl_y_lse, block_b1, block_b2, eps=1e-6, max_clamp=1e4, num_iters=1):
    b, s, h, d = q.shape
    q = q.view(b, -1, block_b1, block_b2, h, d) # (b, a, i, j, h, d)
    k = k.view(b, -1, block_b1, block_b2, h, d) # (b, f, k, l, h, d)
    v = v.view(b, -1, block_b1, block_b2, h, d) # (b, f, k, l, h, d)
    a, f = q.shape[1], k.shape[1]

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    triton.set_allocator(alloc_fn)

    z_d = torch.empty_like(z_lse)
    def grid(META):
        return (b, triton.cdiv(a * block_b1 * block_b2, META["BLOCK_Q"]), h)
    _z_bwd_preprocess[grid](
        a, h,
        z, grad_z, z_d,
        block_b1,
        block_b2,
        d,
    )

    grad_al = torch.empty_like(aL)
    grad_cl = torch.empty_like(cL)
    grad_q = torch.zeros_like(q, dtype=torch.float32)
    grad_y = torch.empty_like(y)
    def grid(META):
        return (triton.cdiv(block_b1, META["BLOCK_K"]), b * h, a * f * block_b2)
    _z_bwd[grid](
        b, h, a, f,
        aL, cL,
        q, y,
        grad_z, z_lse, z_d,
        grad_al, grad_cl,
        grad_q, grad_y,
        block_b1,
        block_b2,
        d,
    )

    aL_y_d = torch.empty_like(al_cl_y_lse)
    def grid(META):
        return (b, triton.cdiv(a * f * block_b2 * block_b1, META["BLOCK_Q"]), h)
    _al_cl_y_bwd_preprocess[grid](
        a, f, h,
        aL, grad_al,
        y, grad_y,
        aL_y_d,
        block_b1,
        block_b2,
        d,
    )

    grad_aR = torch.zeros((b, a, f, block_b1, block_b2, h, d), device=q.device, dtype=torch.float32)
    grad_cR = torch.zeros((b, h, a, f, block_b1, block_b2), device=q.device, dtype=torch.float32)
    grad_v = torch.empty_like(v)
    grad_k = torch.empty_like(k)

    def grid(META):
        return (triton.cdiv(block_b2, META["BLOCK_L"]), b * h, f * block_b1)
    _al_cl_y_bwd[grid](
        b, h, a, f,
        aR, cR,
        k, v, cL,
        grad_al, grad_cl, grad_y,
        al_cl_y_lse, aL_y_d,
        grad_aR, grad_cR,
        grad_k, grad_v,
        eps, max_clamp,
        block_b1,
        block_b2,
        d,
        IS_FIRST_ITER=False, # only supports num_iters > 1 for now
    )

    grad_q = (grad_q + grad_aR.sum(2)).to(q.dtype)
    return grad_q.view(b, s, h, d), grad_k.view(b, s, h, d), grad_v.view(b, s, h, d)


class MonarchAttnImplicitFn(torch.autograd.Function):
    def forward(ctx, q, k, v, sm_scale, num_iters, eps, max_clamp, grad_enabled):
        b, a, block_b1, block_b2, h, d = q.shape
        f = k.shape[1]

        sm_scale_sqrt = sm_scale ** 0.5
        # q = q * sm_scale_sqrt
        # k = k * sm_scale_sqrt

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        def _al_cl_grid(META):
            return (triton.cdiv(block_b2, META["BLOCK_J"]), b * h, a * f * block_b1)
        def _ar_cr_preprocess_grid(META):
            return (triton.cdiv(block_b1, META["BLOCK_I"]), b * h, a * block_b2)
        def _ar_cr_grid(META):
            return (triton.cdiv(block_b1, META["BLOCK_K"]), b * h, a * f * block_b2)

        aR = q.unsqueeze(2).expand(-1, -1, f, -1, -1, -1, -1) # (b, a, f, k, j, h, d)
        cR = torch.empty((b, h, a, f, block_b1, block_b2), device=q.device, dtype=torch.float32)  # (b, h, a, f, k, j)
        aL = torch.empty((b, a, f, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
        cL = torch.empty((b, h, a, f, block_b2, block_b1), device=q.device, dtype=torch.float32)  # (b, h, a, f, j, k)
        
        assert num_iters == 1
        lse = torch.empty((b, h, a, f, block_b2, block_b1), device=q.device, dtype=torch.float32)  # (b, h, a, f, j, i)

        for t in range(num_iters - 1):
            if t == 0:
                al_cl_fwd_descs = _init_al_cl_fwd_descs(b, h, a, f, d, block_b1, block_b2, aR, k, aL, cR)
                ar_cr_preprocess_fwd_descs = _init_ar_cr_fwd_preprocess_descs(b, h, a, f, d, block_b1, block_b2, aL, q, cL, lse)
            if t == 1:
                _modify_al_cl_fwd_descs(b, h, a, f, d, block_b1, block_b2, aR, al_cl_fwd_descs)
            _al_cl_fwd[_al_cl_grid](
                b, h, a, f,
                al_cl_fwd_descs.aR, al_cl_fwd_descs.cR, al_cl_fwd_descs.k,
                al_cl_fwd_descs.aL, cL,
                eps, max_clamp,
                sm_scale_sqrt,
                block_b1,
                block_b2,
                d,
                IS_FIRST_ITER=(t == 0),
            )
            _ar_cr_fwd_preprocess[_ar_cr_preprocess_grid](
                b, h, a, f,
                ar_cr_preprocess_fwd_descs.aL,
                ar_cr_preprocess_fwd_descs.cL,
                ar_cr_preprocess_fwd_descs.q,
                ar_cr_preprocess_fwd_descs.lse,
                sm_scale_sqrt,
                block_b1,
                block_b2,
                d,
            )
            if t == 0:
                aR = torch.empty((b, a, f, block_b1, block_b2, h, d), device=q.device, dtype=q.dtype) # (b, a, f, k, j, h, d)
                ar_cr_fwd_descs = _init_ar_cr_fwd_descs(b, h, a, f, d, block_b1, block_b2, aL, q, aR, cL, lse)
            _ar_cr_fwd[_ar_cr_grid](
                b, h, a, f,
                ar_cr_fwd_descs.aL, ar_cr_fwd_descs.cL, ar_cr_fwd_descs.q,
                ar_cr_fwd_descs.lse, ar_cr_fwd_descs.aR, cR,
                sm_scale_sqrt,
                block_b1,
                block_b2,
                d,
            )

        if grad_enabled:
            al_cl_y_lse = torch.empty((b, a, f, block_b2, block_b1, h, d), device=q.device, dtype=torch.float32) # (b, a, f, j, k, h, d)
        else:
            al_cl_y_lse = None
        y = torch.empty((b, a, f, block_b2, block_b1, h, d), device=q.device, dtype=q.dtype) # (b, a, f, j, k, h, d)
        al_cl_y_fwd_descs = _init_al_cl_y_fwd_descs(b, h, a, f, d, block_b1, block_b2, (num_iters == 1), aR, k, v, aL, y, cR)
        _al_cl_y_fwd[_al_cl_grid](
            b, h, a, f,
            al_cl_y_fwd_descs.aR, al_cl_y_fwd_descs.cR,
            al_cl_y_fwd_descs.k, al_cl_y_fwd_descs.v,
            al_cl_y_fwd_descs.aL, cL,
            al_cl_y_fwd_descs.y, al_cl_y_lse,
            eps, max_clamp,
            sm_scale_sqrt,
            block_b1,
            block_b2,
            d,
            IS_FIRST_ITER=(num_iters == 1),
            OUTPUT_LSE=grad_enabled,
        )
        if grad_enabled:
            assert num_iters == 1
            z_lse = torch.empty((b, h, a, block_b2, block_b1), device=q.device, dtype=torch.float32) # (b, h, a, j, i)
        else:
            z_lse = None
        z = torch.empty((b, a, block_b1, block_b2, h, d), device=q.device, dtype=q.dtype) # (b, a, i, j, h, d)
        z_fwd_descs = _init_z_fwd_descs(b, h, a, f, d, block_b1, block_b2, aL, y, q, z, cL)
        _z_fwd[_ar_cr_preprocess_grid](
            b, h, a, f,
            z_fwd_descs.aL, z_fwd_descs.cL,
            z_fwd_descs.q, z_fwd_descs.y,
            z_fwd_descs.z, z_lse,
            sm_scale_sqrt,
            block_b1,
            block_b2,
            d,
            OUTPUT_LSE=grad_enabled,
        )

        if grad_enabled:
            ctx.save_for_backward(q, k, v, aR, cR, aL, cL, y, z, al_cl_y_lse, z_lse)
            ctx.eps = eps
            ctx.max_clamp = max_clamp

        return z
    
    @staticmethod
    def backward(ctx, grad_z):
        q, k, v, aR, cR, aL, cL, y, z, al_cl_y_lse, z_lse = ctx.saved_tensors
        b, a, block_b1, block_b2, h, d = q.shape
        f = k.shape[1]

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        z_d = torch.empty_like(z_lse)
        def grid(META):
            return (b, triton.cdiv(a * block_b1 * block_b2, META["BLOCK_Q"]), h)
        _z_bwd_preprocess[grid](
            a, h,
            z, grad_z, z_d,
            block_b1,
            block_b2,
            d,
        )

        grad_al = torch.empty_like(aL)
        grad_cl = torch.empty_like(cL)
        grad_q = torch.zeros_like(q, dtype=torch.float32)
        grad_y = torch.empty_like(y)
        def grid(META):
            return (triton.cdiv(block_b1, META["BLOCK_K"]), b * h, a * f * block_b2)
        _z_bwd[grid](
            b, h, a, f,
            aL, cL,
            q, y,
            grad_z, z_lse, z_d,
            grad_al, grad_cl,
            grad_q, grad_y,
            block_b1,
            block_b2,
            d,
        )

        aL_y_d = torch.empty_like(al_cl_y_lse)
        def grid(META):
            return (b, triton.cdiv(a * f * block_b2 * block_b1, META["BLOCK_Q"]), h)
        _al_cl_y_bwd_preprocess[grid](
            a, f, h,
            aL, grad_al,
            y, grad_y,
            aL_y_d,
            block_b1,
            block_b2,
            d,
        )

        grad_aR = torch.zeros((b, a, f, block_b1, block_b2, h, d), device=q.device, dtype=torch.float32)
        grad_cR = torch.zeros((b, h, a, f, block_b1, block_b2), device=q.device, dtype=torch.float32)
        grad_v = torch.empty_like(v)
        grad_k = torch.empty_like(k)

        def grid(META):
            return (triton.cdiv(block_b2, META["BLOCK_L"]), b * h, f * block_b1)
        _al_cl_y_bwd[grid](
            b, h, a, f,
            aR, cR,
            k, v, cL,
            grad_al, grad_cl, grad_y,
            al_cl_y_lse, aL_y_d,
            grad_aR, grad_cR,
            grad_k, grad_v,
            ctx.eps, ctx.max_clamp,
            block_b1,
            block_b2,
            d,
            IS_FIRST_ITER=False, # only supports num_iters > 1 for now
        )

        grad_q = (grad_q + grad_aR.sum(2)).to(q.dtype)
        return grad_q, grad_k, grad_v

def monarch_video_attn(q, k, v, f_tied, h_reduce, w_reduce, h, w):
    b, _, nh, d = q.shape
    def rearrange_fn(x):
        x = x.view(b, -1, f_tied, h_reduce, h // h_reduce, w_reduce, w // w_reduce, nh, d)
        return rearrange(x, 'b a f c i e j h d -> b (a c e) (f i) j h d')
    def return_fn(x):
        return rearrange(x, 'b (a c e) (f i) j h d -> b (a f c i e j) h d', c=h_reduce, e=w_reduce, f=f_tied)
    q = rearrange_fn(q)
    k = rearrange_fn(k)
    v = rearrange_fn(v)
    z = MonarchAttnImplicitFn.apply(q, k, v, (d ** -0.5), 1, 1e-6, 1e4, torch.is_grad_enabled())
    z = return_fn(z)
    return z

__all__ = ["monarch_video_attn"]
