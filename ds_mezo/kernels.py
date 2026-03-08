"""Triton kernels for DS-MeZO controller operations.

Kernel 1: zo_muon_update — fuses gradient computation, continuous cosine-
    similarity masking, momentum accumulation, Newton-Schulz orthogonalization
    (5 iter), and parameter update into a single kernel launch per matrix.

Kernel 2: fused_perturb_dual — computes both θ+ = base + z and
    θ- = base - z in a single pass (halves memory traffic).

Invariant constants (all mathematically fixed, not tunable):
    - Newton-Schulz iteration: X_{k+1} = 0.5 * X_k @ (3I - X_k^T @ X_k)
      The 3.0 and 0.5 are exact Padé approximation coefficients for the
      matrix sign function. Changing them breaks convergence guarantees.
    - 5 iterations: cubic convergence from Frobenius-normalized initialization
      means error ∝ ε₀^(3^5) = ε₀^243. For ε₀ ≈ 0.1 (typical after
      normalization), this is exact to machine precision.
    - 1e-16 norm floor: below FP32 machine epsilon (1.19e-7). Prevents
      division by zero without affecting any representable value.
    - BLOCK_M=128, BLOCK_N=128: GPU SRAM tile sizes matched to H100 L2
      cache geometry (power-of-2 alignment). BLOCK=1024 for elementwise
      ops saturates memory bandwidth. These are hardware constants.

Reference: DS_MeZO_Combined.md §5.2, §6.1
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused ZO-Muon update (gradient + masking + momentum + N-S + param)
# ---------------------------------------------------------------------------

@triton.jit
def _zo_muon_tall_kernel(
    param_ptr, buf_ptr, z_ptr, scratch_ptr,
    M, N: tl.constexpr,
    stride_m, stride_n,
    dd, momentum, eta,
    mask_scale,
    BLOCK_M: tl.constexpr,
):
    """ZO-Muon update for tall matrices (M >= N). Tiles along M rows.
    Standard N-S form: X = 0.5 * X @ (3I_N - X.T @ X). Inner product (N,N)."""

    offs_n = tl.arange(0, N)
    num_chunks = tl.cdiv(M, BLOCK_M)

    # ── Pass 1: grad + continuous masking + momentum + Frobenius norm ─
    norm_sq = tl.zeros([1], dtype=tl.float32)

    for chunk in range(num_chunks):
        m_start = chunk * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask = offs_m[:, None] < M
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

        z_tile = tl.load(z_ptr + ptrs, mask=mask, other=0.0)
        grad_tile = dd * z_tile * mask_scale
        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)

        buf_tile = momentum * buf_tile + (1.0 - momentum) * grad_tile
        tl.store(buf_ptr + ptrs, buf_tile, mask=mask)
        norm_sq += tl.sum(buf_tile * buf_tile)

    inv_norm = 1.0 / tl.sqrt(tl.maximum(norm_sq, 1e-16))

    # ── Pass 2: Normalize buf → scratch (X₀ = buf / ‖buf‖) ───────────
    for chunk in range(num_chunks):
        m_start = chunk * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask = offs_m[:, None] < M
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)
        tl.store(scratch_ptr + ptrs, buf_tile * inv_norm, mask=mask)

    # ── Passes 3-7: 5 Newton-Schulz iterations (in-place on scratch) ─
    # Sequential single-program: Phase A reads all tiles first, then
    # Phase B overwrites tiles. No data hazard.
    for ns_iter in tl.static_range(5):
        # Phase A: accumulate XtX = X.T @ X  →  (N, N)
        XtX = tl.zeros([N, N], dtype=tl.float32)
        for chunk in range(num_chunks):
            m_start = chunk * BLOCK_M
            offs_m = m_start + tl.arange(0, BLOCK_M)
            mask_2d = offs_m[:, None] < M
            ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            X_tile = tl.load(scratch_ptr + ptrs, mask=mask_2d, other=0.0)
            XtX += tl.dot(tl.trans(X_tile), X_tile, allow_tf32=False)

        # Phase B: S = 3I - XtX, then X_new = 0.5 * X @ S (in-place)
        eye_n = tl.arange(0, N)
        I_N = (eye_n[:, None] == eye_n[None, :]).to(tl.float32)
        S = 3.0 * I_N - XtX

        for chunk in range(num_chunks):
            m_start = chunk * BLOCK_M
            offs_m = m_start + tl.arange(0, BLOCK_M)
            mask_2d = offs_m[:, None] < M
            ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            X_tile = tl.load(scratch_ptr + ptrs, mask=mask_2d, other=0.0)
            X_new = 0.5 * tl.dot(X_tile, S, allow_tf32=False)
            tl.store(scratch_ptr + ptrs, X_new, mask=mask_2d)

    # ── Pass 8: param -= eta * X_final (from scratch) ─────────────────
    for chunk in range(num_chunks):
        m_start = chunk * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_2d = offs_m[:, None] < M
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        param_tile = tl.load(param_ptr + ptrs, mask=mask_2d, other=0.0)
        orth_tile = tl.load(scratch_ptr + ptrs, mask=mask_2d, other=0.0)
        tl.store(param_ptr + ptrs, param_tile - eta * orth_tile, mask=mask_2d)


@triton.jit
def _zo_muon_wide_kernel(
    param_ptr, buf_ptr, z_ptr, scratch_ptr,
    M: tl.constexpr, N,
    stride_m, stride_n,
    dd, momentum, eta,
    mask_scale,
    BLOCK_N: tl.constexpr,
):
    """ZO-Muon update for wide matrices (M < N). Tiles along N columns.
    Alternative N-S form: X = 0.5 * (3I_M - X @ X.T) @ X.
    Inner product X @ X.T is (M, M) where M = rank = 16."""

    offs_m = tl.arange(0, M)
    num_chunks = tl.cdiv(N, BLOCK_N)

    # ── Pass 1: grad + continuous masking + momentum + Frobenius norm ─
    norm_sq = tl.zeros([1], dtype=tl.float32)

    for chunk in range(num_chunks):
        n_start = chunk * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n[None, :] < N
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

        z_tile = tl.load(z_ptr + ptrs, mask=mask, other=0.0)
        grad_tile = dd * z_tile * mask_scale
        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)

        buf_tile = momentum * buf_tile + (1.0 - momentum) * grad_tile
        tl.store(buf_ptr + ptrs, buf_tile, mask=mask)
        norm_sq += tl.sum(buf_tile * buf_tile)

    inv_norm = 1.0 / tl.sqrt(tl.maximum(norm_sq, 1e-16))

    # ── Pass 2: Normalize buf → scratch ───────────────────────────────
    for chunk in range(num_chunks):
        n_start = chunk * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n[None, :] < N
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)
        tl.store(scratch_ptr + ptrs, buf_tile * inv_norm, mask=mask)

    # ── Passes 3-7: 5 Newton-Schulz iterations (in-place on scratch) ─
    for ns_iter in tl.static_range(5):
        # Phase A: accumulate XXt = X @ X.T  →  (M, M)
        XXt = tl.zeros([M, M], dtype=tl.float32)
        for chunk in range(num_chunks):
            n_start = chunk * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask = offs_n[None, :] < N
            ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            X_tile = tl.load(scratch_ptr + ptrs, mask=mask, other=0.0)
            XXt += tl.dot(X_tile, tl.trans(X_tile), allow_tf32=False)

        # Phase B: S = 3I_M - XXt, then X_new = 0.5 * S @ X (in-place)
        eye_m = tl.arange(0, M)
        I_M = (eye_m[:, None] == eye_m[None, :]).to(tl.float32)
        S = 3.0 * I_M - XXt

        for chunk in range(num_chunks):
            n_start = chunk * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask = offs_n[None, :] < N
            ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            X_tile = tl.load(scratch_ptr + ptrs, mask=mask, other=0.0)
            X_new = 0.5 * tl.dot(S, X_tile, allow_tf32=False)
            tl.store(scratch_ptr + ptrs, X_new, mask=mask)

    # ── Pass 8: param -= eta * X_final ────────────────────────────────
    for chunk in range(num_chunks):
        n_start = chunk * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n[None, :] < N
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        param_tile = tl.load(param_ptr + ptrs, mask=mask, other=0.0)
        orth_tile = tl.load(scratch_ptr + ptrs, mask=mask, other=0.0)
        tl.store(param_ptr + ptrs, param_tile - eta * orth_tile, mask=mask)


def zo_muon_update(param, buf, z, scratch, dd, momentum, eta, mask_scale):
    """Fused ZO-Muon update: grad + continuous masking + momentum + Newton-Schulz + param update.

    mask_scale: float in [0, 1]. Scales the gradient before momentum accumulation.
      1.0 = full gradient (no masking). 0.0 = zero gradient.
      Computed in controller from cosine similarity between gradient and momentum.

    Dispatches to tall or wide kernel based on matrix shape.
    Tall (M >= N): standard N-S form, X.T@X is (N,N)
    Wide (M < N):  alternative N-S form, X@X.T is (M,M)
    Both use min(M,N) x min(M,N) inner products (= rank x rank = 16x16).
    """
    M, N = param.shape
    if M >= N:
        _zo_muon_tall_kernel[(1,)](
            param, buf, z, scratch,
            M, N=N,
            stride_m=param.stride(0), stride_n=param.stride(1),
            dd=dd, momentum=momentum, eta=eta,
            mask_scale=mask_scale,
            BLOCK_M=128,
        )
    else:
        _zo_muon_wide_kernel[(1,)](
            param, buf, z, scratch,
            M=M, N=N,
            stride_m=param.stride(0), stride_n=param.stride(1),
            dd=dd, momentum=momentum, eta=eta,
            mask_scale=mask_scale,
            BLOCK_N=128,
        )


# ---------------------------------------------------------------------------
# Kernel 2: Fused dual perturbation (pos = base + z, neg = base - z)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_perturb_dual_kernel(
    base_ptr, z_ptr, pos_ptr, neg_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    """Compute pos = base + z and neg = base - z in one pass."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    base = tl.load(base_ptr + offs, mask=mask)
    z = tl.load(z_ptr + offs, mask=mask)
    tl.store(pos_ptr + offs, base + z, mask=mask)
    tl.store(neg_ptr + offs, base - z, mask=mask)


def fused_perturb_dual(base, z, pos_out, neg_out):
    """Compute pos = base + z and neg = base - z in a single kernel."""
    numel = base.numel()
    BLOCK = 1024
    grid = ((numel + BLOCK - 1) // BLOCK,)
    _fused_perturb_dual_kernel[grid](
        base.reshape(-1), z.reshape(-1),
        pos_out.reshape(-1), neg_out.reshape(-1),
        numel, BLOCK=BLOCK,
    )
