"""Triton kernels for DS-MeZO — Hopper-native (sm_90).

Kernel 1: zo_muon_update — fuses gradient computation, momentum accumulation,
    Newton-Schulz orthogonalization (5 iter), and parameter update into a
    single kernel launch per matrix. Saves ~2500 kernel launches per step
    vs unfused PyTorch. Mechanism-critical: this IS the ZO-Muon optimizer.

Kernel 2: fused_power_iter — warm-started power iteration for activation
    subspace tracking. Fuses 3×(H.T@H@V + QR) = 9 launches → 1 per layer.
    Saves ~1000 kernel launches per step. Avoids ~55MB intermediate allocations.
    Mechanism-critical: this IS the AGZO activation subspace tracker.

Kernel 3: fused_agzo_perturbation — fuses the entire AGZO perturbation
    pipeline for one layer into a single kernel: z_B projection through
    activation basis V, B@V + QR(B@V) for column-space basis Q, z_A
    projection through Q. Replaces 6 kernel launches per layer (2×cuBLAS
    matmul + 2×elementwise scale + cuBLAS matmul + cuSOLVER QR) with 1.
    Eliminates cuSOLVER overhead on (r, r_calib)=(16,8) matrix.

Kernel 4: fused_perturb_dual — computes θ+ = base + z and θ- = base - z
    in a single pass. Halves kernel launch count vs separate torch.add/sub
    (2 launches → 1 per matrix, saves ~112 launches/step across all layers).

H100 architecture notes:
  - Adapter matrices are A:(d_out×r), B:(r×d_in) with r=16 typically.
    N-S inner products are 16×16 — below the 16-wide minimum for TF32 HMMA
    to provide speedup. FP32 CUDA cores are used, which is correct for
    N-S convergence anyway (cubic convergence requires full FP32 precision).
  - Power iteration R=rank/2≈8 — too narrow for Tensor Cores.
  - tl.dot constraint: contraction dimension K >= 16.
  - Adapter rank >= 16 required: N-S inner products are rank×rank, and tl.dot
    requires contraction dimension >= 16 for Hopper mma.sync instructions.
  - Kernels 1-3 tile sequentially (single-program, grid=(1,)). This is forced
    by the algorithms (N-S iterations are serial, power iteration accumulates,
    AGZO QR is data-dependent). Kernel 4 is multi-CTA elementwise.
    TMA and warp specialization require producer/consumer overlap — not applicable.
  - All data fits in H100's 50MB L2 cache (adapters ~256KB, activations ~8MB).
    Multi-pass sequential reads benefit from L2 residency without TMA.
  - No tl.make_block_ptr: raw pointer arithmetic is the most stable pattern.
  - No experimental APIs (tl.async_copy, tl.experimental.*). Only stable
    primitives: tl.load, tl.store, tl.dot, tl.trans, tl.static_range.

Invariant constants (mathematically fixed, not tunable):
  - Newton-Schulz: X_{k+1} = 0.5 * X_k @ (3I - X_k.T @ X_k)
    3.0 and 0.5 are exact Padé coefficients for matrix sign function.
  - 5 iterations: cubic convergence → error ∝ ε₀^243 ≈ 0 for ε₀ < 0.5.
  - 1e-16 norm floor: below FP32 machine epsilon (1.19e-7).
  - BLOCK_M=128, BLOCK_N=128: power-of-2 tile sizes for H100 SRAM alignment.
    BLOCK_T=128 for token-dimension tiling in power iteration.

Tensor conventions:
  - All inputs: FP32, 2D, contiguous, CUDA.
  - Adapter rank >= 16 (tl.dot contraction minimum).
  - zo_muon_update: param, buf, z, scratch — identical shape.
  - fused_power_iter: H (T, D), V (D, R). D and R are constexpr.
  - fused_agzo_perturbation: B (R, d_in), V (d_in, RC), z_coeff_B (R, RC),
    z_coeff_A (d_out, RC). R and RC are constexpr.
  - fused_perturb_dual: base, z, pos, neg — identical shape.

Reference: DS_MeZO_Combined.md §5.2, §6.1
"""

import torch
import triton
import triton.language as tl


# ── Kernel 1: Fused ZO-Muon update ────────────────────────────────────────
# Fuses: SPSA gradient → momentum → Frobenius normalize →
# 5× Newton-Schulz orthogonalization → parameter update.
# Single kernel launch per adapter matrix (vs ~21 PyTorch ops unfused).

@triton.jit
def _zo_muon_tall_kernel(
    param_ptr, buf_ptr, z_ptr, scratch_ptr,
    M, N: tl.constexpr,
    stride_m, stride_n,
    dd, momentum, eta,
    BLOCK_M: tl.constexpr,
):
    """ZO-Muon for tall matrices (M >= N). Tiles along M rows.
    N-S form: X = 0.5 * X @ (3I_N - X.T @ X). Inner product is (N, N)."""

    offs_n = tl.arange(0, N)
    num_chunks = tl.cdiv(M, BLOCK_M)

    # Pass 1: SPSA gradient + momentum + Frobenius norm
    norm_sq = tl.zeros([1], dtype=tl.float32)

    for chunk in range(num_chunks):
        m_start = chunk * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask = offs_m[:, None] < M
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

        z_tile = tl.load(z_ptr + ptrs, mask=mask, other=0.0)
        grad_tile = dd * z_tile
        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)

        buf_tile = momentum * buf_tile + (1.0 - momentum) * grad_tile
        tl.store(buf_ptr + ptrs, buf_tile, mask=mask)
        norm_sq += tl.sum(buf_tile * buf_tile)

    inv_norm = 1.0 / tl.sqrt(tl.maximum(norm_sq, 1e-16))

    # Pass 2: Normalize buf → scratch (X₀ = buf / ‖buf‖_F)
    for chunk in range(num_chunks):
        m_start = chunk * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask = offs_m[:, None] < M
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)
        tl.store(scratch_ptr + ptrs, buf_tile * inv_norm, mask=mask)

    # Passes 3-7: 5 Newton-Schulz iterations (in-place on scratch)
    # Phase A reads all tiles, then Phase B overwrites. No data hazard.
    for ns_iter in tl.static_range(5):
        XtX = tl.zeros([N, N], dtype=tl.float32)
        for chunk in range(num_chunks):
            m_start = chunk * BLOCK_M
            offs_m = m_start + tl.arange(0, BLOCK_M)
            mask_2d = offs_m[:, None] < M
            ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            X_tile = tl.load(scratch_ptr + ptrs, mask=mask_2d, other=0.0)
            XtX += tl.dot(tl.trans(X_tile), X_tile, allow_tf32=False)

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

    # Pass 8: param -= eta * X_final
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
    BLOCK_N: tl.constexpr,
):
    """ZO-Muon for wide matrices (M < N). Tiles along N columns.
    N-S form: X = 0.5 * (3I_M - X @ X.T) @ X. Inner product is (M, M)."""

    offs_m = tl.arange(0, M)
    num_chunks = tl.cdiv(N, BLOCK_N)

    # Pass 1: SPSA gradient + momentum + Frobenius norm
    norm_sq = tl.zeros([1], dtype=tl.float32)

    for chunk in range(num_chunks):
        n_start = chunk * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n[None, :] < N
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

        z_tile = tl.load(z_ptr + ptrs, mask=mask, other=0.0)
        grad_tile = dd * z_tile
        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)

        buf_tile = momentum * buf_tile + (1.0 - momentum) * grad_tile
        tl.store(buf_ptr + ptrs, buf_tile, mask=mask)
        norm_sq += tl.sum(buf_tile * buf_tile)

    inv_norm = 1.0 / tl.sqrt(tl.maximum(norm_sq, 1e-16))

    # Pass 2: Normalize buf → scratch
    for chunk in range(num_chunks):
        n_start = chunk * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n[None, :] < N
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)
        tl.store(scratch_ptr + ptrs, buf_tile * inv_norm, mask=mask)

    # Passes 3-7: 5 Newton-Schulz iterations (in-place on scratch)
    for ns_iter in tl.static_range(5):
        XXt = tl.zeros([M, M], dtype=tl.float32)
        for chunk in range(num_chunks):
            n_start = chunk * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask = offs_n[None, :] < N
            ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            X_tile = tl.load(scratch_ptr + ptrs, mask=mask, other=0.0)
            XXt += tl.dot(X_tile, tl.trans(X_tile), allow_tf32=False)

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

    # Pass 8: param -= eta * X_final
    for chunk in range(num_chunks):
        n_start = chunk * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n[None, :] < N
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        param_tile = tl.load(param_ptr + ptrs, mask=mask, other=0.0)
        orth_tile = tl.load(scratch_ptr + ptrs, mask=mask, other=0.0)
        tl.store(param_ptr + ptrs, param_tile - eta * orth_tile, mask=mask)


def zo_muon_update(
    param: torch.Tensor,
    buf: torch.Tensor,
    z: torch.Tensor,
    scratch: torch.Tensor,
    dd: float,
    momentum: float,
    eta: float,
) -> None:
    """Fused ZO-Muon update: grad → momentum → Newton-Schulz → param update.

    Dispatches tall (M >= N) vs wide (M < N) kernel. Both forms use
    min(M,N) × min(M,N) inner products (= rank × rank, minimum 16×16).
    """
    M, N = param.shape
    if M >= N:
        _zo_muon_tall_kernel[(1,)](
            param, buf, z, scratch,
            M, N=N,
            stride_m=param.stride(0), stride_n=param.stride(1),
            dd=dd, momentum=momentum, eta=eta,
            BLOCK_M=128,
        )
    else:
        _zo_muon_wide_kernel[(1,)](
            param, buf, z, scratch,
            M=M, N=N,
            stride_m=param.stride(0), stride_n=param.stride(1),
            dd=dd, momentum=momentum, eta=eta,
            BLOCK_N=128,
        )


# ── Kernel 2: Fused power iteration ───────────────────────────────────────
# Replaces Python loop: for _ in range(iters): V = H.T @ (H @ V); V, _ = qr(V)
# Fuses H.T @ H @ V matmul chain + modified Gram-Schmidt QR into one kernel.

@triton.jit
def _power_iter_kernel(
    H_ptr, V_ptr, out_ptr,
    T, D: tl.constexpr, R: tl.constexpr,
    stride_ht, stride_hd,
    stride_vd, stride_vr,
    stride_od, stride_or,
    num_iters: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Fused power iteration: V_new = QR(H.T @ H @ V), repeated num_iters times.

    H: (T, D) activation matrix, V: (D, R) basis to refine, out: (D, R) result.
    Tiles along T (token dimension). D and R held in registers.
    Modified Gram-Schmidt QR in registers — exact for R ≤ 64.
    """
    offs_d = tl.arange(0, D)
    offs_r = tl.arange(0, R)
    num_chunks = tl.cdiv(T, BLOCK_T)

    V = tl.load(
        V_ptr + offs_d[:, None] * stride_vd + offs_r[None, :] * stride_vr,
    )

    for _it in tl.static_range(num_iters):
        HtHV = tl.zeros([D, R], dtype=tl.float32)

        for chunk in range(num_chunks):
            t_start = chunk * BLOCK_T
            offs_t = t_start + tl.arange(0, BLOCK_T)
            t_mask = offs_t[:, None] < T

            H_tile = tl.load(
                H_ptr + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
                mask=t_mask, other=0.0,
            )
            HV_tile = tl.dot(H_tile, V, allow_tf32=False)
            HtHV += tl.dot(tl.trans(H_tile), HV_tile, allow_tf32=False)

        # Modified Gram-Schmidt QR on HtHV (D × R) — in registers
        # Mask-based gather/scatter for column access
        V = HtHV
        for col in tl.static_range(R):
            col_mask = (offs_r == col).to(tl.float32)
            v_col = tl.sum(V * col_mask[None, :], axis=1)
            for prev in tl.static_range(col):
                prev_mask = (offs_r == prev).to(tl.float32)
                q_prev = tl.sum(V * prev_mask[None, :], axis=1)
                dot_val = tl.sum(v_col * q_prev)
                v_col = v_col - dot_val * q_prev
            norm = tl.sqrt(tl.maximum(tl.sum(v_col * v_col), 1e-16))
            v_col = v_col / norm
            V = V * (1.0 - col_mask[None, :]) + v_col[:, None] * col_mask[None, :]

    tl.store(
        out_ptr + offs_d[:, None] * stride_od + offs_r[None, :] * stride_or,
        V,
    )


def fused_power_iter(
    H: torch.Tensor,
    V: torch.Tensor,
    num_iters: int = 3,
) -> torch.Tensor:
    """Fused power iteration: V_new = QR(H.T @ H @ V), repeated num_iters times.

    Uses Triton kernel when D and R are powers of 2 and fit in shared memory.
    Falls back to PyTorch matmul + QR otherwise (e.g. D=1536 for Qwen2).
    H: (T, D), V: (D, R).
    """
    T, D = H.shape
    _, R = V.shape

    d_is_po2 = (D & (D - 1)) == 0
    r_is_po2 = (R & (R - 1)) == 0

    if d_is_po2 and r_is_po2:
        out = torch.empty(D, R, device="cuda", dtype=torch.float32)
        _power_iter_kernel[(1,)](
            H, V, out,
            T, D=D, R=R,
            stride_ht=H.stride(0), stride_hd=H.stride(1),
            stride_vd=V.stride(0), stride_vr=V.stride(1),
            stride_od=out.stride(0), stride_or=out.stride(1),
            num_iters=num_iters,
            BLOCK_T=128,
        )
        return out

    # PyTorch fallback for non-power-of-2 dimensions
    for _ in range(num_iters):
        HV = H @ V
        V = H.T @ HV
        V, _ = torch.linalg.qr(V)
    return V


# ── Kernel 3: Fused AGZO perturbation ─────────────────────────────────────
# Fuses: z_B = (z_coeff_B @ V.T) * eps, BV = B @ V, Q = QR(BV),
#        z_A = (z_coeff_A @ Q.T) * eps
# Replaces 6 kernel launches per layer (2× matmul + QR + 2× matmul + 2× scale)
# with 1 kernel. Eliminates cuSOLVER overhead on tiny (R, RC) matrix.

@triton.jit
def _agzo_perturb_kernel(
    B_ptr, V_ptr, z_coeff_B_ptr, z_coeff_A_ptr,
    z_A_ptr, z_B_ptr,
    d_in, d_out,
    R: tl.constexpr, RC: tl.constexpr,
    stride_b_r, stride_b_din,
    stride_v_din, stride_v_rc,
    stride_za_dout, stride_za_r,
    stride_zb_r, stride_zb_din,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Fused AGZO perturbation for one adapter layer.

    Phase 1: Tile over d_in — compute z_B = (z_coeff_B @ V.T) * eps
             AND accumulate BV = B @ V. Shared tiling over d_in.
    Phase 2: Modified Gram-Schmidt QR on BV (R × RC) in registers.
    Phase 3: Tile over d_out — compute z_A = (z_coeff_A @ Q.T) * eps.

    Small-dim contractions (RC=8) use explicit rank-1 accumulation.
    Large-dim contractions (BLOCK_D=128) use tl.dot.
    """
    offs_r = tl.arange(0, R)
    offs_rc = tl.arange(0, RC)

    # Load z_coeff_B: (R, RC) — always fits in registers
    zcb = tl.load(
        z_coeff_B_ptr + offs_r[:, None] * RC + offs_rc[None, :],
    )

    # Phase 1: tile over d_in
    BV = tl.zeros([R, RC], dtype=tl.float32)
    num_din_chunks = tl.cdiv(d_in, BLOCK_D)

    for chunk in range(num_din_chunks):
        d_start = chunk * BLOCK_D
        offs_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < d_in

        # Load V_tile: (BLOCK_D, RC)
        V_tile = tl.load(
            V_ptr + offs_d[:, None] * stride_v_din + offs_rc[None, :] * stride_v_rc,
            mask=d_mask[:, None], other=0.0,
        )

        # z_B_tile = zcb @ V_tile.T = (R, RC) @ (RC, BLOCK_D) = (R, BLOCK_D)
        # Manual rank-1 accumulation over RC (too small for tl.dot)
        # Mask-based column extraction
        z_B_tile = tl.zeros([R, BLOCK_D], dtype=tl.float32)
        for k in tl.static_range(RC):
            k_mask = (offs_rc == k).to(tl.float32)
            zcb_k = tl.sum(zcb * k_mask[None, :], axis=1)        # (R,)
            V_k = tl.sum(V_tile * k_mask[None, :], axis=1)       # (BLOCK_D,)
            z_B_tile += zcb_k[:, None] * V_k[None, :]
        z_B_tile *= eps

        # Store z_B chunk
        tl.store(
            z_B_ptr + offs_r[:, None] * stride_zb_r + offs_d[None, :] * stride_zb_din,
            z_B_tile, mask=d_mask[None, :],
        )

        # Load B_tile: (R, BLOCK_D)
        B_tile = tl.load(
            B_ptr + offs_r[:, None] * stride_b_r + offs_d[None, :] * stride_b_din,
            mask=d_mask[None, :], other=0.0,
        )

        # BV += B_tile @ V_tile = (R, BLOCK_D) @ (BLOCK_D, RC) = (R, RC)
        BV += tl.dot(B_tile, V_tile, allow_tf32=False)

    # Phase 2: Modified Gram-Schmidt QR on BV (R, RC) — in registers
    # Uses mask-based gather/scatter (Triton 3.6.0 disallows Q[:, col] indexing)
    Q = BV
    for col in tl.static_range(RC):
        col_mask = (offs_rc == col).to(tl.float32)
        v_col = tl.sum(Q * col_mask[None, :], axis=1)
        for prev in tl.static_range(col):
            prev_mask = (offs_rc == prev).to(tl.float32)
            q_prev = tl.sum(Q * prev_mask[None, :], axis=1)
            dot_val = tl.sum(v_col * q_prev)
            v_col = v_col - dot_val * q_prev
        norm = tl.sqrt(tl.maximum(tl.sum(v_col * v_col), 1e-16))
        v_col = v_col / norm
        Q = Q * (1.0 - col_mask[None, :]) + v_col[:, None] * col_mask[None, :]

    # Phase 3: tile over d_out
    num_dout_chunks = tl.cdiv(d_out, BLOCK_D)

    for chunk in range(num_dout_chunks):
        do_start = chunk * BLOCK_D
        offs_do = do_start + tl.arange(0, BLOCK_D)
        do_mask = offs_do < d_out

        # Load z_coeff_A_tile: (BLOCK_D, RC)
        zca_tile = tl.load(
            z_coeff_A_ptr + offs_do[:, None] * RC + offs_rc[None, :],
            mask=do_mask[:, None], other=0.0,
        )

        # z_A_tile = zca_tile @ Q.T = (BLOCK_D, RC) @ (RC, R) = (BLOCK_D, R)
        # Manual rank-1 accumulation over RC
        # Mask-based column extraction
        z_A_tile = tl.zeros([BLOCK_D, R], dtype=tl.float32)
        for k in tl.static_range(RC):
            k_mask = (offs_rc == k).to(tl.float32)
            zca_k = tl.sum(zca_tile * k_mask[None, :], axis=1)   # (BLOCK_D,)
            Q_k = tl.sum(Q * k_mask[None, :], axis=1)            # (R,)
            z_A_tile += zca_k[:, None] * Q_k[None, :]
        z_A_tile *= eps

        # Store z_A chunk
        tl.store(
            z_A_ptr + offs_do[:, None] * stride_za_dout + offs_r[None, :] * stride_za_r,
            z_A_tile, mask=do_mask[:, None],
        )


def fused_agzo_perturbation(
    B: torch.Tensor,
    V: torch.Tensor,
    z_coeff_B: torch.Tensor,
    z_coeff_A: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused AGZO perturbation: compute z_A and z_B in a single kernel.

    Replaces per-layer: z_B = (z_coeff_B @ V.T) * eps, BV = B @ V,
    Q = QR(BV), z_A = (z_coeff_A @ Q.T) * eps. Fuses 6 launches into 1.
    B: (R, d_in), V: (d_in, RC), z_coeff_B: (R, RC), z_coeff_A: (d_out, RC).
    Returns (z_A, z_B) where z_A is (d_out, R) and z_B is (R, d_in).

    Note: z_coeff_B and z_coeff_A must be contiguous — the kernel indexes
    them with hardcoded stride=(cols, 1) matching the constexpr RC dimension.
    """
    R, d_in = B.shape
    _, RC = V.shape
    d_out, _ = z_coeff_A.shape

    z_A = torch.empty(d_out, R, device="cuda", dtype=torch.float32)
    z_B = torch.empty(R, d_in, device="cuda", dtype=torch.float32)

    _agzo_perturb_kernel[(1,)](
        B, V, z_coeff_B, z_coeff_A,
        z_A, z_B,
        d_in, d_out,
        R=R, RC=RC,
        stride_b_r=B.stride(0), stride_b_din=B.stride(1),
        stride_v_din=V.stride(0), stride_v_rc=V.stride(1),
        stride_za_dout=z_A.stride(0), stride_za_r=z_A.stride(1),
        stride_zb_r=z_B.stride(0), stride_zb_din=z_B.stride(1),
        eps=eps,
        BLOCK_D=128,
    )
    return z_A, z_B


# ── Kernel 4: Fused dual perturbation ─────────────────────────────────────
# Computes θ+ = base + z and θ- = base - z in a single pass.
# Halves kernel launches: 2 ops → 1 per matrix, ~112 launches/step saved.

@triton.jit
def _perturb_dual_kernel(
    base_ptr, z_ptr, pos_ptr, neg_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """Elementwise: pos = base + z, neg = base - z."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = tl.load(base_ptr + offs, mask=mask)
    z = tl.load(z_ptr + offs, mask=mask)
    tl.store(pos_ptr + offs, base + z, mask=mask)
    tl.store(neg_ptr + offs, base - z, mask=mask)


def fused_perturb_dual(
    base: torch.Tensor,
    z: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> None:
    """Fused dual perturbation: pos = base + z, neg = base - z.

    Writes into pre-allocated pos and neg buffers.
    Inputs must not alias outputs (no in-place: base != pos, z != neg, etc.).
    """
    N = base.numel()
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _perturb_dual_kernel[grid](
        base, z, pos, neg,
        N, BLOCK=BLOCK,
    )
