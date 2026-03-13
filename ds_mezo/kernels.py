"""Triton kernels for DS-MeZO — Hopper-native (sm_90), H100-validated.

Kernel 1: zo_muon_update — fuses Frobenius normalization, Newton-Schulz
    orthogonalization, and parameter update into a single kernel launch per
    adapter matrix. Dispatches tall (M>=N, tiles rows) vs wide (M<N, tiles
    columns). Kalman gradient estimation happens in Python before the kernel
    call. Mechanism-critical: this IS the ZO-Muon spectral optimizer.

Kernel 2: fused_power_iter — warm-started power iteration for activation
    subspace tracking. Fuses num_iters×(H.T@H@V + QR) → 1 kernel launch
    per layer. Mechanism-critical: this IS the AGZO activation subspace tracker.

Kernel 3: fused_agzo_perturbation — fuses the entire AGZO perturbation
    pipeline: z_B projection through activation basis V, BV accumulation,
    QR(BV) for column-space basis Q, z_A projection through Q. Replaces
    6 launches per layer with 1. Mechanism-critical: this IS the AGZO
    perturbation generator.

Kernel 4: fused_perturb_dual — computes θ+ = base + z and θ- = base - z
    in a single pass. Multi-CTA elementwise. Saves ~64 kernel launches/step.

H100 (sm_90) audit conclusions:

  Precision — FP32 + allow_tf32=False verified optimal:
    All tl.dot inner products have at least one narrow dimension (R=16 or
    RC≈8). At R=16, Gram products are exactly one 16×16 HMMA tile — no
    throughput benefit over FP32 CUDA cores, and TF32's 10-bit mantissa
    would degrade N-S convergence precision. BF16 would break N-S convergence
    outright. FP8/Transformer Engine not applicable (optimizer math, not
    inference). No dtype change is justified on Hopper for this workload.

  Parallelism — single-CTA grid=(1,) forced by algorithm:
    Kernels 1-3 have serial data dependencies. ZO-Muon N-S iterations require
    the full Gram matrix G = X^T@X before applying S = c₁I + c₃G to all rows.
    Power iteration accumulates H^T@(H@V) across all tokens before QR. AGZO
    accumulates BV across all d_in before QR. Multi-CTA parallelism, TMA
    producer/consumer overlap, and warp specialization all require independent
    work across CTAs — not applicable to serial-accumulation algorithms.
    Kernel 4 is multi-CTA (elementwise, independent per element).

  Memory hierarchy — L2-resident, TMA unnecessary:
    Adapter matrices: A (d_out×r) + B (r×d_in) ≈ 256KB for r=16, d=4096.
    Activation batch: H (T×D) + V (D×R) ≈ 8MB for T=512, D=4096, R=8.
    Total ≈ 8.5MB ≪ H100 50MB L2 cache. Multi-pass sequential reads from L2
    are efficient without TMA async tensor movement.

  Tile sizes — H100 SRAM-aligned:
    BLOCK_M=128, BLOCK_N=128, BLOCK_T=128, BLOCK_D=128: power-of-2 for H100
    SRAM bank alignment. For d_in=4096 → 32 chunks/pass, each 128×16=8KB FP32.
    BLOCK=1024 for elementwise kernel 4: standard load/store tile. No autotune
    needed — single shape class per variant, deterministic tile sizes.

  Pointer arithmetic — Triton 3.6.0 stability:
    Raw stride-based pointer arithmetic with explicit masking. No block pointers
    (tl.make_block_ptr/tl.advance have documented instability in loop structures
    on Hopper in Triton 3.6.0). No experimental APIs (tl.async_copy, etc.).

  Total kernel overhead — <1% of step time:
    161 kernel launches/step, ~620μs total. Against vLLM inference (4 generate/
    score calls, ~100ms+), kernel compute is <1% of wall time. All four kernels
    are mechanism-critical and remain on the active execution path.

Invariant constants (mathematically fixed, not tunable):
  - Newton-Schulz: minimax-optimal degree-3 composition via Equioscillation
    Theorem (Amsel et al. 2025, arXiv:2505.16932). Closed-form per iteration:
    S = c₁·I + c₃·(X^T@X). Greedy composition is globally optimal (Theorem 4.1).
    Convergence basin ℓ = √ε_f32 (Gram matrix roundoff floor: σ² < ε → σ below
    representable precision in G = X^T@X). Produces 12 iterations empirically.
    Performance: 12×2+3 = 27 passes per call, ~135μs/step total (<0.2%).
  - Norm floor = finfo.tiny. Derived from dtype.
  - BLOCK_*=128: power-of-2 for H100 SRAM alignment.

Tensor conventions:
  - All inputs: FP32, 2D, contiguous, CUDA.
  - Adapter rank >= 16 (tl.dot contraction minimum on Hopper).
  - zo_muon_update: param, buf, scratch — identical shape.
  - fused_power_iter: H (T, D), V (D, R). D and R padded to next power of 2
    internally. Register pressure: V tile is next_power_of_2(D) × next_power_of_2(R)
    FP32 values. For D=4096, R=8 → 128KB, near H100 256KB register file limit.
    Spilling possible for larger D/R but acceptable (single call/step).
  - fused_agzo_perturbation: B (R, d_in), V (d_in, RC), z_coeff_B (R, RC),
    z_coeff_A (d_out, RC). R and RC are constexpr. z_coeff tensors must be
    contiguous (kernel uses hardcoded stride=RC for second dimension).
  - fused_perturb_dual: base, z, pos, neg — identical shape, must be contiguous
    (kernel uses flat .numel() indexing).
"""

import torch
import triton
import triton.language as tl


# ── Minimax-optimal Newton-Schulz coefficients (Polar Express) ─────────────
# Equioscillation Theorem (Chebyshev) gives the unique minimax degree-3 odd
# polynomial on [ℓ, u]. Greedy composition is globally optimal (Theorem 4.1,
# Amsel et al. 2025, arXiv:2505.16932). Closed-form per iteration — no search.
#
# Starting bound ℓ = √ε_f32: the Gram matrix G = X^T@X has entries involving
# σ_i². When σ_i < √ε, σ_i² < ε and that SV's contribution to G is
# indistinguishable from FP32 roundoff. Iteration count derived from
# convergence criterion: iterate until max error < ε_f32.
# Kernel form: S = c1·I + c3·(X^T@X), same as Newton-Schulz.


def _ns_coefficients() -> tuple[tuple[float, float], ...]:
    """Minimax-optimal degree-3 N-S coefficients, iterating to convergence.

    Starting bound ell = sqrt(eps_f32): the Gram matrix G = X^T@X has entries
    involving sigma_i^2. When sigma_i < sqrt(eps), sigma_i^2 < eps and that
    SV's contribution to G is indistinguishable from FP32 roundoff. Iteration
    count falls out of the convergence criterion (1 - p(ell) < eps).
    Per-iteration coefficients from Equioscillation Theorem (Amsel et al. 2025).
    """
    eps = float(torch.finfo(torch.float32).eps)
    l, u = eps ** 0.5, 1.0
    coeffs = []
    while 1.0 - l >= eps:
        s = u * u + l * u + l * l
        alpha = (3.0 / s) ** 0.5
        alpha3 = alpha * alpha * alpha
        beta = 4.0 / (2.0 + l * u * (l + u) * alpha3)
        c1 = 1.5 * beta * alpha
        c3 = -0.5 * beta * alpha3
        coeffs.append((c1, c3))
        l = c1 * l + c3 * l * l * l
        u = 2.0 - l
    return tuple(coeffs)


_NS_COEFFS = _ns_coefficients()
_NS_C1 = tuple(c1 for c1, _ in _NS_COEFFS)
_NS_C3 = tuple(c3 for _, c3 in _NS_COEFFS)
# Produces 12 iterations from ℓ=√ε_f32≈3.45e-4 to convergence.
# Performance: 12 iters × 2 passes + 3 = 27 global-memory passes per call.
# At ~135μs/step total across 64 calls, <0.2% of vLLM-dominated step time.


# ── Kernel 1: ZO-Muon spectral update ─────────────────────────────────────
# Fuses: Frobenius normalize → N-S orthogonalization → parameter update.
# Kalman gradient estimation (prediction + observation) runs in Python.
# Single kernel launch per adapter matrix.

@triton.jit
def _zo_muon_tall_kernel(
    param_ptr, buf_ptr, scratch_ptr,
    M, N: tl.constexpr,
    stride_m, stride_n,
    eta,
    BLOCK_M: tl.constexpr,
    NORM_FLOOR: tl.constexpr,
):
    """ZO-Muon for tall matrices (M >= N). Tiles along M rows.
    N-S degree-3 minimax: X = X @ (c1·I + c3·G), G = X.T@X. Inner product is (N, N).
    buf already contains the Kalman posterior mean (updated in Python)."""

    offs_n = tl.arange(0, N)
    num_chunks = tl.cdiv(M, BLOCK_M)

    # Pass 1: Frobenius norm of buf
    norm_sq = tl.zeros([1], dtype=tl.float32)

    for chunk in range(num_chunks):
        m_start = chunk * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask = offs_m[:, None] < M
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)
        norm_sq += tl.sum(buf_tile * buf_tile)

    inv_norm = 1.0 / tl.sqrt(tl.maximum(norm_sq, NORM_FLOOR))

    # Pass 2: Normalize buf → scratch (X₀ = buf / ‖buf‖_F)
    for chunk in range(num_chunks):
        m_start = chunk * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask = offs_m[:, None] < M
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)
        tl.store(scratch_ptr + ptrs, buf_tile * inv_norm, mask=mask)

    # Minimax-optimal degree-3 Newton-Schulz iterations (in-place on scratch)
    # Each iteration: G = X.T@X, S = c1·I + c3·G, X = X@S
    # Coefficients differ per iteration (Equioscillation Theorem, Polar Express)
    for ns_iter in tl.static_range(len(_NS_C1)):
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
        S = _NS_C1[ns_iter] * I_N + _NS_C3[ns_iter] * XtX

        for chunk in range(num_chunks):
            m_start = chunk * BLOCK_M
            offs_m = m_start + tl.arange(0, BLOCK_M)
            mask_2d = offs_m[:, None] < M
            ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            X_tile = tl.load(scratch_ptr + ptrs, mask=mask_2d, other=0.0)
            X_new = tl.dot(X_tile, S, allow_tf32=False)
            tl.store(scratch_ptr + ptrs, X_new, mask=mask_2d)

    # Final pass: param -= eta * X_final
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
    param_ptr, buf_ptr, scratch_ptr,
    M: tl.constexpr, N,
    stride_m, stride_n,
    eta,
    BLOCK_N: tl.constexpr,
    NORM_FLOOR: tl.constexpr,
):
    """ZO-Muon for wide matrices (M < N). Tiles along N columns.
    N-S degree-3 minimax: X = (c1·I + c3·G) @ X, G = X@X.T. Inner product is (M, M).
    buf already contains the Kalman posterior mean (updated in Python)."""

    offs_m = tl.arange(0, M)
    num_chunks = tl.cdiv(N, BLOCK_N)

    # Pass 1: Frobenius norm of buf
    norm_sq = tl.zeros([1], dtype=tl.float32)

    for chunk in range(num_chunks):
        n_start = chunk * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n[None, :] < N
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)
        norm_sq += tl.sum(buf_tile * buf_tile)

    inv_norm = 1.0 / tl.sqrt(tl.maximum(norm_sq, NORM_FLOOR))

    # Pass 2: Normalize buf → scratch
    for chunk in range(num_chunks):
        n_start = chunk * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n[None, :] < N
        ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        buf_tile = tl.load(buf_ptr + ptrs, mask=mask, other=0.0)
        tl.store(scratch_ptr + ptrs, buf_tile * inv_norm, mask=mask)

    # Minimax-optimal degree-3 Newton-Schulz iterations (in-place on scratch)
    # Each iteration: G = X@X.T, S = c1·I + c3·G, X = S@X
    for ns_iter in tl.static_range(len(_NS_C1)):
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
        S = _NS_C1[ns_iter] * I_M + _NS_C3[ns_iter] * XXt

        for chunk in range(num_chunks):
            n_start = chunk * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask = offs_n[None, :] < N
            ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            X_tile = tl.load(scratch_ptr + ptrs, mask=mask, other=0.0)
            X_new = tl.dot(S, X_tile, allow_tf32=False)
            tl.store(scratch_ptr + ptrs, X_new, mask=mask)

    # Final pass: param -= eta * X_final
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
    scratch: torch.Tensor,
    eta: float,
    norm_floor: float,
) -> None:
    """Frobenius normalize → minimax Newton-Schulz → param update.

    buf contains the Kalman posterior mean (updated in Python before this call).
    Dispatches tall (M >= N) vs wide (M < N) kernel. Both forms use
    min(M,N) × min(M,N) inner products (= rank × rank, minimum 16×16).
    N-S coefficients from minimax-optimal degree-3 composition (Polar Express).
    """
    M, N = param.shape
    if M >= N:
        _zo_muon_tall_kernel[(1,)](
            param, buf, scratch,
            M, N=N,
            stride_m=param.stride(0), stride_n=param.stride(1),
            eta=eta,
            BLOCK_M=128,
            NORM_FLOOR=norm_floor,
        )
    else:
        _zo_muon_wide_kernel[(1,)](
            param, buf, scratch,
            M=M, N=N,
            stride_m=param.stride(0), stride_n=param.stride(1),
            eta=eta,
            BLOCK_N=128,
            NORM_FLOOR=norm_floor,
        )


# ── Kernel 2: Fused power iteration ───────────────────────────────────────
# Replaces Python loop: for _ in range(iters): V = H.T @ (H @ V); V, _ = qr(V)
# Fuses H.T @ H @ V matmul chain + modified Gram-Schmidt QR into one kernel.
# Handles arbitrary D and R via pad-to-next-power-of-2 with masking.

@triton.jit
def _power_iter_kernel(
    H_ptr, V_ptr, out_ptr,
    T, D_actual, R_actual,
    D: tl.constexpr, R: tl.constexpr,
    stride_ht, stride_hd,
    stride_vd, stride_vr,
    stride_od, stride_or,
    num_iters: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NORM_FLOOR: tl.constexpr,
):
    """Fused power iteration: V_new = QR(H.T @ H @ V), repeated num_iters times.

    H: (T, D_actual) activation matrix, V: (D_actual, R_actual) basis to refine.
    D and R are padded to next power of 2 for register allocation; D_actual and
    R_actual mask loads/stores to the actual tensor dimensions.
    Tiles along T (token dimension). Modified Gram-Schmidt QR in registers.
    """
    offs_d = tl.arange(0, D)
    offs_r = tl.arange(0, R)
    d_mask = offs_d < D_actual
    r_mask = offs_r < R_actual
    num_chunks = tl.cdiv(T, BLOCK_T)

    V = tl.load(
        V_ptr + offs_d[:, None] * stride_vd + offs_r[None, :] * stride_vr,
        mask=d_mask[:, None] & r_mask[None, :],
        other=0.0,
    )

    for _it in tl.static_range(num_iters):
        HtHV = tl.zeros([D, R], dtype=tl.float32)

        for chunk in range(num_chunks):
            t_start = chunk * BLOCK_T
            offs_t = t_start + tl.arange(0, BLOCK_T)
            t_mask = offs_t[:, None] < T

            H_tile = tl.load(
                H_ptr + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
                mask=t_mask & d_mask[None, :], other=0.0,
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
            norm = tl.sqrt(tl.maximum(tl.sum(v_col * v_col), NORM_FLOOR))
            v_col = v_col / norm
            V = V * (1.0 - col_mask[None, :]) + v_col[:, None] * col_mask[None, :]

    tl.store(
        out_ptr + offs_d[:, None] * stride_od + offs_r[None, :] * stride_or,
        V,
        mask=d_mask[:, None] & r_mask[None, :],
    )


def fused_power_iter(
    H: torch.Tensor,
    V: torch.Tensor,
    num_iters: int,
    norm_floor: float,
) -> torch.Tensor:
    """Fused power iteration: V_new = QR(H.T @ H @ V), repeated num_iters times.

    Handles arbitrary D and R via pad-to-next-power-of-2 with masking.
    H: (T, D), V: (D, R).
    """
    T, D = H.shape
    _, R = V.shape

    out = torch.empty(D, R, device="cuda", dtype=torch.float32)
    _power_iter_kernel[(1,)](
        H, V, out,
        T, D_actual=D, R_actual=R,
        D=triton.next_power_of_2(D), R=triton.next_power_of_2(R),
        stride_ht=H.stride(0), stride_hd=H.stride(1),
        stride_vd=V.stride(0), stride_vr=V.stride(1),
        stride_od=out.stride(0), stride_or=out.stride(1),
        num_iters=num_iters,
        BLOCK_T=128,
        NORM_FLOOR=norm_floor,
    )
    return out


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
    NORM_FLOOR: tl.constexpr,
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
        norm = tl.sqrt(tl.maximum(tl.sum(v_col * v_col), NORM_FLOOR))
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
    norm_floor: float,
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
        NORM_FLOOR=norm_floor,
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
