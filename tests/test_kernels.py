"""Numerical correctness test for Triton kernels vs PyTorch reference."""
import torch
import sys
sys.path.insert(0, "/home/ubuntu/DS-MeZO")
from ds_mezo.kernels import zo_muon_update, fused_perturb_dual


def pytorch_ref_zo_muon(param, buf, z, dd, momentum, eta, apply_mask, is_wide):
    """PyTorch reference matching the Triton kernel logic."""
    grad = dd * z
    if apply_mask:
        grad_sign = torch.sign(grad)
        buf_sign = torch.sign(buf)
        mask = (grad_sign == buf_sign).float()
        grad = grad * mask
    buf_new = momentum * buf + (1 - momentum) * grad
    buf.copy_(buf_new)

    # Newton-Schulz on buf (normalized)
    norm = buf.norm("fro").clamp(min=1e-8)
    X = buf / norm
    if is_wide:
        # Alternative form for wide matrices
        I = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
        for _ in range(5):
            X = 0.5 * (3 * I - X @ X.T) @ X
    else:
        I = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
        for _ in range(5):
            X = 0.5 * X @ (3 * I - X.T @ X)
    param.sub_(eta * X)


def test_case(name, M, N, apply_mask, dd=0.5, momentum=0.9, eta=1e-4):
    print(f"\n--- {name}: ({M}x{N}), mask={apply_mask} ---")
    torch.manual_seed(42)
    device = "cuda"

    param_ref = torch.randn(M, N, device=device, dtype=torch.float32)
    buf_ref = torch.randn(M, N, device=device, dtype=torch.float32)
    z = torch.randn(M, N, device=device, dtype=torch.float32)

    param_tri = param_ref.clone()
    buf_tri = buf_ref.clone()
    scratch = torch.zeros(M, N, device=device, dtype=torch.float32)

    is_wide = M < N
    pytorch_ref_zo_muon(param_ref, buf_ref, z, dd, momentum, eta, apply_mask, is_wide)
    zo_muon_update(param_tri, buf_tri, z, scratch, dd, momentum, eta, apply_mask)

    buf_diff = (buf_ref - buf_tri).abs().max().item()
    param_diff = (param_ref - param_tri).abs().max().item()
    print(f"  buf   max diff: {buf_diff:.2e}")
    print(f"  param max diff: {param_diff:.2e}")

    ok = buf_diff < 1e-4 and param_diff < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_perturb_dual():
    print("\n--- fused_perturb_dual ---")
    device = "cuda"
    base = torch.randn(16, 896, device=device)
    z = torch.randn(16, 896, device=device)
    pos = torch.empty_like(base)
    neg = torch.empty_like(base)
    fused_perturb_dual(base, z, pos, neg)

    pos_diff = (pos - (base + z)).abs().max().item()
    neg_diff = (neg - (base - z)).abs().max().item()
    print(f"  pos max diff: {pos_diff:.2e}")
    print(f"  neg max diff: {neg_diff:.2e}")
    ok = pos_diff < 1e-6 and neg_diff < 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = []
    results.append(test_case("Tall A (q_proj)", 896, 16, apply_mask=False))
    results.append(test_case("Tall A (q_proj) masked", 896, 16, apply_mask=True))
    results.append(test_case("Wide B", 16, 896, apply_mask=False))
    results.append(test_case("Small tall A (v_proj)", 128, 16, apply_mask=False))
    results.append(test_perturb_dual())

    print(f"\n{'='*40}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
