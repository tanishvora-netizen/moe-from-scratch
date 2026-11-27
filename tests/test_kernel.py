"""The grouped-GEMM kernel against a plain PyTorch reference, forward and both gradients.
Shapes: T tokens, E experts, M rows of the output, K the inner dimension, N the output width.
"""
import pytest
import torch

triton = pytest.importorskip("triton")
from moe.kernels.grouped_gemm import grouped_gemm  # noqa: E402
from moe.kernels.tile_schedule import make_tile_schedule  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
TOL = dict(atol=3e-2, rtol=3e-2) if DTYPE == torch.bfloat16 else dict(atol=1e-4, rtol=1e-4)


def reference(a, b, offsets, row_idx):
    """grouped_gemm computed one expert at a time in fp32."""
    a_rows = a[row_idx] if row_idx is not None else a
    o = offsets.tolist()
    return torch.cat([a_rows[o[e]:o[e + 1]].float() @ b[e].float() for e in range(b.shape[0])])


def make_case(counts, T, K, N, gather, seed=0):
    """Random grouped_gemm inputs where expert e owns counts[e] rows. gather=True adds a row_idx into T rows."""
    g = torch.Generator().manual_seed(seed)
    E, M = len(counts), sum(counts)
    offsets = torch.nn.functional.pad(torch.tensor(counts).cumsum(0), (1, 0)).to(DEVICE)
    a = torch.randn(T if gather else M, K, generator=g).to(DEVICE, DTYPE)
    b = (torch.randn(E, K, N, generator=g) * 0.1).to(DEVICE, DTYPE)
    row_idx = torch.randint(0, T, (M,), generator=g).to(DEVICE) if gather else None
    return a, b, offsets, row_idx


CASES = [
    # counts per expert, T, K, N
    pytest.param([5, 0, 70, 1, 33], 40, 32, 48, id="ragged-counts"),
    pytest.param([64, 64, 64, 64], 100, 64, 64, id="full-tiles"),
    pytest.param([0, 0, 130, 0], 50, 48, 80, id="empty-experts"),
    pytest.param([17] * 8, 60, 96, 40, id="equal-partial-tiles"),
]


@pytest.mark.parametrize("gather", [True, False])
@pytest.mark.parametrize("counts,T,K,N", CASES)
def test_forward(counts, T, K, N, gather):
    a, b, offsets, row_idx = make_case(counts, T, K, N, gather)
    out = grouped_gemm(a, b, offsets, row_idx)
    torch.testing.assert_close(out.float(), reference(a, b, offsets, row_idx), **TOL)


@pytest.mark.parametrize("gather", [True, False])
@pytest.mark.parametrize("counts,T,K,N", CASES)
def test_backward(counts, T, K, N, gather):
    a, b, offsets, row_idx = make_case(counts, T, K, N, gather, seed=1)
    a1, b1 = a.clone().requires_grad_(), b.clone().requires_grad_()
    a2, b2 = a.clone().float().requires_grad_(), b.clone().float().requires_grad_()
    # Rounded to DTYPE because .float()'s backward hands grouped_gemm a DTYPE gradient. With an unrounded
    # grad_out, the reference sees a different dC, and that rounding error grows with the rows per expert.
    grad_out = torch.randn(sum(counts), N, device=DEVICE).to(DTYPE).float()

    (grouped_gemm(a1, b1, offsets, row_idx).float() * grad_out).sum().backward()
    (reference(a2, b2, offsets, row_idx) * grad_out).sum().backward()
    torch.testing.assert_close(a1.grad.float(), a2.grad, **TOL)
    torch.testing.assert_close(b1.grad.float(), b2.grad, **TOL)


def test_tile_schedule_covers_every_row_once():
    counts = [5, 0, 70, 1, 33, 0, 128]
    offsets = torch.nn.functional.pad(torch.tensor(counts).cumsum(0), (1, 0))
    M, BM = sum(counts), 32
    expert, start, end = make_tile_schedule(offsets, M, block_m=BM)
    covered = torch.zeros(M, dtype=torch.long)
    for e, s, t in zip(expert.tolist(), start.tolist(), end.tolist(), strict=True):
        if s >= t:
            continue
        rows = torch.arange(s, min(s + BM, t))
        assert offsets[e] <= rows.min() and rows.max() < offsets[e + 1], "tile crosses an expert boundary"
        covered[rows] += 1
    assert torch.all(covered == 1)
