"""The tile schedule: which expert and which rows each grouped-GEMM output tile covers.

Pure torch, so it runs on the GPU without a host sync. Shapes: E experts, M assignment rows.
"""
import torch

# Tile height along the token/row dimension. The tile schedule depends on it,
# so it is fixed instead of autotuned (BLOCK_N / BLOCK_K / warps are autotuned).
BLOCK_M = 64


def make_tile_schedule(
    offsets: torch.Tensor, M: int, block_m: int = BLOCK_M
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each M-tile, return (expert id, first row, end row) as int32 tensors.

    offsets: (E+1,), expert e owns rows [offsets[e], offsets[e+1]). Tiles never straddle experts,
    so each tile only touches one expert's weight matrix.

    The number of real tiles depends on the routing, which lives on the GPU. Instead
    of syncing to read it, we launch an upper bound of cdiv(M, block_m) + E tiles;
    surplus tiles get start == end and exit immediately inside the kernel.
    """
    E = offsets.numel() - 1
    counts = offsets[1:] - offsets[:-1]
    tiles_per_expert = (counts + block_m - 1) // block_m
    tile_end = torch.cumsum(tiles_per_expert, 0)           # inclusive prefix sum
    max_tiles = (M + block_m - 1) // block_m + E
    t = torch.arange(max_tiles, device=offsets.device, dtype=tile_end.dtype)
    expert = torch.searchsorted(tile_end, t, right=True)   # first expert whose tile range ends after t
    valid = expert < E
    expert = expert.clamp(max=E - 1)
    tile_in_expert = t - (tile_end - tiles_per_expert)[expert]
    start = offsets[expert] + tile_in_expert * block_m
    end = offsets[expert + 1]
    start = torch.where(valid, start, end)                 # invalid tiles -> empty range
    return expert.to(torch.int32), start.to(torch.int32), end.to(torch.int32)
