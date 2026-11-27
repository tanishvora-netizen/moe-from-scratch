"""The tile schedule, which gives each grouped-GEMM output tile one expert and a row range.
It is plain torch, so it runs on the GPU without a host sync. Shapes: E experts, M assignment rows.
"""
import torch

# Rows per forward-kernel tile. The schedule depends on it, so it stays fixed while BLOCK_N and BLOCK_K are autotuned.
BLOCK_M = 64


def make_tile_schedule(
    offsets: torch.Tensor, M: int, block_m: int = BLOCK_M
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (expert, start, end), int32 tensors with one entry per tile of block_m rows.

    offsets is (E+1,), and expert e owns rows [offsets[e], offsets[e+1]). Tiles never cross an expert boundary,
    so each tile reads one expert's weights. The real tile count depends on routing that lives on the GPU.
    To avoid a host sync, the schedule has cdiv(M, block_m) + E entries, an upper bound. Surplus tiles have
    start == end, and the kernel returns on them at once.
    """
    E = offsets.numel() - 1
    counts = offsets[1:] - offsets[:-1]
    tiles_per_expert = (counts + block_m - 1) // block_m
    tile_end = torch.cumsum(tiles_per_expert, 0)           # tile_end[e] is one past expert e's last tile
    max_tiles = (M + block_m - 1) // block_m + E
    t = torch.arange(max_tiles, device=offsets.device, dtype=tile_end.dtype)
    expert = torch.searchsorted(tile_end, t, right=True)   # first expert whose tile range ends after t
    valid = expert < E                                     # surplus tiles map to expert E
    expert = expert.clamp(max=E - 1)
    tile_in_expert = t - (tile_end - tiles_per_expert)[expert]
    start = offsets[expert] + tile_in_expert * block_m
    end = offsets[expert + 1]
    start = torch.where(valid, start, end)                 # a surplus tile gets the empty range [end, end)
    return expert.to(torch.int32), start.to(torch.int32), end.to(torch.int32)
