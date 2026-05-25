import torch
# import fastgraphcompute.extensions
import os
import os.path as osp
from .bin_by_coordinates import bin_by_coordinates
from .index_replacer import index_replacer
from typing import Optional, Tuple

# load the custom extension library
torch.ops.load_library(osp.join(osp.dirname(
    osp.realpath(__file__)), 'binned_knn_ops.so'))


def _compute_pca_projection(coords: torch.Tensor, k: int, subsample: int) -> torch.Tensor:
    """Compute a (d, k) PCA projection matrix from a subsample of coords.

    Eigenvector signs are canonicalised (largest-magnitude entry positive)
    so that the projection is deterministic across runs.
    """
    n = coords.shape[0]
    d = coords.shape[1]
    if subsample > 0 and n > subsample:
        idx = torch.randperm(n, device=coords.device)[:subsample]
        sample = coords.index_select(0, idx)
    else:
        sample = coords
    sample = sample.to(dtype=torch.float32)
    # Centre the sample
    sample = sample - sample.mean(dim=0, keepdim=True)
    # torch.pca_lowrank: returns U, S, V with V of shape (d, q)
    q = min(k + 2, d)
    _, _, V = torch.pca_lowrank(sample, q=q, center=False)
    Vk = V[:, :k].contiguous()
    # Sign canonicalisation: flip so the largest-magnitude entry of each column is positive
    abs_Vk = Vk.abs()
    max_idx = abs_Vk.argmax(dim=0)
    signs = torch.sign(Vk[max_idx, torch.arange(k, device=Vk.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    Vk = Vk * signs.unsqueeze(0)
    return Vk


@torch.jit.script
def binned_select_knn(K: int,
                      coords: torch.Tensor,
                      row_splits: torch.Tensor,
                      direction: Optional[torch.Tensor] = None,
                      n_bins: Optional[torch.Tensor] = None,
                      max_bin_dims: int = 3,
                      torch_compatible_indices: bool = False,
                      bin_coords: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform K-Nearest Neighbors selection using binning with C++ autograd support.

    Args:
        K (int): Number of nearest neighbors.
        coords (torch.Tensor): Input coordinates for points.
        row_splits (torch.Tensor): Row splits following ragged tensor convention.
        direction (torch.Tensor, optional): Direction constraint for neighbors. 0: can only be neighbour, 1: can only have neighbour, 2: neither
        n_bins (torch.Tensor, optional): Number of bins per dimension.
        max_bin_dims (int, optional): Maximum number of bin dimensions.
        torch_compatible_indices (bool, optional): Compatibility flag for PyTorch behavior.
        bin_coords (torch.Tensor, optional): Pre-computed coordinates in the binning space
            (e.g. a PCA projection of ``coords``). If provided, must have shape
            ``[coords.shape[0], k]`` with ``k in [1, 5]``; binning happens in this space
            while distance computation still uses the full ``coords``. The exactness
            guarantee is preserved as long as the projection from ``coords`` to
            ``bin_coords`` is contractive (true for any linear projection by a matrix
            with orthonormal columns).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Indices and distances of the nearest neighbors.
    """
    # Validate input coordinates
    if coords.shape[1] == 0:
        raise ValueError(
            "Input coordinates must have at least one dimension. Got 0 dimensions.")
    if max_bin_dims == 0:
        raise ValueError("max_bin_dims must be greater than 0. Got 0.")

    # Type checking for JIT compatibility
    if not isinstance(K, int):
        K = int(K)
    if not isinstance(max_bin_dims, int):
        max_bin_dims = int(max_bin_dims)

    # Automatically adjust max_bin_dims based on coordinate dimensions
    # FGC supports max_bin_dims of 2, 3, 4, or 5 only
    # Limit to min of coordinate dimensions and 5
    coord_dims = coords.shape[1]
    max_bin_dims = min(max_bin_dims, coord_dims, 5)
    max_bin_dims = max(max_bin_dims, 2)  # Ensure at least 2

    # Ensure row_splits is a tensor
    if not isinstance(row_splits, torch.Tensor):
        row_splits = torch.tensor(
            row_splits, dtype=torch.int64, device=coords.device)

    # Ensure coordinates are float32 for CUDA kernel compatibility
    if coords.dtype != torch.float32:
        coords = coords.to(dtype=torch.float32)

    # Convert n_bins to tensor if it's an integer
    if n_bins is not None and not isinstance(n_bins, torch.Tensor):
        n_bins = torch.tensor(n_bins, dtype=torch.int64, device=coords.device)

    # Autograd preserves input tensors across the backward pass automatically;
    # only contiguity is required for the CUDA kernel.
    coords = coords.contiguous()

    row_splits = row_splits.contiguous()

    if direction is not None:
        direction = direction.contiguous()

    if n_bins is not None:
        n_bins = n_bins.contiguous()

    row_splits = row_splits.to(dtype=torch.int64, copy=False)

    if n_bins is not None and isinstance(n_bins, torch.Tensor):
        n_bins = n_bins.to(dtype=torch.int64, copy=False)

    if bin_coords is not None:
        if bin_coords.dtype != torch.float32:
            bin_coords = bin_coords.to(dtype=torch.float32)
        bin_coords = bin_coords.contiguous()

    # Use the C++ autograd kernel
    # Note: use_int32_indices defaults to False here for ABI compatibility;
    # the public binned_select_knn / binned_select_knn_pca wrappers expose
    # the kwarg in commit (d).
    idx, dist = torch.ops.fastgraphcompute_custom_ops.binned_select_knn_autograd(
        coords, row_splits, K, direction, n_bins, max_bin_dims, torch_compatible_indices, bin_coords, False)

    return idx, dist


def binned_select_knn_pca(K: int,
                          coords: torch.Tensor,
                          row_splits: torch.Tensor,
                          direction: Optional[torch.Tensor] = None,
                          n_bins: Optional[torch.Tensor] = None,
                          max_bin_dims: int = 3,
                          torch_compatible_indices: bool = False,
                          pca_subsample: int = 50000) -> Tuple[torch.Tensor, torch.Tensor]:
    """PCA-projected variant of :func:`binned_select_knn`.

    Computes the top-``max_bin_dims`` principal components of ``coords`` (on a
    random subsample for speed) and uses the projection as the binning space.
    Distance computation still uses the full ``coords``, so the result is exact
    in the same sense as the axis-aligned default: partial PCA is a contractive
    linear projection, so the termination test (w*s)^2 > r_K^2 in the projected
    space remains a valid lower bound in the full space.

    For ``coords.shape[1] <= max_bin_dims`` this falls back to the standard
    ``binned_select_knn`` (no projection needed).
    """
    coord_dims = coords.shape[1]
    k = min(max_bin_dims, coord_dims, 5)
    k = max(k, 2)
    if coord_dims <= k or os.environ.get('FGC_DISABLE_PCA') == '1':
        return binned_select_knn(K, coords, row_splits, direction, n_bins,
                                 max_bin_dims, torch_compatible_indices, None)
    Vk = _compute_pca_projection(coords.detach(), k, pca_subsample)
    bin_coords = coords.to(dtype=torch.float32).matmul(Vk)
    return binned_select_knn(K, coords, row_splits, direction, n_bins,
                             max_bin_dims, torch_compatible_indices, bin_coords)
