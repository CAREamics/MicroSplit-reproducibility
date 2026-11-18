"""
Lazy-loading multi-scale dataset for MicroSplit training on large datasets.

Replaces the eager-loading LCMultiChDloader from CAREamics with a dataset
that loads individual frames on demand from disk, caches them in an LRU cache,
and computes multi-scale patches on-the-fly. This reduces memory from O(N*H*W*C)
to O(cache_size * H * W * C), making training on 3500+ frame datasets feasible.

The __getitem__ contract is identical to LCMultiChDloader: returns
(input_patch, normalized_target_patch) where input has multi-scale context.
"""

import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import tifffile
from skimage.transform import resize

from careamics.lvae_training.dataset.utils.index_manager import GridIndexManager
from careamics.lvae_training.dataset.types import TilingMode


class LazyLCDataset:
    """Lazy-loading lateral-context multi-channel dataset for MicroSplit.

    Instead of loading all images into memory at initialization, this dataset:
    - Stores file paths and loads individual frames on demand
    - Uses an LRU cache to keep recently accessed frames in memory
    - Computes multi-scale (downsampled) versions on-the-fly per frame
    - Computes normalization statistics via a streaming pass

    Parameters
    ----------
    channel_dirs : dict[str, Path]
        Mapping from channel name to directory path.
        Must include all target channels plus "combined".
    file_list : list[str]
        Sorted filenames shared across all channel directories.
    frame_indices : list[int]
        Indices into file_list for this split (train/val/test).
    image_size : int
        Patch size (height = width). Default 64.
    grid_size : int
        Grid spacing for patch tiling. Controls patches per frame.
        Larger = fewer patches per epoch. Default 32.
    multiscale_lowres_count : int
        Number of resolution scales (including full res). Default 3.
    input_idx : int
        Channel index for the combined input channel.
    target_idx_list : list[int]
        Channel indices for target channels.
    channel_names : list[str]
        Ordered channel names matching the stacking order in the data array.
    enable_random_cropping : bool
        If True, randomly crop patches (training). If False, use grid positions (val/test).
    enable_rotation : bool
        If True, apply random flips and 90-degree rotations.
    max_val : float or list[float] or None
        Upper clip value (one value, or one per channel). If None, the true
        maximum of the training frames is computed in ``compute_mean_std``.
    cache_size : int
        Number of frames to keep in the LRU cache.
    tiling_mode : TilingMode
        How to handle image boundaries for patch extraction.
    """

    def __init__(
        self,
        channel_dirs: dict,
        file_list: List[str],
        frame_indices: List[int],
        image_size: int = 64,
        grid_size: int = 32,
        multiscale_lowres_count: int = 3,
        input_idx: int = 5,
        target_idx_list: Optional[List[int]] = None,
        channel_names: Optional[List[str]] = None,
        enable_random_cropping: bool = True,
        enable_rotation: bool = True,
        max_val: Optional[Union[float, List[float]]] = None,
        cache_size: int = 64,
        tiling_mode: TilingMode = TilingMode.ShiftBoundary,
    ):
        self.channel_dirs = {k: Path(v) for k, v in channel_dirs.items()}
        self.file_list = file_list
        self.frame_indices = list(frame_indices)
        self.image_size = image_size
        self.grid_size = grid_size
        self.multiscale_lowres_count = multiscale_lowres_count
        self.input_idx = input_idx
        self.target_idx_list = target_idx_list if target_idx_list is not None else list(range(input_idx))
        self.channel_names = channel_names if channel_names is not None else list(self.channel_dirs.keys())
        self.enable_random_cropping = enable_random_cropping
        self.enable_rotation = enable_rotation
        self.tiling_mode = tiling_mode
        self.num_channels = len(self.channel_names)

        # Read one image to get spatial dimensions
        sample_path = self.channel_dirs[self.channel_names[0]] / self.file_list[0]
        sample_img = tifffile.imread(str(sample_path))
        self.frame_h, self.frame_w = sample_img.shape[:2]

        # Max val clipping
        self.max_val = max_val

        # Normalization stats (must be set via set_mean_std before training)
        self._mean = None
        self._std = None

        # LRU caches
        self._frame_cache = OrderedDict()
        self._scale_cache = OrderedDict()
        self._cache_size = cache_size
        self._scale_cache_size = cache_size * (multiscale_lowres_count - 1)

        # Rotation augmentation (no external dependency needed)

        # Build grid index manager
        self._build_idx_manager()

    def _build_idx_manager(self):
        """Build the GridIndexManager for patch indexing."""
        num_frames = len(self.frame_indices)
        data_shape = (num_frames, self.frame_h, self.frame_w, self.num_channels)
        patch_shape = (1, self.image_size, self.image_size, self.num_channels)
        grid_shape = (1, self.grid_size, self.grid_size, self.num_channels)
        self.idx_manager = GridIndexManager(data_shape, grid_shape, patch_shape, self.tiling_mode)

    # ------------------------------------------------------------------
    # Streaming statistics computation
    # ------------------------------------------------------------------

    def compute_stats_streaming(self) -> Tuple[np.ndarray, np.ndarray, Union[float, List[float]]]:
        """Compute per-channel mean, std, and max_val by streaming through all frames.

        Uses Welford-like accumulation: tracks running sum and sum-of-squares
        per channel, requiring only O(C) memory regardless of dataset size.

        max_val is the true maximum over all channels and all frames, as in
        CAREamics ``MultiChDloader`` (``_quantile = 1.0``,
        ``_channelwise_quantile = False``). Clipping at it leaves every
        training pixel unchanged.

        Returns
        -------
        mean : ndarray, shape (C,)
        std : ndarray, shape (C,)
        max_val : float
        """
        n_pixels = 0
        running_sum = np.zeros(self.num_channels, dtype=np.float64)
        running_sum_sq = np.zeros(self.num_channels, dtype=np.float64)
        max_val = -np.inf

        for frame_idx in self.frame_indices:
            frame = self._read_frame_raw(frame_idx)  # (H, W, C)
            frame_f = frame.astype(np.float64)

            for ch in range(self.num_channels):
                ch_data = frame_f[..., ch]
                running_sum[ch] += ch_data.sum()
                running_sum_sq[ch] += (ch_data ** 2).sum()

            n_pixels += self.frame_h * self.frame_w
            max_val = max(max_val, float(frame.max()))

        mean = running_sum / n_pixels
        variance = running_sum_sq / n_pixels - mean ** 2
        # Clamp variance to avoid sqrt of tiny negatives from floating point
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)

        return mean.astype(np.float32), std.astype(np.float32), max_val

    def compute_mean_std(self):
        """Compute normalization statistics via streaming.

        Returns the same dict structure as MultiChDloader.compute_mean_std()
        with "input" and "target" keys.
        """
        mean, std, max_val = self.compute_stats_streaming()
        if self.max_val is None:
            self.max_val = max_val

        # Build mean/std dicts matching CAREamics format: (1, C, 1, 1)
        mean_all = mean[None, :, None, None]  # (1, num_channels, 1, 1)
        std_all = std[None, :, None, None]

        mean_dict = {
            "input": mean_all[:, self.input_idx:self.input_idx + 1],
            "target": mean_all[:, self.target_idx_list],
        }
        std_dict = {
            "input": std_all[:, self.input_idx:self.input_idx + 1],
            "target": std_all[:, self.target_idx_list],
        }
        return mean_dict, std_dict

    # ------------------------------------------------------------------
    # Frame loading and caching
    # ------------------------------------------------------------------

    def _read_frame_raw(self, frame_idx: int) -> np.ndarray:
        """Read all channels for one frame from disk.

        Returns
        -------
        ndarray, shape (H, W, C)
        """
        channels = []
        filename = self.file_list[frame_idx]
        for ch_name in self.channel_names:
            path = self.channel_dirs[ch_name] / filename
            img = tifffile.imread(str(path))
            channels.append(img)
        return np.stack(channels, axis=-1)

    def _load_frame(self, frame_idx: int) -> np.ndarray:
        """Load a frame with LRU caching. Returns (H, W, C) float32, clipped."""
        if frame_idx in self._frame_cache:
            self._frame_cache.move_to_end(frame_idx)
            return self._frame_cache[frame_idx]

        frame = self._read_frame_raw(frame_idx).astype(np.float32)

        # Upper clip
        if self.max_val is not None:
            if isinstance(self.max_val, list):
                for ch in range(self.num_channels):
                    frame[..., ch] = np.minimum(frame[..., ch], self.max_val[ch])
            else:
                frame = np.minimum(frame, self.max_val)

        # Evict oldest if cache full
        if len(self._frame_cache) >= self._cache_size:
            self._frame_cache.popitem(last=False)
        self._frame_cache[frame_idx] = frame
        return frame

    def _get_downsampled(self, frame_idx: int, scale_level: int) -> np.ndarray:
        """Get a downsampled version of a frame (cached per frame+scale)."""
        cache_key = (frame_idx, scale_level)
        if cache_key in self._scale_cache:
            self._scale_cache.move_to_end(cache_key)
            return self._scale_cache[cache_key]

        frame = self._load_frame(frame_idx)
        downsampled = frame
        for _ in range(scale_level):
            h, w = downsampled.shape[:2]
            new_shape = (h // 2, w // 2, downsampled.shape[2])
            downsampled = resize(
                downsampled, new_shape,
                preserve_range=True, anti_aliasing=True,
            ).astype(np.float32)

        # Evict oldest if cache full
        if len(self._scale_cache) >= self._scale_cache_size:
            self._scale_cache.popitem(last=False)
        self._scale_cache[cache_key] = downsampled
        return downsampled

    # ------------------------------------------------------------------
    # Patch extraction helpers
    # ------------------------------------------------------------------

    def _extract_patch(self, data_2d: np.ndarray, h_start: int, w_start: int) -> np.ndarray:
        """Extract a patch with reflect-padding if needed at boundaries.

        Parameters
        ----------
        data_2d : ndarray, shape (H, W)
        h_start, w_start : int, top-left corner (can be negative or exceed bounds)

        Returns
        -------
        ndarray, shape (image_size, image_size), float32
        """
        h_end = h_start + self.image_size
        w_end = w_start + self.image_size
        H, W = data_2d.shape

        # Fast path: no boundary issues
        if h_start >= 0 and w_start >= 0 and h_end <= H and w_end <= W:
            return data_2d[h_start:h_end, w_start:w_end].astype(np.float32)

        # Need padding
        valid_h_start = max(0, h_start)
        valid_h_end = min(H, h_end)
        valid_w_start = max(0, w_start)
        valid_w_end = min(W, w_end)

        patch = data_2d[valid_h_start:valid_h_end, valid_w_start:valid_w_end]

        pad_top = valid_h_start - h_start
        pad_bottom = h_end - valid_h_end
        pad_left = valid_w_start - w_start
        pad_right = w_end - valid_w_end

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")

        return patch.astype(np.float32)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def set_mean_std(self, mean_val, std_val):
        self._mean = mean_val
        self._std = std_val

    def get_mean_std(self):
        return self._mean, self._std

    def _normalize_input(self, inp: np.ndarray) -> np.ndarray:
        """Normalize input using input mean/std. inp shape: (scales, H, W)."""
        mean_dict, std_dict = self.get_mean_std()
        mean = float(mean_dict["input"].mean())
        std = float(std_dict["input"].mean())
        return ((inp - mean) / std).astype(np.float32)

    def _normalize_target(self, target: np.ndarray) -> np.ndarray:
        """Normalize target per-channel. target shape: (C, H, W)."""
        mean_dict, std_dict = self.get_mean_std()
        mean = mean_dict["target"].squeeze(0)  # (C, 1, 1)
        std = std_dict["target"].squeeze(0)
        return ((target - mean) / std).astype(np.float32)

    # ------------------------------------------------------------------
    # Rotation augmentation
    # ------------------------------------------------------------------

    def _apply_rotation(self, inp: np.ndarray, target: np.ndarray):
        """Apply random flips and 90-degree rotations to input and target.

        Applies the same transform to all scales/channels consistently
        using numpy operations (no external dependency).

        Parameters
        ----------
        inp : ndarray, shape (scales, H, W)
        target : ndarray, shape (C, H, W)

        Returns
        -------
        rotated_inp, rotated_target
        """
        # Random horizontal flip
        if np.random.random() < 0.5:
            inp = np.ascontiguousarray(inp[:, :, ::-1])
            target = np.ascontiguousarray(target[:, :, ::-1])
        # Random vertical flip
        if np.random.random() < 0.5:
            inp = np.ascontiguousarray(inp[:, ::-1, :])
            target = np.ascontiguousarray(target[:, ::-1, :])
        # Random 90-degree rotation (0, 1, 2, or 3 times)
        k = np.random.randint(4)
        if k > 0:
            inp = np.rot90(inp, k, axes=(1, 2)).copy()
            target = np.rot90(target, k, axes=(1, 2)).copy()
        return inp, target

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.idx_manager.total_grid_count()

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        idx = index if isinstance(index, int) else index[0]

        # 1. Get frame and spatial location from GridIndexManager
        patch_loc = self.idx_manager.get_patch_location_from_dataset_idx(idx)
        frame_local_idx = patch_loc[0]
        frame_idx = self.frame_indices[frame_local_idx]

        # 2. Load frame (from cache or disk)
        frame = self._load_frame(frame_idx)  # (H, W, C) float32, clipped

        # 3. Determine crop location
        if self.enable_random_cropping:
            max_h = self.frame_h - self.image_size
            max_w = self.frame_w - self.image_size
            if max_h > 0:
                h_start = np.random.randint(0, max_h)
                w_start = np.random.randint(0, max_w)
            else:
                h_start, w_start = 0, 0
        else:
            # Deterministic location from grid
            h_start = patch_loc[1]
            w_start = patch_loc[2]

        # 4. Extract highest-res patch for ALL channels
        h_end = h_start + self.image_size
        w_end = w_start + self.image_size
        patch = frame[h_start:h_end, w_start:w_end, :]  # (64, 64, C)

        # 5. Build multi-scale input (combined channel at multiple resolutions)
        input_patches = [patch[:, :, self.input_idx][np.newaxis]]  # [(1, 64, 64)]

        h_center = h_start + self.image_size // 2
        w_center = w_start + self.image_size // 2

        for scale in range(1, self.multiscale_lowres_count):
            scaled_frame = self._get_downsampled(frame_idx, scale)

            # Adjust center coordinates for this scale
            h_center = h_center // 2
            w_center = w_center // 2

            hs = h_center - self.image_size // 2
            ws = w_center - self.image_size // 2

            # Extract patch for combined channel with padding
            scaled_patch = self._extract_patch(
                scaled_frame[:, :, self.input_idx], hs, ws
            )
            input_patches.append(scaled_patch[np.newaxis])  # [(1, 64, 64)]

        inp = np.concatenate(input_patches, axis=0)  # (1+LC, 64, 64)

        # 6. Build target (highest res only, target channels)
        target = np.stack(
            [patch[:, :, ch] for ch in self.target_idx_list], axis=0
        ).astype(np.float32)  # (num_targets, 64, 64)

        # 7. Rotation augmentation
        if self.enable_rotation:
            inp, target = self._apply_rotation(inp, target)

        # 8. Normalize
        inp = self._normalize_input(inp)
        target = self._normalize_target(target)

        return inp, target

    # ------------------------------------------------------------------
    # Accessors for compatibility with training pipeline
    # ------------------------------------------------------------------

    def get_max_val(self):
        return self.max_val

    def get_num_frames(self):
        return len(self.frame_indices)

    def reduce_data(self, t_list):
        """Reduce to a subset of frames (e.g., for test_mode)."""
        self.frame_indices = [self.frame_indices[i] for i in t_list]
        self._build_idx_manager()
        self._frame_cache.clear()
        self._scale_cache.clear()

    def get_grid_size(self):
        return self.grid_size

    def get_idx_manager(self):
        return self.idx_manager

    def per_side_overlap_pixelcount(self):
        return (self.image_size - self.grid_size) // 2

    def set_img_sz(self, image_size, grid_size):
        """Change patch/grid size and rebuild index manager.

        Required by CAREamics prediction pipeline (get_single_file_mmse).
        """
        self.image_size = image_size
        self.grid_size = grid_size
        self._build_idx_manager()

    def get_data_shape(self):
        """Return the logical data shape: (num_frames, H, W, num_channels)."""
        return (len(self.frame_indices), self.frame_h, self.frame_w, self.num_channels)
