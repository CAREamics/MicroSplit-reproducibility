"""
MicroSplit plate-level prediction utilities for Cell Painting data.

Provides tiling, stitching, inference and saving functions that are shared
across all Cell Painting experiments.  The only experiment-specific input is
the ``channel_names`` list (and the model/stats objects, which are loaded by
:mod:`model_io`).
"""

import contextlib
import gc
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import tifffile
from skimage.transform import resize

from .image_io import read_channel_paths
from .manifest import (
    PATH_PREFIX,
    append_metadata_row,
    completed_fovs,
    row_channel_paths,
)
from .metrics import compute_metrics


DEFAULT_CHANNEL_NAMES = ["DNA", "RNA", "ER", "AGP", "Mito"]


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def generate_tile_positions(
    H: int,
    W: int,
    image_size: int,
    grid_size: int,
) -> Tuple[List[int], List[int]]:
    """Generate tile start positions using ShiftBoundary logic.

    Parameters
    ----------
    H, W : int
        Full image dimensions.
    image_size : int
        Patch height/width.
    grid_size : int
        Stride between patches.

    Returns
    -------
    h_positions, w_positions : list of int
        Start row/column indices.
    """
    def _positions(total: int, img_sz: int, grd_sz: int) -> List[int]:
        pos = []
        p = 0
        while p + img_sz <= total:
            pos.append(p)
            p += grd_sz
        # Ensure the last tile reaches the end
        if not pos or pos[-1] + img_sz < total:
            pos.append(max(0, total - img_sz))
        return pos

    return _positions(H, image_size, grid_size), _positions(W, image_size, grid_size)


def build_multiscale_frames(
    combined: np.ndarray,
    multiscale_lowres_count: int,
) -> List[np.ndarray]:
    """Build progressively downsampled versions of the combined image.

    Parameters
    ----------
    combined : ndarray (H, W)
        Full-resolution combined image.
    multiscale_lowres_count : int
        Total number of scale levels (1 = no downsampling).

    Returns
    -------
    list of ndarray
        ``[full_res, half_res, quarter_res, ...]``, length
        ``multiscale_lowres_count``.
    """
    frames = [combined.astype(np.float32)]
    current = combined.astype(np.float32)
    for _ in range(1, multiscale_lowres_count):
        h, w = current.shape
        current = resize(
            current, (h // 2, w // 2),
            preserve_range=True, anti_aliasing=True,
        ).astype(np.float32)
        frames.append(current)
    return frames


def extract_patch_padded(
    data_2d: np.ndarray,
    h_start: int,
    w_start: int,
    size: int,
) -> np.ndarray:
    """Extract a square patch with reflect-padding for out-of-bounds regions."""
    H, W = data_2d.shape
    h_end, w_end = h_start + size, w_start + size

    if h_start >= 0 and w_start >= 0 and h_end <= H and w_end <= W:
        return data_2d[h_start:h_end, w_start:w_end].astype(np.float32)

    vh_s, vh_e = max(0, h_start), min(H, h_end)
    vw_s, vw_e = max(0, w_start), min(W, w_end)
    patch = data_2d[vh_s:vh_e, vw_s:vw_e]

    pad_t = vh_s - h_start
    pad_b = h_end - vh_e
    pad_l = vw_s - w_start
    pad_r = w_end - vw_e
    if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
        patch = np.pad(patch, ((pad_t, pad_b), (pad_l, pad_r)), mode="reflect")
    return patch.astype(np.float32)


def build_input_patch(
    multiscale_frames: List[np.ndarray],
    h_start: int,
    w_start: int,
    image_size: int,
    multiscale_lowres_count: int,
) -> np.ndarray:
    """Build the multi-scale input tensor for one tile position.

    Returns
    -------
    ndarray of shape (multiscale_lowres_count, image_size, image_size).
    """
    patches = [
        multiscale_frames[0][
            h_start : h_start + image_size,
            w_start : w_start + image_size,
        ][np.newaxis]
    ]
    h_center = h_start + image_size // 2
    w_center = w_start + image_size // 2

    for scale in range(1, multiscale_lowres_count):
        h_center = h_center // 2
        w_center = w_center // 2
        hs = h_center - image_size // 2
        ws = w_center - image_size // 2
        scaled_patch = extract_patch_padded(
            multiscale_frames[scale], hs, ws, image_size
        )
        patches.append(scaled_patch[np.newaxis])

    return np.concatenate(patches, axis=0)  # (scales, H, W)


# ---------------------------------------------------------------------------
# FOV prediction
# ---------------------------------------------------------------------------

def predict_fov(
    model,
    channel_images: Dict[str, np.ndarray],
    stats: Dict,
    channel_names: Optional[List[str]] = None,
    image_size: int = 64,
    grid_size: int = 32,
    multiscale_lowres_count: int = 3,
    mmse_count: int = 50,
    num_posterior_samples: int = 0,
    posterior_seeds: Tuple[int, ...] = (42, 123, 456),
    batch_size: int = 32,
    use_autocast: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run MicroSplit prediction on a single field of view.

    Parameters
    ----------
    model : VAEModule
        Trained model (already on the appropriate device).
    channel_images : dict
        Mapping channel_name -> ndarray (H, W), raw uint16 pixel values.
    stats : dict
        Normalization statistics as saved by the training script
        (keys: ``mean_input``, ``std_input``, ``mean_target``,
        ``std_target``, ``max_val``).
    channel_names : list of str, optional
        Ordered channel names. Defaults to
        ``["DNA", "RNA", "ER", "AGP", "Mito"]``.
    image_size : int
        Patch size (H=W). Default: 64.
    grid_size : int
        Stride between patches. Default: 32.
    multiscale_lowres_count : int
        Number of resolution scales. Default: 3.
    mmse_count : int
        Number of Monte-Carlo samples for MMSE. Default: 50.
    num_posterior_samples : int
        How many individual posterior samples to return. Default: 0 (MMSE only).
    posterior_seeds : tuple of int
        RNG seeds for each saved posterior sample.
    batch_size : int
        Tile batch size for GPU inference. Default: 32.
    use_autocast : bool
        Run the forward pass under bfloat16 autocast on CUDA (throughput only;
        the model, tiling and MMSE averaging are unchanged). Default: False.

    Returns
    -------
    mmse_prediction : ndarray (H, W, C) float32, denormalized
    posterior_samples : list of ndarray (H, W, C) float32, denormalized
    """
    if channel_names is None:
        channel_names = DEFAULT_CHANNEL_NAMES

    device = next(model.parameters()).device
    max_val    = stats["max_val"]
    mean_input = float(stats["mean_input"].mean())
    std_input  = float(stats["std_input"].mean())
    mean_target = stats["mean_target"].squeeze()
    std_target  = stats["std_target"].squeeze()
    if mean_target.ndim > 1:
        mean_target = mean_target.reshape(-1)
        std_target  = std_target.reshape(-1)

    H, W = next(iter(channel_images.values())).shape
    num_channels = len(channel_names)

    # Combined input = average of the raw channels, then clipped at the training
    # maximum, in the same order as training (combine_channels, then LazyLCDataset).
    # max_val is a single value (true maximum, as in CAREamics); older stats files
    # hold one value per channel with the combined channel last.
    channel_stack = np.stack(
        [channel_images[ch].astype(np.float32) for ch in channel_names], axis=-1
    )  # (H, W, C)
    max_val = np.asarray(max_val, dtype=np.float32)
    input_cap = float(max_val) if max_val.ndim == 0 else float(max_val.reshape(-1)[-1])
    combined = channel_stack.mean(axis=-1).astype(np.float32)
    combined = np.minimum(combined, input_cap)

    ms_frames  = build_multiscale_frames(combined, multiscale_lowres_count)
    h_positions, w_positions = generate_tile_positions(H, W, image_size, grid_size)
    n_tiles    = len(h_positions) * len(w_positions)

    # Build all input patches
    all_patches = np.empty(
        (n_tiles, multiscale_lowres_count, image_size, image_size),
        dtype=np.float32,
    )
    tile_idx = 0
    for h in h_positions:
        for w in w_positions:
            all_patches[tile_idx] = build_input_patch(
                ms_frames, h, w, image_size, multiscale_lowres_count
            )
            tile_idx += 1
    all_patches = (all_patches - mean_input) / std_input

    mmse_accum = np.zeros(
        (n_tiles, num_channels, image_size, image_size), dtype=np.float64
    )
    posterior_tile_samples = [
        np.zeros((n_tiles, num_channels, image_size, image_size), dtype=np.float32)
        for _ in range(num_posterior_samples)
    ]

    model.eval()
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if (use_autocast and device.type == "cuda")
        else contextlib.nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        for sample_idx in range(mmse_count):
            is_saved_sample = sample_idx < num_posterior_samples
            if is_saved_sample:
                torch.manual_seed(posterior_seeds[sample_idx])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(posterior_seeds[sample_idx])

            tiles_out = []
            for bs in range(0, n_tiles, batch_size):
                be  = min(bs + batch_size, n_tiles)
                inp = torch.from_numpy(all_patches[bs:be]).to(device)
                rec, _ = model(inp)
                if model.model.predict_logvar is not None:
                    rec, _ = torch.chunk(rec, 2, dim=1)
                tiles_out.append(rec[:, :num_channels].float().cpu().numpy())

            tiles_out = np.concatenate(tiles_out, axis=0)
            mmse_accum += tiles_out
            if is_saved_sample:
                posterior_tile_samples[sample_idx] = tiles_out.astype(np.float32)

    mmse_tiles = (mmse_accum / mmse_count).astype(np.float32)

    # -----------------------------------------------------------------------
    # Stitching
    # -----------------------------------------------------------------------
    def stitch(tile_predictions: np.ndarray) -> np.ndarray:
        output  = np.zeros((H, W, num_channels), dtype=np.float32)
        overlap = (image_size - grid_size) // 2
        ti = 0
        for h in h_positions:
            for w in w_positions:
                pred   = tile_predictions[ti]
                h_from = overlap if h > 0 else 0
                w_from = overlap if w > 0 else 0
                h_to   = image_size - overlap if (h + image_size < H) else image_size
                w_to   = image_size - overlap if (w + image_size < W) else image_size
                for ch in range(num_channels):
                    output[
                        h + h_from : h + h_to,
                        w + w_from : w + w_to,
                        ch,
                    ] = pred[ch, h_from:h_to, w_from:w_to]
                ti += 1
        return output

    def denormalize(img: np.ndarray) -> np.ndarray:
        for ch in range(num_channels):
            img[..., ch] = img[..., ch] * std_target[ch] + mean_target[ch]
        return img

    mmse_stitched        = denormalize(stitch(mmse_tiles))
    posterior_stitched   = [
        denormalize(stitch(pt)) for pt in posterior_tile_samples
    ]

    return mmse_stitched, posterior_stitched


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_fov_predictions(
    output_dir: str,
    well: str,
    site: int,
    mmse_prediction: np.ndarray,
    posterior_samples: List[np.ndarray],
    channel_names: Optional[List[str]] = None,
    save_uint16: bool = True,
) -> None:
    """Save MMSE + posterior sample TIFFs for one FOV.

    Output structure::

        output_dir/
            mmse/    mmse_{well}_s{site:02d}_{channel}.tif
            sample_0/ sample0_{well}_s{site:02d}_{channel}.tif
            sample_1/ ...

    Parameters
    ----------
    output_dir : str or Path
    well : str
    site : int
    mmse_prediction : ndarray (H, W, C) float32
    posterior_samples : list of ndarray (H, W, C) float32
    channel_names : list of str, optional
    save_uint16 : bool
        If True (default) clip and cast to uint16; otherwise save float32.
    """
    if channel_names is None:
        channel_names = DEFAULT_CHANNEL_NAMES

    # MMSE
    mmse_dir = os.path.join(output_dir, "mmse")
    os.makedirs(mmse_dir, exist_ok=True)
    for ch_idx, ch_name in enumerate(channel_names):
        img = mmse_prediction[..., ch_idx]
        if save_uint16:
            img = np.clip(img, 0, 65535).astype(np.uint16)
        tifffile.imwrite(
            os.path.join(mmse_dir, f"mmse_{well}_s{site:02d}_{ch_name}.tif"),
            img,
        )

    # Posterior samples
    for s_idx, sample in enumerate(posterior_samples):
        sdir = os.path.join(output_dir, f"sample_{s_idx}")
        os.makedirs(sdir, exist_ok=True)
        for ch_idx, ch_name in enumerate(channel_names):
            img = sample[..., ch_idx]
            if save_uint16:
                img = np.clip(img, 0, 65535).astype(np.uint16)
            tifffile.imwrite(
                os.path.join(sdir, f"sample{s_idx}_{well}_s{site:02d}_{ch_name}.tif"),
                img,
            )


def predict_manifest_fovs(
    rows: pd.DataFrame,
    model,
    stats: Dict,
    channel_names: List[str],
    output_dir: str,
    mmse_count: int = 50,
    batch_size: int = 256,
    use_autocast: bool = False,
    grid_size: int = 32,
    save_float32: bool = False,
    resume: bool = True,
) -> Optional[pd.DataFrame]:
    """Predict the FOVs of a split manifest whose rows list per-channel image paths.

    Each FOV's MMSE prediction is written to ``output_dir/mmse/`` and one row is
    appended to ``output_dir/metadata.csv`` as soon as it finishes, so a killed
    job can be resumed. The row holds the manifest's metadata columns, the raw
    image paths, the MMSE file names, PSNR / SSIM per channel and the run time.
    A per-channel ``metrics_summary.csv`` is written at the end.

    Parameters
    ----------
    rows : pandas.DataFrame
        Manifest rows to predict, one per FOV, with ``well``, ``site`` and one
        ``path_{channel}`` column per channel.
    model : VAEModule
        Trained model (already on the appropriate device).
    stats : dict
        Normalization statistics as saved by the training script.
    channel_names : list of str
        Ordered channel names.
    output_dir : str
        Output directory for this plate.
    mmse_count : int
        Number of Monte-Carlo samples for MMSE. Default: 50.
    batch_size : int
        Tile batch size for GPU inference. Default: 256.
    use_autocast : bool
        Run the forward pass under bfloat16 autocast on CUDA. Default: False.
    grid_size : int
        Stride between patches. Default: 32.
    save_float32 : bool
        Save float32 instead of uint16 TIFFs. Default: False.
    resume : bool
        Skip FOVs already present in ``metadata.csv`` with all MMSE files on
        disk. Default: True.

    Returns
    -------
    pandas.DataFrame or None
        Contents of ``metadata.csv`` after the run (None if nothing was written).
    """
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.csv")
    mmse_dir = os.path.join(output_dir, "mmse")

    def mmse_name(well: str, site: int, channel: str) -> str:
        return f"mmse_{well}_s{site:02d}_{channel}.tif"

    done = completed_fovs(meta_path) if resume else set()
    done = {
        (well, site) for well, site in done
        if all(os.path.isfile(os.path.join(mmse_dir, mmse_name(well, site, ch)))
               for ch in channel_names)
    }
    records = rows.to_dict("records")
    todo = [r for r in records if (str(r["well"]), int(r["site"])) not in done]
    meta_columns = [c for c in rows.columns if not c.startswith(PATH_PREFIX)]
    print(f"FOVs: {len(records)} listed, {len(records) - len(todo)} already done, "
          f"{len(todo)} to process")

    t0 = time.time()
    for i, r in enumerate(todo):
        well, site = str(r["well"]), int(r["site"])
        fov_t0 = time.time()
        paths = row_channel_paths(r, channel_names)
        channel_images = read_channel_paths(paths)

        mmse_pred, _ = predict_fov(
            model=model, channel_images=channel_images, stats=stats,
            channel_names=channel_names, image_size=64, grid_size=grid_size,
            multiscale_lowres_count=3, mmse_count=mmse_count, num_posterior_samples=0,
            batch_size=batch_size, use_autocast=use_autocast,
        )
        save_fov_predictions(output_dir, well, site, mmse_pred, [],
                             channel_names=channel_names, save_uint16=not save_float32)
        metrics = compute_metrics(channel_images, mmse_pred, channel_names)

        row = {c: r[c] for c in meta_columns}
        row.update({f"raw_{ch}": paths[ch] for ch in channel_names})
        row["mmse_files"] = ",".join(f"mmse/{mmse_name(well, site, ch)}" for ch in channel_names)
        row.update(metrics)
        row["seconds"] = round(time.time() - fov_t0, 1)
        append_metadata_row(meta_path, row)

        eta = (len(todo) - i - 1) * (time.time() - t0) / (i + 1)
        psnr = " ".join(f"{ch}={metrics[f'psnr_{ch}']:.1f}" for ch in channel_names)
        print(f"[{i + 1}/{len(todo)}] {well} s{site} {row['seconds']:.0f}s | "
              f"ETA {eta / 3600:.1f}h | {psnr}")
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    if not os.path.isfile(meta_path) or os.path.getsize(meta_path) == 0:
        return None
    meta = pd.read_csv(meta_path)
    summary = [{"channel": ch, "mean_psnr": meta[f"psnr_{ch}"].mean(),
                "mean_ssim": meta[f"ssim_{ch}"].mean()} for ch in channel_names]
    summary.append({
        "channel": "all_channels",
        "mean_psnr": meta[[f"psnr_{ch}" for ch in channel_names]].values.mean(),
        "mean_ssim": meta[[f"ssim_{ch}" for ch in channel_names]].values.mean(),
    })
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    return meta
