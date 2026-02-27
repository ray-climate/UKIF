#!/usr/bin/env python3
"""Apply pre-fitted TsHARP model to downscale 1 km SPEI to 10 m.

This script is the recipient-facing end of the TsHARP pipeline.
All calibration (regression fit, residual computation) has been done
upstream. You only need to supply your 10 m Sentinel-2 NDVI GeoTIFFs
for the target months (default: May, June, July).

Method
------
    SPEI_10m = f(NDVI_10m) + upsample(R_1km)

where:
  f(·)   = pre-fitted polynomial regression (NDVI → SPEI) at 1 km
  R_1km  = pre-computed residual (SPEI_observed − f(NDVI)) at 1 km
  NDVI is expressed as an anomaly relative to the 1 km climatological mean

Usage
-----
    python apply_tsHARP_10m.py \\
        --ndvi-files NDVI_May.tif NDVI_Jun.tif NDVI_Jul.tif \\
        --year 2022 \\
        --output-dir ./output_2022

The --year argument selects the matching residual_1km_YEAR.tif from precomputed/.
All other inputs default to the precomputed/ folder bundled with this script.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

LOG = logging.getLogger(__name__)

# Default paths (relative to this script's location)
_HERE = Path(__file__).resolve().parent
DEFAULT_REGRESSION_MODEL = _HERE / "precomputed" / "tsHARP_regression_model.json"
DEFAULT_CLIMATOLOGY_TIF  = _HERE / "precomputed" / "climatology_ndvi_1km.tif"


def default_residual_path(year: int) -> Path:
    """Return the expected residual GeoTIFF path for a given year."""
    return _HERE / "precomputed" / f"residual_1km_{year}.tif"


# ── Raster helpers ────────────────────────────────────────────────────────────

def read_geotiff(path: Path) -> tuple[np.ndarray, Affine, str]:
    """Read a single-band GeoTIFF → (array float32, transform, crs_string)."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32, copy=False)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs.to_string()
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data, transform, crs


def reproject_array(
    source: np.ndarray,
    src_transform: Affine,
    src_crs: str,
    dst_shape: tuple[int, int],
    dst_transform: Affine,
    dst_crs: str,
    resampling: Resampling,
) -> np.ndarray:
    """Reproject a 2-D float32 array to a new grid."""
    destination = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source,
        destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return destination


def save_geotiff(path: Path, array: np.ndarray, transform: Affine, crs: str) -> None:
    """Write a float32 array as a compressed GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="lzw",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32, copy=False), 1)
    LOG.info("  Saved %s", path)


# ── NDVI helpers ──────────────────────────────────────────────────────────────

def read_ndvi_stack(paths: Sequence[Path]) -> tuple[np.ndarray, Affine, str]:
    """Load and stack NDVI GeoTIFFs. All files must share the same grid."""
    arrays, transform, crs = [], None, None
    for i, p in enumerate(paths):
        with rasterio.open(p) as src:
            band = src.read(1).astype(np.float32, copy=False)
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan
            band[~np.isfinite(band)] = np.nan
            # Clip to valid NDVI range
            band[(band < -1.0) | (band > 1.0)] = np.nan
            arrays.append(band)
            if i == 0:
                transform = src.transform
                crs = src.crs.to_string()
            else:
                if src.transform != transform or src.shape != arrays[0].shape:
                    raise ValueError(
                        f"NDVI file {p.name} has a different grid than {paths[0].name}. "
                        "All NDVI files must share the same extent and resolution."
                    )
    return np.stack(arrays, axis=0), transform, crs  # (N_months, H, W)


def compute_metric(stack: np.ndarray, metric: str) -> np.ndarray:
    """Summarise a (N_months, H, W) NDVI stack into a single (H, W) image."""
    metric = metric.lower()
    if metric == "mean":
        with np.errstate(invalid="ignore"):
            result = np.nanmean(stack, axis=0)
        result[~np.isfinite(result)] = np.nan
    elif metric == "delta":
        result = stack[-1] - stack[0]
        result[~np.isfinite(stack[0]) | ~np.isfinite(stack[-1])] = np.nan
    elif metric == "last":
        result = stack[-1].copy()
        result[~np.isfinite(result)] = np.nan
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use mean, delta, or last.")
    result[np.isnan(stack).all(axis=0)] = np.nan
    return result.astype(np.float32, copy=False)


# ── Main downscaling logic ────────────────────────────────────────────────────

def apply_tsharp(
    ndvi_files: list[Path],
    regression_model_path: Path,
    residual_tif_path: Path,
    climatology_tif_path: Path,
    output_dir: Path,
    year: int,
) -> None:
    """Run Phase 4 of TsHARP: apply pre-fitted model to 10 m NDVI."""

    # ── Load regression model ────────────────────────────────────────────
    with regression_model_path.open() as fh:
        model = json.load(fh)
    coeffs    = np.array(model["coefficients"], dtype=np.float64)
    ndvi_var  = model.get("ndvi_variable", "ndvi_anom")
    ndvi_metric = model.get("ndvi_metric", "mean")
    ndvi_months = model.get("ndvi_months", [5, 6, 7])

    LOG.info("Regression model: variable=%s  metric=%s  months=%s  degree=%d  R²=%.4f",
             ndvi_var, ndvi_metric, ndvi_months, model.get("degree", 1),
             model.get("R2", float("nan")))

    if len(ndvi_files) != len(ndvi_months):
        raise ValueError(
            f"Regression was trained on {len(ndvi_months)} months {ndvi_months} "
            f"but {len(ndvi_files)} NDVI file(s) were provided. "
            "Supply one file per month in the same order."
        )

    # ── Load 10 m NDVI ───────────────────────────────────────────────────
    LOG.info("Loading 10 m NDVI (%d files)...", len(ndvi_files))
    ndvi_stack, fine_transform, fine_crs = read_ndvi_stack(ndvi_files)
    LOG.info("  10 m NDVI shape: %s  CRS: %s", ndvi_stack.shape, fine_crs)

    # Raw temporal summary at 10 m
    ndvi_metric_raw = compute_metric(ndvi_stack, ndvi_metric)
    valid_raw = int(np.count_nonzero(np.isfinite(ndvi_metric_raw)))
    LOG.info("  Raw NDVI metric (%s): valid=%d  mean=%.4f  std=%.4f",
             ndvi_metric, valid_raw,
             float(np.nanmean(ndvi_metric_raw)), float(np.nanstd(ndvi_metric_raw)))

    # ── Convert to anomaly space if needed ──────────────────────────────
    if ndvi_var == "ndvi_anom":
        LOG.info("Converting to NDVI anomaly using climatology...")
        clim_1km, clim_transform, clim_crs = read_geotiff(climatology_tif_path)
        LOG.info("  Climatology 1 km shape: %s", clim_1km.shape)

        # Aggregate raw 10 m metric to 1 km to compute sensor offset
        ndvi_agg_1km = reproject_array(
            ndvi_metric_raw, fine_transform, fine_crs,
            clim_1km.shape, clim_transform, clim_crs,
            Resampling.average,
        )
        offset_valid = (ndvi_agg_1km - clim_1km)
        offset_valid = offset_valid[np.isfinite(offset_valid)]
        sensor_offset = float(np.nanmean(offset_valid)) if offset_valid.size > 0 else 0.0
        LOG.info("  Sensor calibration offset (10 m − climatology): %+.4f", sensor_offset)

        # Upsample 1 km climatology to 10 m
        clim_10m = reproject_array(
            clim_1km, clim_transform, clim_crs,
            ndvi_metric_raw.shape, fine_transform, fine_crs,
            Resampling.bilinear,
        )

        # Anomaly = (raw − offset) − climatology
        ndvi_metric_10m = (ndvi_metric_raw - sensor_offset) - clim_10m
        ndvi_metric_10m[~np.isfinite(ndvi_metric_raw) | ~np.isfinite(clim_10m)] = np.nan
        LOG.info("  NDVI anomaly: valid=%d  mean=%.4f  std=%.4f",
                 int(np.count_nonzero(np.isfinite(ndvi_metric_10m))),
                 float(np.nanmean(ndvi_metric_10m)), float(np.nanstd(ndvi_metric_10m)))
    else:
        ndvi_metric_10m = ndvi_metric_raw

    # ── Apply polynomial regression at 10 m ─────────────────────────────
    LOG.info("Applying regression at 10 m...")
    spei_regression_10m = np.polyval(coeffs, ndvi_metric_10m).astype(np.float32)
    spei_regression_10m[~np.isfinite(ndvi_metric_10m)] = np.nan

    # ── Load 1 km residual and upsample to 10 m ─────────────────────────
    LOG.info("Upsampling 1 km residual to 10 m...")
    residual_1km, res_transform, res_crs = read_geotiff(residual_tif_path)
    LOG.info("  Residual 1 km shape: %s", residual_1km.shape)

    residual_10m = reproject_array(
        residual_1km, res_transform, res_crs,
        ndvi_metric_10m.shape, fine_transform, fine_crs,
        Resampling.bilinear,
    )
    LOG.info("  Upsampled residual: valid=%d",
             int(np.count_nonzero(np.isfinite(residual_10m))))

    # ── Combine ──────────────────────────────────────────────────────────
    spei_10m = spei_regression_10m + residual_10m
    spei_10m[~np.isfinite(spei_regression_10m) | ~np.isfinite(residual_10m)] = np.nan

    valid = spei_10m[np.isfinite(spei_10m)]
    LOG.info("10 m SPEI: valid=%d  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
             valid.size, float(valid.mean()), float(valid.std()),
             float(valid.min()), float(valid.max()))

    # ── Save outputs ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out_tif = output_dir / f"spei_10m_{year}.tif"
    out_npy = output_dir / f"spei_10m_{year}.npy"
    save_geotiff(out_tif, spei_10m, fine_transform, fine_crs)
    np.save(out_npy, spei_10m)
    LOG.info("  Saved array: %s", out_npy)

    # ── Quick map ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    vabs = min(2.5, float(np.nanpercentile(np.abs(spei_10m[np.isfinite(spei_10m)]), 99)))

    im0 = axes[0].imshow(ndvi_metric_10m, cmap="Greens",
                         vmin=float(np.nanpercentile(ndvi_metric_10m[np.isfinite(ndvi_metric_10m)], 1)),
                         vmax=float(np.nanpercentile(ndvi_metric_10m[np.isfinite(ndvi_metric_10m)], 99)))
    axes[0].set_title(f"NDVI {ndvi_var} ({ndvi_metric}, 10 m)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    im1 = axes[1].imshow(spei_10m, cmap="RdYlBu", vmin=-vabs, vmax=vabs)
    axes[1].set_title(f"TsHARP SPEI 10 m ({year})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    fig.tight_layout()
    out_png = output_dir / f"spei_10m_{year}_map.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOG.info("  Saved map: %s", out_png)

    LOG.info("Done. Outputs in: %s", output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ndvi-files", nargs="+", type=Path, required=True,
        help=(
            "10 m Sentinel-2 NDVI GeoTIFFs — one file per month, in the same "
            "order as the training months (default: May, June, July)."
        ),
    )
    parser.add_argument(
        "--year", type=int, required=True,
        help=(
            "Target year (e.g. 2022). Used to auto-select precomputed/residual_1km_YEAR.tif "
            "and to name output files. Must match the year of your NDVI input files."
        ),
    )
    parser.add_argument(
        "--regression-model", type=Path, default=DEFAULT_REGRESSION_MODEL,
        help=f"Path to tsHARP_regression_model.json (default: {DEFAULT_REGRESSION_MODEL}).",
    )
    parser.add_argument(
        "--residual-tif", type=Path, default=None,
        help=(
            "Path to the 1 km residual GeoTIFF. "
            "Defaults to precomputed/residual_1km_YEAR.tif where YEAR matches --year."
        ),
    )
    parser.add_argument(
        "--climatology-tif", type=Path, default=DEFAULT_CLIMATOLOGY_TIF,
        help=f"Path to the 1 km NDVI climatology mean GeoTIFF (default: {DEFAULT_CLIMATOLOGY_TIF}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tsHARP_output"),
        help="Directory for output files (default: ./tsHARP_output).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )

    # Resolve residual path: explicit override or auto-select by year
    residual_tif = args.residual_tif or default_residual_path(args.year)
    if not residual_tif.exists():
        LOG.error(
            "Residual file not found for year %d: %s\n"
            "Available residuals: %s",
            args.year, residual_tif,
            ", ".join(str(p.name) for p in sorted(
                (_HERE / "precomputed").glob("residual_1km_*.tif")
            )) or "none"
        )
        return 1

    # Validate other inputs
    for p in [args.regression_model, args.climatology_tif]:
        if not p.exists():
            LOG.error("File not found: %s", p)
            return 1

    ndvi_files = [Path(f) for f in args.ndvi_files]
    for p in ndvi_files:
        if not p.exists():
            LOG.error("NDVI file not found: %s", p)
            return 1

    apply_tsharp(
        ndvi_files=ndvi_files,
        regression_model_path=args.regression_model,
        residual_tif_path=residual_tif,
        climatology_tif_path=args.climatology_tif,
        output_dir=args.output_dir.expanduser().resolve(),
        year=args.year,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
