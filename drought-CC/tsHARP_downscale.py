#!/usr/bin/env python3
"""TsHARP-style downscaling of 1 km SPEI to 10 m using Sentinel-2 NDVI.

Method:
    At 1km:  f(NDVI_1km) -> SPEI_hat     (polynomial regression)
             R_1km = SPEI_observed - SPEI_hat   (residual)
    At 10m:  SPEI_10m = f(NDVI_10m) + R_1km_upsampled

The residual term guarantees mass conservation when aggregated back to 1km.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling
from scipy import stats as sp_stats

from apply_beta_ndvi_correction import (
    clean_ndvi,
    compute_metric,
    load_meta,
    plot_zoom_comparison,
    read_ndvi_stack,
    reproject_array,
    save_geotiff,
)
from calibrate_ndvi_beta import (
    load_feature_archive,
    subset_stack,
)

LOG = logging.getLogger(__name__)

# Variables that require climatology subtraction before applying at 10 m
_ANOM_VARIABLES = {"ndvi_anom"}

# ── Climatology helpers ──────────────────────────────────────────────────


def load_climatology_ndvi_mean(
    climatology_path: Path,
    ndvi_months: Sequence[int],
    ndvi_metric: str,
) -> tuple[np.ndarray, Affine, str]:
    """Load 1 km NDVI climatological mean and compute temporal summary.

    Returns (ndvi_clim_summary_2d, transform, crs) where the summary has
    the same temporal aggregation (mean/delta/last) applied over the
    requested months.
    """
    with np.load(climatology_path) as data:
        clim_mean = data["climatology_mean"].astype(np.float32, copy=False)
        months = [int(v) for v in data["months"]]
        variables = [str(v).lower() for v in data["variables"]]
        transform = Affine.from_gdal(*data["transform"])
        crs_raw = data["crs"]
        if isinstance(crs_raw, np.ndarray):
            crs_str = str(crs_raw.item()) if crs_raw.shape == () else str(crs_raw.tolist())
        else:
            crs_str = str(crs_raw)

    # clim_mean shape: (M_all, H, W, V) — extract NDVI for requested months
    if "ndvi" not in variables:
        raise ValueError(f"Climatology lacks 'ndvi'; available: {variables}")
    var_idx = variables.index("ndvi")

    month_indices = []
    for m in ndvi_months:
        if m not in months:
            raise ValueError(f"Month {m} not in climatology (available: {months})")
        month_indices.append(months.index(m))

    # (len(ndvi_months), H, W)
    ndvi_clim = clim_mean[month_indices, :, :, var_idx]
    summary = _compute_summary_metric_1km(ndvi_clim, ndvi_metric)
    LOG.info(
        "  Climatology NDVI (%s, months=%s): mean=%.4f  std=%.4f",
        ndvi_metric, list(ndvi_months),
        float(np.nanmean(summary)), float(np.nanstd(summary)),
    )
    return summary, transform, crs_str


# ── Phase 1: Diagnostic ─────────────────────────────────────────────────

NDVI_VARIABLES = ("ndvi", "ndvi_anom", "ndvi_deficit")


def _compute_summary_metric_1km(
    archive_stack: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Compute a summary metric over months (first axis) of a 3-D array (M, H, W).

    Parameters
    ----------
    archive_stack : (M, H, W) array from subset_stack with variable axis squeezed.
    metric : 'mean', 'delta', or 'last'.
    """
    metric = metric.lower()
    if metric == "mean":
        with np.errstate(invalid="ignore"):
            result = np.nanmean(archive_stack, axis=0)
        result[~np.isfinite(result)] = np.nan
    elif metric == "delta":
        result = archive_stack[-1] - archive_stack[0]
        invalid = ~np.isfinite(archive_stack[0]) | ~np.isfinite(archive_stack[-1])
        result[invalid] = np.nan
    elif metric == "last":
        result = archive_stack[-1].copy()
        result[~np.isfinite(result)] = np.nan
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return result.astype(np.float32)


def run_diagnostic(
    feature_dir: Path,
    target_dir: Path,
    years: Sequence[int],
    ndvi_months: Sequence[int],
    ndvi_metric: str,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """Phase 1 — compute NDVI-SPEI correlations at 1 km for each NDVI variable."""
    LOG.info("=== Phase 1: Diagnostic (NDVI-SPEI correlation at 1 km) ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, float]] = {}

    for var in NDVI_VARIABLES:
        all_ndvi: list[np.ndarray] = []
        all_spei: list[np.ndarray] = []

        for year in years:
            archive = load_feature_archive(feature_dir / f"X_{year}.npz")
            target = np.load(target_dir / f"y_{year}.npy").astype(np.float32)

            try:
                ndvi_stack = subset_stack(archive, ndvi_months, [var])
            except ValueError:
                LOG.warning("Variable '%s' not in archive for year %d — skipping", var, year)
                continue
            ndvi_3d = ndvi_stack[..., 0]  # (M, H, W)
            summary = _compute_summary_metric_1km(ndvi_3d, ndvi_metric)

            mask = np.isfinite(summary) & np.isfinite(target)
            all_ndvi.append(summary[mask])
            all_spei.append(target[mask])

        if not all_ndvi:
            LOG.warning("No valid data for variable '%s'", var)
            continue

        ndvi_flat = np.concatenate(all_ndvi)
        spei_flat = np.concatenate(all_spei)

        r, p = sp_stats.pearsonr(ndvi_flat, spei_flat)
        r2 = r ** 2
        results[var] = {"r": float(r), "r2": float(r2), "p_value": float(p), "n": int(ndvi_flat.size)}
        LOG.info("  %-14s  r=%+.4f  R²=%.4f  p=%.2e  n=%d", var, r, r2, p, ndvi_flat.size)

        # Hexbin scatter plot
        fig, ax = plt.subplots(figsize=(6, 5))
        hb = ax.hexbin(ndvi_flat, spei_flat, gridsize=80, cmap="YlOrRd", mincnt=1)
        fig.colorbar(hb, ax=ax, label="count")
        ax.set_xlabel(f"{var} ({ndvi_metric}, May-Jul)")
        ax.set_ylabel("SPEI (observed)")
        ax.set_title(f"{var} vs SPEI at 1 km\nr={r:+.4f}, R²={r2:.4f}, p={p:.2e}")
        fig.tight_layout()
        fig.savefig(output_dir / f"diagnostic_{var}.png", dpi=180)
        plt.close(fig)

    # Print table
    LOG.info("  %-14s  %8s  %8s  %10s  %8s", "Variable", "r", "R²", "p-value", "n")
    LOG.info("  %s", "-" * 56)
    for var, vals in results.items():
        LOG.info(
            "  %-14s  %+8.4f  %8.4f  %10.2e  %8d",
            var, vals["r"], vals["r2"], vals["p_value"], vals["n"],
        )

    with open(output_dir / "diagnostic_correlations.json", "w") as fh:
        json.dump(results, fh, indent=2)
    LOG.info("Diagnostic outputs saved to %s", output_dir)
    return results


def auto_select_variable(results: dict[str, dict[str, float]]) -> str:
    """Select the NDVI variable with highest |r|."""
    best_var = max(results, key=lambda v: abs(results[v]["r"]))
    LOG.info("Auto-selected variable: '%s' (|r|=%.4f)", best_var, abs(results[best_var]["r"]))
    return best_var


# ── Phase 2: Fit regression at 1 km ────────────────────────────────────

def fit_regression(
    feature_dir: Path,
    target_dir: Path,
    years: Sequence[int],
    ndvi_months: Sequence[int],
    ndvi_metric: str,
    ndvi_variable: str,
    regression_type: str,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Phase 2 — fit polynomial regression NDVI -> SPEI at 1 km."""
    LOG.info("=== Phase 2: Fit regression at 1 km ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    degree = 1 if regression_type == "linear" else 2

    all_ndvi: list[np.ndarray] = []
    all_spei: list[np.ndarray] = []

    for year in years:
        archive = load_feature_archive(feature_dir / f"X_{year}.npz")
        target = np.load(target_dir / f"y_{year}.npy").astype(np.float32)
        ndvi_stack = subset_stack(archive, ndvi_months, [ndvi_variable])[..., 0]
        summary = _compute_summary_metric_1km(ndvi_stack, ndvi_metric)
        mask = np.isfinite(summary) & np.isfinite(target)
        all_ndvi.append(summary[mask])
        all_spei.append(target[mask])

    ndvi_flat = np.concatenate(all_ndvi)
    spei_flat = np.concatenate(all_spei)

    coeffs = np.polyfit(ndvi_flat, spei_flat, degree)
    spei_hat = np.polyval(coeffs, ndvi_flat)
    residuals = spei_flat - spei_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((spei_flat - np.mean(spei_flat)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    info: dict[str, Any] = {
        "ndvi_variable": ndvi_variable,
        "ndvi_metric": ndvi_metric,
        "ndvi_months": list(ndvi_months),
        "regression_type": regression_type,
        "degree": degree,
        "coefficients": [float(c) for c in coeffs],
        "R2": r2,
        "RMSE": rmse,
        "n_samples": int(ndvi_flat.size),
        "years": list(years),
    }
    LOG.info(
        "  Regression: degree=%d  R²=%.4f  RMSE=%.4f  n=%d",
        degree, r2, rmse, ndvi_flat.size,
    )
    LOG.info("  Coefficients: %s", coeffs)

    with open(output_dir / "tsHARP_regression_model.json", "w") as fh:
        json.dump(info, fh, indent=2)

    # Plot fitted curve over scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(ndvi_flat, spei_flat, gridsize=80, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax, label="count")
    x_sort = np.linspace(float(np.nanmin(ndvi_flat)), float(np.nanmax(ndvi_flat)), 200)
    ax.plot(x_sort, np.polyval(coeffs, x_sort), "b-", lw=2, label=f"fit (R²={r2:.4f})")
    ax.set_xlabel(f"{ndvi_variable} ({ndvi_metric})")
    ax.set_ylabel("SPEI (observed)")
    ax.set_title(f"TsHARP regression at 1 km ({regression_type})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tsHARP_regression_fit.png", dpi=180)
    plt.close(fig)

    return coeffs, info


# ── Phase 3: Compute 1 km residual ─────────────────────────────────────

def compute_residual_1km(
    feature_dir: Path,
    target_dir: Path,
    prediction_year: int,
    ndvi_months: Sequence[int],
    ndvi_metric: str,
    ndvi_variable: str,
    coeffs: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase 3 — compute residual R = SPEI_obs - f(NDVI) at 1 km for prediction year.

    Returns (ndvi_summary_1km, spei_hat_1km, residual_1km).
    """
    LOG.info("=== Phase 3: Compute 1 km residual for year %d ===", prediction_year)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive = load_feature_archive(feature_dir / f"X_{prediction_year}.npz")
    target = np.load(target_dir / f"y_{prediction_year}.npy").astype(np.float32)
    meta_transform, meta_crs, meta_shape = load_meta(
        feature_dir / f"X_{prediction_year}_meta.json"
    )

    ndvi_stack = subset_stack(archive, ndvi_months, [ndvi_variable])[..., 0]
    ndvi_summary = _compute_summary_metric_1km(ndvi_stack, ndvi_metric)

    spei_hat = np.polyval(coeffs, ndvi_summary).astype(np.float32)
    residual = target - spei_hat
    invalid = ~np.isfinite(ndvi_summary) | ~np.isfinite(target)
    spei_hat[invalid] = np.nan
    residual[invalid] = np.nan

    res_valid = residual[np.isfinite(residual)]
    LOG.info(
        "  Residual stats: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        float(np.mean(res_valid)), float(np.std(res_valid)),
        float(np.min(res_valid)), float(np.max(res_valid)),
    )

    # Save outputs
    np.save(output_dir / f"residual_1km_{prediction_year}.npy", residual)
    save_geotiff(
        output_dir / f"spei_observed_1km_{prediction_year}.tif",
        target, meta_transform, meta_crs,
    )
    save_geotiff(
        output_dir / f"residual_1km_{prediction_year}.tif",
        residual, meta_transform, meta_crs,
    )
    save_geotiff(
        output_dir / f"spei_regression_1km_{prediction_year}.tif",
        spei_hat, meta_transform, meta_crs,
    )

    # Residual spatial map
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vabs = max(abs(float(np.nanmin(target))), abs(float(np.nanmax(target))))

    im0 = axes[0].imshow(target, cmap="RdYlBu", vmin=-vabs, vmax=vabs)
    axes[0].set_title(f"Observed SPEI ({prediction_year})")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    im1 = axes[1].imshow(spei_hat, cmap="RdYlBu", vmin=-vabs, vmax=vabs)
    axes[1].set_title("f(NDVI) regression estimate")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    res_abs = max(abs(float(np.nanmin(res_valid))), abs(float(np.nanmax(res_valid))))
    im2 = axes[2].imshow(residual, cmap="RdBu", vmin=-res_abs, vmax=res_abs)
    axes[2].set_title("Residual (observed - regression)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)

    fig.suptitle(f"TsHARP Phase 3: 1 km regression & residual ({prediction_year})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / f"residual_map_{prediction_year}.png", dpi=180)
    plt.close(fig)

    return ndvi_summary, spei_hat, residual


# ── Phase 4: Downscale to 10 m ─────────────────────────────────────────

def downscale_to_10m(
    ndvi_files: Sequence[Path],
    ndvi_metric_name: str,
    ndvi_variable: str,
    coeffs: np.ndarray,
    residual_1km: np.ndarray,
    feature_meta_path: Path,
    prediction_year: int,
    output_dir: Path,
    climatology_path: Path | None = None,
    ndvi_months: Sequence[int] = (5, 6, 7),
) -> tuple[np.ndarray, np.ndarray, Affine, str]:
    """Phase 4 — apply TsHARP downscaling to produce 10 m SPEI.

    When *ndvi_variable* is an anomaly-based variable (e.g. ``ndvi_anom``),
    the 10 m raw NDVI is converted to the same variable space by subtracting
    the upsampled 1 km climatological mean before the regression is applied.

    Returns (spei_10m, ndvi_metric_10m, fine_transform, fine_crs).
    """
    LOG.info("=== Phase 4: Downscale to 10 m ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load 10 m NDVI
    ndvi_stack, fine_transform, fine_crs = read_ndvi_stack(ndvi_files)
    LOG.info("  10 m NDVI stack shape: %s", ndvi_stack.shape)

    # Compute raw metric at 10 m (reuse compute_metric from apply_beta)
    ndvi_metric_10m_raw = compute_metric(ndvi_stack, ndvi_metric_name)
    LOG.info(
        "  10 m raw NDVI metric: valid=%d  mean=%.4f  std=%.4f",
        int(np.count_nonzero(np.isfinite(ndvi_metric_10m_raw))),
        float(np.nanmean(ndvi_metric_10m_raw)),
        float(np.nanstd(ndvi_metric_10m_raw)),
    )

    # Convert to anomaly space if needed
    if ndvi_variable in _ANOM_VARIABLES:
        if climatology_path is None:
            raise ValueError(
                f"Variable '{ndvi_variable}' requires --climatology-path to convert "
                "10 m NDVI to anomaly space."
            )
        LOG.info("  Converting 10 m NDVI to anomaly space (variable=%s)", ndvi_variable)
        clim_summary_1km, clim_transform, clim_crs = load_climatology_ndvi_mean(
            climatology_path, ndvi_months, ndvi_metric_name,
        )
        # Upsample 1 km climatology to 10 m grid
        clim_10m = reproject_array(
            clim_summary_1km,
            clim_transform, clim_crs,
            ndvi_metric_10m_raw.shape,
            fine_transform, fine_crs,
            Resampling.bilinear,
        )

        # Compute sensor-calibration offset: aggregate 10 m raw NDVI to
        # 1 km, then compare with the MODIS climatology at the same scale.
        # Subtracting this offset removes the systematic bias between the
        # 10 m sensor (e.g. Sentinel-2 albedo product) and the MODIS-based
        # climatology, so the anomaly truly reflects spatial deviation.
        ndvi_raw_at_1km = reproject_array(
            ndvi_metric_10m_raw,
            fine_transform, fine_crs,
            clim_summary_1km.shape,
            clim_transform, clim_crs,
            Resampling.average,
        )
        offset_1km = ndvi_raw_at_1km - clim_summary_1km
        offset_valid = offset_1km[np.isfinite(offset_1km)]
        sensor_offset = float(np.nanmean(offset_valid)) if offset_valid.size > 0 else 0.0
        LOG.info(
            "  Sensor calibration offset (10 m mean − climatology): %+.4f "
            "(will be removed from 10 m NDVI before anomaly computation)",
            sensor_offset,
        )

        ndvi_metric_10m = (ndvi_metric_10m_raw - sensor_offset) - clim_10m
        ndvi_metric_10m[~np.isfinite(ndvi_metric_10m_raw) | ~np.isfinite(clim_10m)] = np.nan
        LOG.info(
            "  10 m NDVI anomaly: valid=%d  mean=%.4f  std=%.4f",
            int(np.count_nonzero(np.isfinite(ndvi_metric_10m))),
            float(np.nanmean(ndvi_metric_10m)),
            float(np.nanstd(ndvi_metric_10m)),
        )
    else:
        ndvi_metric_10m = ndvi_metric_10m_raw

    # Apply regression at 10 m
    spei_regression_10m = np.polyval(coeffs, ndvi_metric_10m).astype(np.float32)
    spei_regression_10m[~np.isfinite(ndvi_metric_10m)] = np.nan

    # Upsample residual from 1 km to 10 m via bilinear interpolation
    coarse_transform, coarse_crs, _ = load_meta(feature_meta_path)
    residual_upsampled = reproject_array(
        residual_1km,
        coarse_transform, coarse_crs,
        ndvi_metric_10m.shape,
        fine_transform, fine_crs,
        Resampling.bilinear,
    )
    LOG.info(
        "  Upsampled residual: valid=%d",
        int(np.count_nonzero(np.isfinite(residual_upsampled))),
    )

    # Combine: SPEI_10m = f(NDVI_10m) + R_1km_upsampled
    spei_10m = spei_regression_10m + residual_upsampled
    spei_10m[~np.isfinite(spei_regression_10m) | ~np.isfinite(residual_upsampled)] = np.nan

    valid_10m = spei_10m[np.isfinite(spei_10m)]
    LOG.info(
        "  10 m SPEI: valid=%d  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        valid_10m.size,
        float(np.mean(valid_10m)), float(np.std(valid_10m)),
        float(np.min(valid_10m)), float(np.max(valid_10m)),
    )

    # Save outputs
    np.save(output_dir / f"tsHARP_spei_10m_{prediction_year}.npy", spei_10m)
    save_geotiff(
        output_dir / f"tsHARP_spei_10m_{prediction_year}.tif",
        spei_10m, fine_transform, fine_crs,
    )
    save_geotiff(
        output_dir / f"tsHARP_regression_10m_{prediction_year}.tif",
        spei_regression_10m, fine_transform, fine_crs,
    )
    save_geotiff(
        output_dir / f"residual_upsampled_10m_{prediction_year}.tif",
        residual_upsampled, fine_transform, fine_crs,
    )

    return spei_10m, ndvi_metric_10m_raw, fine_transform, fine_crs


# ── Phase 5: Validation ────────────────────────────────────────────────

def validate(
    spei_10m: np.ndarray,
    ndvi_metric_10m: np.ndarray,
    fine_transform: Affine,
    fine_crs: str,
    observed_spei_1km: np.ndarray,
    coarse_transform: Affine,
    coarse_crs: str,
    coarse_shape: tuple[int, int],
    prediction_year: int,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Phase 5 — mass conservation check and zoom comparison."""
    LOG.info("=== Phase 5: Validation ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate 10 m SPEI back to 1 km via average resampling
    spei_agg = reproject_array(
        spei_10m,
        fine_transform, fine_crs,
        coarse_shape,
        coarse_transform, coarse_crs,
        Resampling.average,
    )

    # Mass conservation: compare aggregated 10 m with observed 1 km
    mask = np.isfinite(spei_agg) & np.isfinite(observed_spei_1km)
    n_valid = int(np.count_nonzero(mask))
    if n_valid > 1:
        r, p = sp_stats.pearsonr(spei_agg[mask], observed_spei_1km[mask])
        rmse = float(np.sqrt(np.mean((spei_agg[mask] - observed_spei_1km[mask]) ** 2)))
        mae = float(np.mean(np.abs(spei_agg[mask] - observed_spei_1km[mask])))
        bias = float(np.mean(spei_agg[mask] - observed_spei_1km[mask]))
    else:
        r, p, rmse, mae, bias = 0.0, 1.0, float("nan"), float("nan"), float("nan")

    LOG.info("  Mass conservation check:")
    LOG.info("    Pearson r   = %.4f  (p=%.2e)", r, p)
    LOG.info("    RMSE        = %.4f", rmse)
    LOG.info("    MAE         = %.4f", mae)
    LOG.info("    Mean bias   = %.4f", bias)
    LOG.info("    Valid pixels = %d", n_valid)

    report: dict[str, Any] = {
        "mass_conservation": {
            "pearson_r": float(r),
            "p_value": float(p),
            "RMSE": rmse,
            "MAE": mae,
            "mean_bias": bias,
            "n_pixels": n_valid,
        },
        "spei_10m_stats": {
            "valid": int(np.count_nonzero(np.isfinite(spei_10m))),
            "mean": float(np.nanmean(spei_10m)),
            "std": float(np.nanstd(spei_10m)),
            "min": float(np.nanmin(spei_10m)),
            "max": float(np.nanmax(spei_10m)),
        },
        "prediction_year": prediction_year,
    }

    with open(output_dir / "tsHARP_validation_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    # Mass conservation scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    if n_valid > 0:
        hb = ax.hexbin(
            observed_spei_1km[mask], spei_agg[mask],
            gridsize=80, cmap="YlOrRd", mincnt=1,
        )
        fig.colorbar(hb, ax=ax, label="count")
        lims = [
            min(float(np.nanmin(observed_spei_1km[mask])), float(np.nanmin(spei_agg[mask]))),
            max(float(np.nanmax(observed_spei_1km[mask])), float(np.nanmax(spei_agg[mask]))),
        ]
        ax.plot(lims, lims, "k--", lw=1, label="1:1")
        ax.legend()
    ax.set_xlabel("Observed SPEI (1 km)")
    ax.set_ylabel("Aggregated TsHARP SPEI (10 m -> 1 km)")
    ax.set_title(f"Mass conservation check (r={r:.4f})")
    fig.tight_layout()
    fig.savefig(output_dir / "mass_conservation_check.png", dpi=180)
    plt.close(fig)

    # Zoom comparison panels: NDVI 10m | SPEI 1km | TsHARP SPEI 10m
    coarse_upsampled = reproject_array(
        observed_spei_1km,
        coarse_transform, coarse_crs,
        spei_10m.shape,
        fine_transform, fine_crs,
        Resampling.bilinear,
    )
    try:
        plot_zoom_comparison(
            observed_spei_1km,
            coarse_upsampled,
            spei_10m,
            ndvi_metric_10m,
            coarse_transform,
            fine_transform,
            f"TsHARP Downscaling: NDVI | SPEI 1km | TsHARP SPEI 10m ({prediction_year})",
            output_dir / "zoom_comparison.png",
            rng=random.Random(seed),
            coarse_pixels_per_side=1,
        )
    except RuntimeError as exc:
        LOG.warning(
            "Zoom comparison skipped: %s (10 m NDVI may cover only a subregion)", exc
        )

    LOG.info("Validation outputs saved to %s", output_dir)
    return report


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2020, 2021, 2022],
        help="Training years for regression fit (default: 2020 2021 2022).",
    )
    parser.add_argument(
        "--prediction-year", type=int, default=2022,
        help="Year to produce downscaled map for (default: 2022).",
    )
    parser.add_argument(
        "--ndvi-months", nargs="+", type=int, default=[5, 6, 7],
        help="Months used to compute NDVI summary (default: 5 6 7 = May-Jul).",
    )
    parser.add_argument(
        "--ndvi-metric", choices=["mean", "delta", "last"], default="mean",
        help="Summary metric over months (default: mean).",
    )
    parser.add_argument(
        "--ndvi-variable", default="auto",
        help=(
            "NDVI variable to use: 'auto' selects highest |r| from diagnostic, "
            "or specify ndvi/ndvi_anom/ndvi_deficit directly (default: auto)."
        ),
    )
    parser.add_argument(
        "--regression-type", choices=["linear", "quadratic"], default="linear",
        help="Regression degree: linear (degree 1) or quadratic (degree 2).",
    )
    parser.add_argument(
        "--ndvi-files", nargs="+", type=Path, required=True,
        help="Sentinel-2 10 m NDVI GeoTIFFs (one per month, e.g. May Jun Jul).",
    )
    parser.add_argument(
        "--feature-dir", type=Path, default=Path("prepared_inputs_uk"),
        help="Directory containing X_YEAR.npz archives.",
    )
    parser.add_argument(
        "--target-dir", type=Path, default=Path("prepared_targets_uk"),
        help="Directory containing y_YEAR.npy targets.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tsHARP_output"),
        help="Output directory (default: tsHARP_output).",
    )
    parser.add_argument(
        "--climatology-path", type=Path,
        help=(
            "Path to climatology NPZ (X_climatology_*.npz). Required when "
            "using ndvi_anom to convert 10 m NDVI to anomaly space. "
            "Default: <feature-dir>/X_climatology_mjja_ndvi-lst-precip.npz"
        ),
    )
    parser.add_argument(
        "--diagnostic-only", action="store_true",
        help="Run Phase 1 only (correlation diagnostics) and exit.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for zoom panels.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(levelname)s] %(message)s",
    )

    feature_dir = args.feature_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Diagnostic ──────────────────────────────────────────
    diag_results = run_diagnostic(
        feature_dir, target_dir,
        args.years, args.ndvi_months, args.ndvi_metric,
        output_dir,
    )

    if not diag_results:
        LOG.error("No diagnostic results — check that feature archives contain NDVI variables.")
        return 1

    if args.diagnostic_only:
        LOG.info("--diagnostic-only: stopping after Phase 1.")
        return 0

    # ── Resolve NDVI variable ────────────────────────────────────────
    if args.ndvi_variable == "auto":
        ndvi_variable = auto_select_variable(diag_results)
    else:
        ndvi_variable = args.ndvi_variable.lower()
        if ndvi_variable not in diag_results:
            LOG.warning(
                "Variable '%s' not in diagnostics; proceeding anyway.", ndvi_variable
            )

    # ── Phase 2: Fit regression ──────────────────────────────────────
    coeffs, reg_info = fit_regression(
        feature_dir, target_dir,
        args.years, args.ndvi_months, args.ndvi_metric,
        ndvi_variable, args.regression_type,
        output_dir,
    )

    # ── Phase 3: Compute 1 km residual ───────────────────────────────
    prediction_year = args.prediction_year
    feature_meta_path = feature_dir / f"X_{prediction_year}_meta.json"
    coarse_transform, coarse_crs, coarse_shape = load_meta(feature_meta_path)

    ndvi_summary_1km, spei_hat_1km, residual_1km = compute_residual_1km(
        feature_dir, target_dir,
        prediction_year, args.ndvi_months, args.ndvi_metric,
        ndvi_variable, coeffs,
        output_dir,
    )

    # ── Resolve climatology path ─────────────────────────────────────
    climatology_path = None
    if ndvi_variable in _ANOM_VARIABLES:
        if args.climatology_path:
            climatology_path = args.climatology_path.expanduser().resolve()
        else:
            # Auto-detect from feature_dir
            candidates = sorted(feature_dir.glob("X_climatology*.npz"))
            if candidates:
                climatology_path = candidates[0]
                LOG.info("Auto-detected climatology: %s", climatology_path)
            else:
                LOG.error(
                    "Variable '%s' requires a climatology file but none found in %s. "
                    "Provide --climatology-path explicitly.",
                    ndvi_variable, feature_dir,
                )
                return 1

    # ── Phase 4: Downscale to 10 m ───────────────────────────────────
    spei_10m, ndvi_metric_10m, fine_transform, fine_crs = downscale_to_10m(
        args.ndvi_files, args.ndvi_metric, ndvi_variable,
        coeffs, residual_1km,
        feature_meta_path, prediction_year,
        output_dir,
        climatology_path=climatology_path,
        ndvi_months=args.ndvi_months,
    )

    # ── Phase 5: Validation ──────────────────────────────────────────
    observed_spei = np.load(target_dir / f"y_{prediction_year}.npy").astype(np.float32)

    report = validate(
        spei_10m, ndvi_metric_10m,
        fine_transform, fine_crs,
        observed_spei,
        coarse_transform, coarse_crs, coarse_shape,
        prediction_year,
        output_dir,
        seed=args.seed,
    )

    LOG.info("=== TsHARP downscaling complete ===")
    LOG.info("Outputs in: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
