#!/usr/bin/env python3
"""Generate UK SPEI prediction maps from trained XGBoost model.

This script uses a trained XGBoost model to predict SPEI (drought) values
across the UK. It can optionally compare predictions with observed SPEI data.

Usage:
    python step_4_predict_map_uk.py --year 2022 --model-path may_jun_jul/trained_xgb_model.json
"""

import argparse
import json
import logging
import sys
from calendar import month_abbr
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = BASE_DIR / "prepared_inputs_uk"
DEFAULT_TARGET_DIR = BASE_DIR / "prepared_targets_uk"
DEFAULT_MODEL_DIR = DEFAULT_FEATURE_DIR / "may_jun_jul"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "trained_xgb_model.json"

# Colormap for SPEI (RdYlBu: negative=red, positive=blue)
SPEI_CMAP = plt.get_cmap("RdYlBu")


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Calendar year to predict (default: 2022)."
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        help=(
            "Path to feature stack NPZ file "
            "(default: prepared_inputs_uk/X_<year>.npz)."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to trained XGBoost model "
            "(default: prepared_inputs_uk/may_jun_jul/trained_xgb_model.json)."
        )
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        help=(
            "Optional SPEI target array for comparison "
            "(default: prepared_targets_uk/y_<year>.npy)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for outputs (default: same as model)."
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-2.5,
        help="Minimum value for SPEI colormap (default: -2.5)."
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=2.5,
        help="Maximum value for SPEI colormap (default: 2.5)."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)."
    )
    parser.add_argument(
        "--require-ndvi",
        action="store_true",
        help="Require NDVI values to be present and finite for prediction pixels."
    )
    parser.add_argument(
        "--mask-variable",
        type=str,
        help=(
            "Optional variable name that must be finite for a pixel to be predicted "
            "(e.g., ndvi to mask oceans)."
        )
    )
    return parser.parse_args(argv)


# ============================================================================
# File Loading
# ============================================================================

def load_feature_stack(path: Path) -> tuple[np.ndarray, dict]:
    """Load feature stack from NPZ file."""
    if not path.exists():
        raise FileNotFoundError(f"Feature stack not found: {path}")

    logging.info(f"Loading features from {path.name}")

    with np.load(path) as data:
        if "X" not in data.files:
            raise ValueError(f"NPZ file {path} does not contain 'X' array")

        stack = data["X"]

        # Extract metadata
        meta = {"shape": list(stack.shape)}

        if "months" in data.files:
            months = data["months"]
            meta["months"] = [int(m) for m in months]

        if "variables" in data.files:
            variables = data["variables"]
            meta["variables"] = [str(v) for v in variables]

        if "transform" in data.files:
            meta["transform"] = list(data["transform"])

        if "crs" in data.files:
            crs_val = data["crs"]
            meta["crs"] = str(crs_val) if isinstance(crs_val, str) else str(crs_val.item())

    if stack.ndim != 4:
        raise ValueError(
            f"Expected stack with shape (months, rows, cols, vars); "
            f"got {stack.shape}"
        )

    return stack.astype(np.float32, copy=False), meta


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with path.open("r", encoding="utf8") as fh:
        return json.load(fh)


def load_target(path: Path) -> np.ndarray:
    """Load target SPEI array."""
    if not path.exists():
        raise FileNotFoundError(f"Target file not found: {path}")
    return np.load(path).astype(np.float32, copy=False)


# ============================================================================
# Metadata Utilities
# ============================================================================

def extract_transform(meta: dict) -> Optional[Affine]:
    """Extract Affine transform from metadata."""
    transform = meta.get("transform")
    if not transform:
        return None
    return Affine.from_gdal(*transform)


def extract_crs(meta: dict) -> Optional[CRS]:
    """Extract CRS from metadata."""
    crs_str = meta.get("crs")
    if not crs_str:
        return None
    return CRS.from_string(crs_str)


def build_feature_names(months: list[int], variables: list[str]) -> list[str]:
    """Build feature names for XGBoost."""
    names = []
    for month in months:
        month_label = month_abbr[month] if 1 <= month <= 12 else f"M{month:02d}"
        for var in variables:
            names.append(f"{month_label}_{var.upper()}")
    return names


def load_model_metadata(model_path: Path) -> Optional[dict]:
    """Load metadata (metrics) saved alongside the trained model."""
    metrics_path = model_path.with_name("trained_xgb_metrics.json")
    if not metrics_path.exists():
        logging.info("Model metrics file not found; using full feature stack")
        return None
    try:
        meta = load_json(metrics_path)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning(f"Failed to load model metadata: {exc}")
        return None
    logging.info(f"Loaded model metadata from {metrics_path.name}")
    return meta


def align_stack_to_model(
    stack: np.ndarray,
    stack_months: list[int],
    stack_variables: list[str],
    model_meta: Optional[dict]
) -> tuple[np.ndarray, list[int], list[str], Optional[list[str]]]:
    """Align feature stack to match the months/variables used during training."""
    if not model_meta:
        return stack, stack_months, stack_variables, None

    aligned_stack = stack
    aligned_months = stack_months
    aligned_variables = stack_variables
    feature_labels = model_meta.get("feature_labels")

    desired_months = model_meta.get("months") or []
    if desired_months:
        month_lookup = {month: idx for idx, month in enumerate(stack_months)}
        missing = [month for month in desired_months if month not in month_lookup]
        if missing:
            raise ValueError(
                f"Feature stack missing months required by model: {missing}"
            )
        month_indices = [month_lookup[month] for month in desired_months]
        aligned_stack = aligned_stack[month_indices, :, :, :]
        aligned_months = desired_months
        logging.info(
            "Subset feature stack to months used in training: %s",
            ", ".join(str(m) for m in desired_months)
        )

    desired_variables = model_meta.get("variables") or []
    if desired_variables:
        variable_lookup = {
            name.lower(): idx for idx, name in enumerate(stack_variables)
        }
        missing_vars = [
            var for var in desired_variables if var.lower() not in variable_lookup
        ]
        if missing_vars:
            raise ValueError(
                f"Feature stack missing variables required by model: {missing_vars}"
            )
        var_indices = [variable_lookup[var.lower()] for var in desired_variables]
        aligned_stack = aligned_stack[:, :, :, var_indices]
        aligned_variables = [stack_variables[idx] for idx in var_indices]
        logging.info(
            "Subset feature stack to variables used in training: %s",
            ", ".join(aligned_variables)
        )

    return aligned_stack, aligned_months, aligned_variables, feature_labels


# ============================================================================
# Prediction
# ============================================================================

def predict_map(
    stack: np.ndarray,
    model_path: Path,
    feature_names: list[str],
    stack_variables: list[str],
    require_ndvi: bool,
    mask_variable: Optional[str],
    predefined_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Generate SPEI prediction map from feature stack."""
    months, height, width, vars_count = stack.shape
    pixels = height * width

    logging.info(
        f"Predicting for {height}x{width} grid "
        f"({pixels:,} pixels, {months} months, {vars_count} variables)"
    )

    # Flatten features: (months, H, W, vars) -> (H*W, months*vars)
    features_flat = stack.transpose(1, 2, 0, 3).reshape(pixels, months * vars_count)

    # Create valid mask (pixels with any finite features)
    finite_any = np.any(np.isfinite(features_flat), axis=1)

    # Optional NDVI enforcement when requested
    ndvi_indices = [
        idx for idx, name in enumerate(stack_variables)
        if name.lower() == "ndvi"
    ]
    if require_ndvi:
        if not ndvi_indices:
            raise ValueError(
                "NDVI values are required for prediction but are not present in the feature stack."
            )
        ndvi_subset = stack[:, :, :, ndvi_indices]
        ndvi_flat = ndvi_subset.transpose(1, 2, 0, 3).reshape(pixels, -1)
        ndvi_finite = np.any(np.isfinite(ndvi_flat), axis=1)
        valid_mask = finite_any & ndvi_finite
    else:
        valid_mask = finite_any

    # Additional mask variable
    if predefined_mask is not None:
        valid_mask &= predefined_mask.reshape(-1)
    elif mask_variable:
        lower = mask_variable.lower()
        try:
            idx = stack_variables.index(lower)
        except ValueError as exc:
            raise ValueError(
                f"Mask variable '{mask_variable}' not found in feature stack."
            ) from exc
        mask_data = stack[:, :, :, idx]
        mask_valid = np.any(np.isfinite(mask_data), axis=0)
        valid_mask &= mask_valid

    valid_count = int(valid_mask.sum())
    logging.info(f"Valid pixels: {valid_count:,} / {pixels:,} ({valid_count/pixels*100:.1f}%)")

    if valid_count == 0:
        raise RuntimeError("No valid pixels remaining after NaN filtering")

    # Load model
    logging.info(f"Loading model from {model_path}")
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # Prepare features for prediction
    feature_subset = features_flat[valid_mask]
    if feature_names and len(feature_names) == feature_subset.shape[1]:
        dmatrix = xgb.DMatrix(feature_subset, feature_names=feature_names)
        logging.debug(f"Using {len(feature_names)} feature names")
    else:
        dmatrix = xgb.DMatrix(feature_subset)
        logging.warning("Feature names not provided or count mismatch")

    # Predict
    logging.info("Running prediction...")
    predictions = np.full(pixels, np.nan, dtype=np.float32)
    predictions[valid_mask] = booster.predict(dmatrix)

    # Reshape to 2D
    prediction_map = predictions.reshape(height, width)

    # Compute statistics
    finite_preds = np.isfinite(prediction_map)
    if finite_preds.any():
        pred_vals = prediction_map[finite_preds]
        logging.info(
            f"Prediction stats: "
            f"min={pred_vals.min():.3f}, "
            f"max={pred_vals.max():.3f}, "
            f"mean={pred_vals.mean():.3f}, "
            f"std={pred_vals.std():.3f}"
        )

    return prediction_map


# ============================================================================
# Visualization
# ============================================================================

def save_prediction_plot(
    prediction_map: np.ndarray,
    output_path: Path,
    year: int,
    vmin: float,
    vmax: float
) -> None:
    """Save prediction visualization."""
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(prediction_map, cmap=SPEI_CMAP, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=f"Predicted SPEI ({year})", shrink=0.35)
    plt.title(f"Predicted SPEI - {year}", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved prediction plot to {output_path}")


def save_observed_plot(
    observed_map: np.ndarray,
    output_path: Path,
    year: int,
    vmin: float,
    vmax: float
) -> None:
    """Save observed SPEI visualization."""
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(observed_map, cmap=SPEI_CMAP, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=f"Observed SPEI ({year})", shrink=0.35)
    plt.title(f"Observed SPEI - {year}", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved observed plot to {output_path}")


def save_comparison_plot(
    predicted: np.ndarray,
    observed: np.ndarray,
    output_path: Path,
    year: int,
    vmin: float,
    vmax: float
) -> None:
    """Save side-by-side comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Predicted
    im1 = axes[0].imshow(predicted, cmap=SPEI_CMAP, vmin=vmin, vmax=vmax)
    axes[0].set_title("Predicted SPEI", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], shrink=0.7)

    # Observed
    im2 = axes[1].imshow(observed, cmap=SPEI_CMAP, vmin=vmin, vmax=vmax)
    axes[1].set_title("Observed SPEI", fontsize=12, fontweight='bold')
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], shrink=0.7)

    fig.suptitle(f"SPEI Comparison - {year}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved comparison plot to {output_path}")


# ============================================================================
# Target Alignment
# ============================================================================

def align_target_to_prediction(
    target: np.ndarray,
    target_transform: Affine,
    target_crs: CRS,
    prediction_shape: tuple[int, int],
    prediction_transform: Affine,
    prediction_crs: CRS
) -> np.ndarray:
    """Reproject target SPEI to match prediction grid."""
    logging.info("Aligning observed SPEI to prediction grid...")

    destination = np.full(prediction_shape, np.nan, dtype=np.float32)

    reproject(
        source=target,
        destination=destination,
        src_transform=target_transform,
        src_crs=target_crs,
        dst_transform=prediction_transform,
        dst_crs=prediction_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    return destination


# ============================================================================
# Main Function
# ============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(levelname)s] %(message)s"
    )

    # Resolve paths
    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Default feature path
    if args.feature_path:
        feature_path = args.feature_path.expanduser().resolve()
    else:
        feature_path = DEFAULT_FEATURE_DIR / f"X_{args.year}.npz"

    # Default output directory
    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    stack, stack_meta = load_feature_stack(feature_path)
    months = stack_meta.get("months", [])
    variables = stack_meta.get("variables", [])

    mask_array = None
    if args.mask_variable and variables:
        lower_vars = [str(v).lower() for v in variables]
        mask_key = args.mask_variable.lower()
        if mask_key in lower_vars:
            idx = lower_vars.index(mask_key)
            mask_data = stack[:, :, :, idx]
            mask_array = np.any(np.isfinite(mask_data), axis=0)
        else:
            logging.warning(
                "Mask variable '%s' not found in raw stack; mask will be applied after subsetting if possible.",
                args.mask_variable,
            )

    # Align stack with model expectations when metadata is available
    model_meta = load_model_metadata(model_path)
    stack, months, variables, meta_feature_names = align_stack_to_model(
        stack,
        months,
        variables,
        model_meta,
    )

    # Build feature names
    if meta_feature_names:
        feature_names = meta_feature_names
        logging.info(f"Using {len(feature_names)} feature labels from model metadata")
    elif months and variables:
        feature_names = build_feature_names(months, variables)
        logging.info(f"Built {len(feature_names)} feature names from stack metadata")
    else:
        feature_names = []
        logging.warning("Could not build feature names (missing months/variables)")

    # Extract geospatial metadata
    prediction_transform = extract_transform(stack_meta)
    prediction_crs = extract_crs(stack_meta)

    # Generate prediction
    prediction_map = predict_map(
        stack,
        model_path,
        feature_names,
        variables,
        args.require_ndvi,
        args.mask_variable,
        mask_array,
    )

    # Save prediction array
    pred_array_path = output_dir / f"predicted_spei_{args.year}.npy"
    np.save(pred_array_path, prediction_map.astype(np.float32, copy=False))
    logging.info(f"Saved prediction array to {pred_array_path}")

    # Save prediction plot
    pred_plot_path = output_dir / f"predicted_spei_{args.year}_map.png"
    save_prediction_plot(prediction_map, pred_plot_path, args.year, args.vmin, args.vmax)

    # Handle observed SPEI (optional)
    if args.target_path:
        target_path = args.target_path.expanduser().resolve()
    else:
        target_path = DEFAULT_TARGET_DIR / f"y_{args.year}.npy"

    target_meta_path = target_path.parent / f"{target_path.stem}_meta.json"

    if target_path.exists() and target_meta_path.exists():
        try:
            # Load target
            target = load_target(target_path)
            target_meta = load_json(target_meta_path)

            target_transform = extract_transform(target_meta)
            target_crs = extract_crs(target_meta)

            if (prediction_transform and prediction_crs and
                target_transform and target_crs):

                # Align target to prediction grid
                aligned_target = align_target_to_prediction(
                    target,
                    target_transform,
                    target_crs,
                    prediction_map.shape,
                    prediction_transform,
                    prediction_crs
                )

                # Save aligned target
                aligned_array_path = output_dir / f"actual_spei_{args.year}_aligned.npy"
                np.save(aligned_array_path, aligned_target)
                logging.info(f"Saved aligned target to {aligned_array_path}")

                # Save observed plot
                obs_plot_path = output_dir / f"actual_spei_{args.year}_map.png"
                save_observed_plot(aligned_target, obs_plot_path, args.year, args.vmin, args.vmax)

                # Save comparison plot
                comp_plot_path = output_dir / f"comparison_spei_{args.year}.png"
                save_comparison_plot(
                    prediction_map,
                    aligned_target,
                    comp_plot_path,
                    args.year,
                    args.vmin,
                    args.vmax
                )

                # Compute comparison metrics
                both_finite = np.isfinite(prediction_map) & np.isfinite(aligned_target)
                if both_finite.any():
                    pred_vals = prediction_map[both_finite]
                    obs_vals = aligned_target[both_finite]
                    residuals = pred_vals - obs_vals
                    mae = float(np.mean(np.abs(residuals)))
                    rmse = float(np.sqrt(np.mean(residuals ** 2)))
                    logging.info(f"Comparison metrics: MAE={mae:.3f}, RMSE={rmse:.3f}")
                else:
                    logging.warning("No overlapping valid pixels for comparison")

            else:
                logging.warning(
                    "Missing geospatial metadata; skipping target alignment"
                )

        except Exception as e:
            logging.error(f"Failed to process observed SPEI: {e}")
    else:
        logging.info("No observed SPEI data found; skipping comparison")

    logging.info("Prediction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
