#!/usr/bin/env python3
"""Flatten UK-wide feature tensors, align SPEI targets, and train XGBoost models.

This script combines the outputs from step_1 (features) and step_2 (targets),
flattens them into 2D arrays suitable for machine learning, and trains XGBoost
regression models to predict drought conditions (SPEI values).

Usage:
    python step_3_flatten_inputs_uk.py --years 2020 2021 2022 --months 5 6 7
"""

import argparse
import calendar
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = BASE_DIR / "prepared_inputs_uk"
DEFAULT_TARGET_DIR = BASE_DIR / "prepared_targets_uk"
DEFAULT_OUTPUT_DIR = DEFAULT_FEATURE_DIR
DEFAULT_MONTHS = (5, 6, 7)  # May, June, July (growing season)
DEFAULT_VARIABLES = (
    "lst",
    "soil_evap",
    "precip",
    "lst_anom",
    "soil_evap_anom",
    "precip_anom",
)

# Output filenames
MODEL_FILENAME = "trained_xgb_model.json"
METRICS_FILENAME = "trained_xgb_metrics.json"
PLOT_FEATURE_IMPORTANCE = "feature_importance_gain.png"
PLOT_PARITY = "predicted_vs_observed.png"
PLOT_RESIDUALS = "residuals_hist_train_test.png"
PLOT_RESID_VS_PRED = "residuals_vs_predicted.png"
PLOT_PER_YEAR = "test_metrics_by_year.png"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    years: list[int]
    variables: list[str]
    feature_dir: Path
    target_dir: Path
    output_dir: Path
    require_ndvi: bool


def has_ndvi(variables: Sequence[str]) -> bool:
    """Return True if NDVI is part of the variable list."""
    return any(var.lower() == "ndvi" for var in variables)


def drop_ndvi(variables: Sequence[str]) -> list[str]:
    """Remove NDVI (& derived anomaly) from the provided variable sequence."""
    return [
        var
        for var in variables
        if var.lower() not in {
            "ndvi",
            "ndvi_anom",
            "ndvi_zscore",
            "ndvi_deficit",
            "ndvi_integral",
        }
    ]


def inspect_available_variables(path: Path) -> list[str]:
    """Inspect the variable names stored in a feature archive."""
    if not path.exists():
        raise FileNotFoundError(f"Feature archive not found: {path}")
    with np.load(path) as data:
        if "variables" not in data.files:
            raise ValueError(f"Feature archive {path} does not store 'variables' metadata")
        variables = [str(v).lower() for v in data["variables"]]
    return variables


def enforce_optional_ndvi(cfg: DatasetConfig) -> None:
    """Drop NDVI from the config when it is unavailable and optional."""
    if not has_ndvi(cfg.variables):
        return

    missing_years: list[int] = []
    for year in cfg.years:
        archive_path = cfg.feature_dir / f"X_{year}.npz"
        available = inspect_available_variables(archive_path)
        if "ndvi" not in available:
            missing_years.append(year)

    if not missing_years:
        return

    if cfg.require_ndvi:
        years_text = ", ".join(str(year) for year in missing_years)
        raise ValueError(
            "NDVI features were requested but are missing for the "
            f"following years: {years_text}"
        )

    cfg.variables = drop_ndvi(cfg.variables)
    if not cfg.variables:
        raise ValueError(
            "NDVI data is unavailable and no other variables were provided. "
            "Specify additional variables via --variables or supply NDVI inputs."
        )

    years_text = ", ".join(str(year) for year in missing_years)
    logging.warning(
        "NDVI not found in feature archives for years %s; continuing without NDVI features.",
        years_text,
    )


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022],
        help="Calendar years to include."
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=list(DEFAULT_MONTHS),
        help="Months to include (accepts numbers or names, e.g. 5 6 7 or may jun jul)."
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        type=str,
        default=list(DEFAULT_VARIABLES),
        help=(
            "Variables to include (raw or *_anom). "
            "Default includes both raw and anomaly features for ndvi/lst/soil_evap/precip."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
        help="Directory containing X_YEAR.npz feature archives."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory containing y_YEAR.npy target arrays."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory where experiment artifacts will be written."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for test split (default: 0.2)."
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for splitting and model training (default: 42)."
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds for diagnostics (default: 3)."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generation of diagnostic plots."
    )
    parser.add_argument(
        "--ndvi-weight",
        type=float,
        default=1.0,
        help="Multiplier for NDVI features to emphasize contribution (default: 1.0)."
    )
    parser.add_argument(
        "--require-ndvi",
        action="store_true",
        help="Fail if NDVI features are missing instead of skipping them."
    )
    parser.add_argument(
        "--month-group",
        action="append",
        metavar="LABEL:MONTHS",
        help=(
            "Optional experiment specification, e.g. mayjul:5,6,7 or aprjun:apr,jun,jul. "
            "Repeat to run multiple experiments."
        )
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity."
    )
    parser.add_argument(
        "--drought-threshold",
        type=float,
        default=-1.0,
        help=(
            "SPEI threshold defining drought pixels for weighting (default: -1.0)."
        )
    )
    parser.add_argument(
        "--drought-weight",
        type=float,
        default=1.0,
        help="Sample weight multiplier applied to drought pixels (default: 1.0 = no weighting)."
    )
    return parser.parse_args(argv)


# ============================================================================
# Month Parsing Utilities
# ============================================================================

# Build month lookup dictionary
MONTH_LOOKUP = {name.lower(): idx for idx, name in enumerate(calendar.month_abbr) if idx}
MONTH_LOOKUP.update({name.lower(): idx for idx, name in enumerate(calendar.month_name) if idx})


def parse_month(value: int | str) -> int:
    """Parse month from integer or string."""
    if isinstance(value, int):
        month = value
    else:
        text = str(value).strip().lower()
        if text.isdigit():
            month = int(text)
        else:
            if text not in MONTH_LOOKUP:
                raise ValueError(f"Invalid month label: {value}")
            month = MONTH_LOOKUP[text]
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month: {value}")
    return month


def normalize_months(months: Sequence[int | str]) -> list[int]:
    """Normalize and deduplicate months."""
    seen = set()
    ordered = []
    for value in months:
        month = parse_month(value)
        if month not in seen:
            ordered.append(month)
            seen.add(month)
    return ordered


def normalize_variables(variables: Sequence[str]) -> list[str]:
    """Normalize and deduplicate variables."""
    seen = set()
    ordered = []
    for var in variables:
        key = var.lower()
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


# ============================================================================
# Path and Configuration Resolution
# ============================================================================

def resolve_paths(args: argparse.Namespace) -> tuple[DatasetConfig, list[int]]:
    """Resolve and validate paths."""
    months = normalize_months(args.months)
    variables = normalize_variables(args.variables)
    years = sorted({int(year) for year in args.years})

    feature_dir = args.feature_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for directory in (feature_dir, target_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = DatasetConfig(
        years=years,
        variables=variables,
        feature_dir=feature_dir,
        target_dir=target_dir,
        output_dir=output_dir,
        require_ndvi=args.require_ndvi,
    )
    return cfg, months


def sanitize_label(label: str) -> str:
    """Sanitize label for use as directory name."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.lower())
    cleaned = cleaned.strip("_")
    return cleaned or "experiment"


def default_label_for_months(months: Sequence[int]) -> str:
    """Generate default label from month list."""
    parts = [calendar.month_abbr[m].lower() for m in months]
    return "_".join(parts)


def parse_month_groups(
    raw_groups: Optional[Sequence[str]],
    default_months: list[int]
) -> list[tuple[str, list[int]]]:
    """Parse month group specifications."""
    if not raw_groups:
        label = default_label_for_months(default_months)
        return [(label, default_months)]

    groups = []
    used_labels = set()

    for spec in raw_groups:
        if ":" not in spec:
            raise ValueError(f"Month group must use LABEL:MONTHS format, got: {spec}")
        raw_label, month_str = spec.split(":", 1)
        months = normalize_months(part.strip() for part in month_str.replace(",", " ").split())
        base_label = sanitize_label(raw_label)
        label = base_label
        suffix = 1
        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        used_labels.add(label)
        groups.append((label, months))

    return groups


def experiment_directory(base: Path, label: str) -> Path:
    """Return the folder used to store experiment outputs."""
    return base if base.name == label else base / label


# ============================================================================
# Feature and Target Loading
# ============================================================================

def build_feature_labels(months: Sequence[int], variables: Sequence[str]) -> list[str]:
    """Build feature labels for model interpretation."""
    labels = []
    for month in months:
        month_name = calendar.month_abbr[month]
        for var in variables:
            labels.append(f"{month_name}_{var.upper()}")
    return labels


def load_feature_archive(path: Path) -> dict[str, np.ndarray]:
    """Load feature archive from NPZ file."""
    if not path.exists():
        raise FileNotFoundError(f"Feature archive not found: {path}")
    with np.load(path) as data:
        stack = data["X"]
        months = data["months"].astype(np.int16)
        variables = np.array([str(v) for v in data["variables"]], dtype=object)
    return {"stack": stack, "months": months, "variables": variables}


def subset_feature_stack(
    archive: dict[str, np.ndarray],
    months: Sequence[int],
    variables: Sequence[str]
) -> np.ndarray:
    """Extract subset of features matching requested months and variables."""
    available_months = archive["months"]
    available_variables = archive["variables"]

    # Find month indices
    month_indices = []
    for month in months:
        matches = np.where(available_months == month)[0]
        if matches.size == 0:
            raise ValueError(
                f"Month {month} not found in archive "
                f"(available: {available_months.tolist()})"
            )
        month_indices.append(int(matches[0]))

    # Find variable indices
    variable_indices = []
    for var in variables:
        matches = np.where(available_variables == var)[0]
        if matches.size == 0:
            raise ValueError(
                f"Variable '{var}' not found in archive "
                f"(available: {available_variables.tolist()})"
            )
        variable_indices.append(int(matches[0]))

    # Extract subset
    stack = archive["stack"][month_indices]
    stack = stack[:, :, :, variable_indices]
    return stack.astype(np.float32, copy=False)


def load_targets(path: Path) -> np.ndarray:
    """Load target array from NPY file."""
    if not path.exists():
        raise FileNotFoundError(f"Target array not found: {path}")
    return np.load(path).astype(np.float32, copy=False)


def flatten_year_data(
    year: int,
    cfg: DatasetConfig,
    months: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Flatten features and targets for a single year."""
    feature_path = cfg.feature_dir / f"X_{year}.npz"
    target_path = cfg.target_dir / f"y_{year}.npy"

    # Load data
    archive = load_feature_archive(feature_path)
    stack = subset_feature_stack(archive, months, cfg.variables)
    target = load_targets(target_path)

    # Check spatial alignment
    if stack.shape[1:3] != target.shape:
        raise ValueError(
            f"Spatial shape mismatch for {year}: "
            f"features {stack.shape[1:3]} vs target {target.shape}"
        )

    # Flatten: (n_months, H, W, n_vars) -> (H*W, n_months*n_vars)
    features_flat = stack.transpose(1, 2, 0, 3).reshape(-1, stack.shape[0] * stack.shape[3])
    target_flat = target.reshape(-1)

    # Filter valid samples
    target_finite = np.isfinite(target_flat)
    feature_counts = np.sum(np.isfinite(features_flat), axis=1)
    valid_mask = target_finite & (feature_counts > 0)

    features_valid = features_flat[valid_mask]
    target_valid = target_flat[valid_mask]

    logging.debug(
        f"Year {year} | samples={features_flat.shape[0]} | "
        f"kept={features_valid.shape[0]} | "
        f"dropped_target={int(np.count_nonzero(~target_finite))} | "
        f"dropped_all_nan={int(np.count_nonzero(target_finite & (feature_counts == 0)))}"
    )

    return features_valid, target_valid, features_valid.shape[0]


def compute_sample_weights(
    targets: np.ndarray,
    threshold: float,
    drought_weight: float
) -> tuple[np.ndarray, int]:
    """Return per-sample weights emphasizing drought pixels."""
    weights = np.ones(targets.shape[0], dtype=np.float32)
    drought_mask = targets <= threshold
    drought_count = int(np.count_nonzero(drought_mask))
    if drought_weight != 1.0:
        weights[drought_mask] = drought_weight
    return weights, drought_count


# ============================================================================
# Metrics and Statistics
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    residuals = y_pred - y_true
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs(residuals / denom)) * 100.0)
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape_percent": mape,
        "explained_variance": evs
    }


def summarize_residuals(residuals: np.ndarray) -> dict[str, float]:
    """Compute residual statistics."""
    mean_val = float(np.mean(residuals))
    std_val = float(np.std(residuals))
    skew = (
        float(np.mean(((residuals - mean_val) / std_val) ** 3))
        if std_val > 1e-9 else float("nan")
    )
    return {
        "mean": mean_val,
        "std": std_val,
        "median": float(np.median(residuals)),
        "q25": float(np.percentile(residuals, 25)),
        "q75": float(np.percentile(residuals, 75)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "skewness": skew,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_feature_importance(
    model: xgb.XGBRegressor,
    feature_labels: Sequence[str],
    out_path: Path
) -> None:
    """Plot feature importance by gain."""
    fig, ax = plt.subplots(figsize=(10, 5))
    booster = model.get_booster()
    booster.feature_names = list(feature_labels)
    xgb.plot_importance(
        booster,
        ax=ax,
        importance_type="gain",
        height=0.6,
        show_values=False
    )
    ax.set_title("XGBoost Feature Importance (Gain)")
    ax.set_xlabel("Average Gain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved {out_path.name}")


def plot_parity(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    out_path: Path
) -> None:
    """Plot predicted vs observed (parity plot)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    lims = [
        float(min(y_train.min(), y_test.min())),
        float(max(y_train.max(), y_test.max())),
    ]

    # Train (hexbin for density)
    hb = axes[0].hexbin(
        y_train,
        y_pred_train,
        gridsize=60,
        cmap="viridis",
        bins="log",
        mincnt=1,
    )
    axes[0].plot(lims, lims, color="white", linestyle="--", linewidth=1)
    axes[0].set_title("Train density")
    axes[0].set_xlabel("Observed SPEI")
    axes[0].set_ylabel("Predicted SPEI")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].grid(True, linestyle="--", alpha=0.2)
    cbar = fig.colorbar(hb, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Point density (log)")

    # Test (scatter)
    axes[1].scatter(
        y_test,
        y_pred_test,
        s=18,
        alpha=0.7,
        color="tab:orange",
        edgecolors="none"
    )
    axes[1].plot(lims, lims, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Test scatter")
    axes[1].set_xlabel("Observed SPEI")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Predicted vs Observed SPEI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved {out_path.name}")


def plot_residual_hist(
    res_train: np.ndarray,
    res_test: np.ndarray,
    out_path: Path
) -> None:
    """Plot residual distribution histogram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    combined = np.concatenate([res_train, res_test])
    bins = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 40)
    ax.hist(res_train, bins=bins, alpha=0.5, label="Train", color="tab:blue", density=True)
    ax.hist(res_test, bins=bins, alpha=0.6, label="Test", color="tab:orange", density=True)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error (Pred - True)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved {out_path.name}")


def plot_residuals_vs_pred(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    out_path: Path
) -> None:
    """Plot residuals vs predicted values."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, s=12, alpha=0.4, color="tab:red", edgecolors="none")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted SPEI (Test)")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals vs Predicted (Test Set)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved {out_path.name}")


def plot_per_year_metrics(
    per_year: dict[int, dict[str, float]],
    out_path: Path
) -> None:
    """Plot metrics by year."""
    years = sorted(per_year.keys())
    metrics = ("r2", "rmse")
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 4), sharex=True)

    for ax, metric in zip(axes, metrics):
        values = [per_year[year][metric] for year in years]
        ax.bar([str(y) for y in years], values, color="tab:green", alpha=0.7)
        ax.set_title(metric.upper())
        ax.set_xlabel("Year")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(bottom=min(0, min(values)) * 1.05 if metric == "r2" else 0)
        for idx, val in enumerate(values):
            ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Test Metrics by Year")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved {out_path.name}")


# ============================================================================
# Cross-Validation
# ============================================================================

def run_cross_validation(
    features: np.ndarray,
    targets: np.ndarray,
    years: np.ndarray,
    params: dict,
    folds: int,
    feature_weights: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None
) -> list[dict[str, float]]:
    """Run k-fold cross-validation."""
    if folds < 2:
        return []

    cv = KFold(n_splits=folds, shuffle=True, random_state=params.get("random_state", 42))
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(features, years), start=1):
        model = xgb.XGBRegressor(**params)
        sw_train = sample_weights[train_idx] if sample_weights is not None else None
        model.fit(
            features[train_idx],
            targets[train_idx],
            sample_weight=sw_train,
            feature_weights=feature_weights,
        )
        predictions = model.predict(features[val_idx])
        metrics = compute_metrics(targets[val_idx], predictions)
        fold_metrics.append(metrics)
        logging.info(
            f"[CV] Fold {fold_idx} | R²={metrics['r2']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f}"
        )

    return fold_metrics


def summarize_cv(fold_metrics: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Summarize cross-validation results."""
    summary = {}
    if not fold_metrics:
        return summary

    metric_keys = fold_metrics[0].keys()
    for key in metric_keys:
        values = np.array([fold[key] for fold in fold_metrics], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0))
        }
    return summary


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_and_evaluate(
    cfg: DatasetConfig,
    months: Sequence[int],
    label: str,
    output_dir: Path,
    test_size: float,
    random_seed: int,
    cv_folds: int,
    skip_plots: bool,
    ndvi_weight: float,
    drought_threshold: float,
    drought_weight: float
) -> dict[str, float]:
    """Train XGBoost model and evaluate performance."""
    feature_labels = build_feature_labels(months, cfg.variables)
    ndvi_mask = np.array(
        ["NDVI" in lbl.upper() for lbl in feature_labels],
        dtype=bool
    )

    # Load all years
    all_features = []
    all_targets = []
    all_year_labels = []
    per_year_counts = {}

    for year in cfg.years:
        features_year, targets_year, count = flatten_year_data(year, cfg, months)
        all_features.append(features_year)
        all_targets.append(targets_year)
        all_year_labels.append(np.full(targets_year.shape[0], year, dtype=np.int16))
        per_year_counts[year] = count

    X_all = np.concatenate(all_features, axis=0)
    y_all = np.concatenate(all_targets, axis=0)
    years_all = np.concatenate(all_year_labels, axis=0)

    sample_weights, drought_count = compute_sample_weights(
        y_all,
        drought_threshold,
        drought_weight,
    )
    logging.info(
        f"[{label}] Drought samples (y <= {drought_threshold:.2f}): {drought_count} "
        f"({drought_count / y_all.size * 100:.1f}%); weight={drought_weight:.2f}"
    )

    # Apply feature weights
    column_weights = np.ones(X_all.shape[1], dtype=np.float32)
    if ndvi_weight != 1.0 and ndvi_mask.any():
        logging.info(
            f"[{label}] Setting NDVI column weight to {ndvi_weight:.3f} "
            f"for {int(ndvi_mask.sum())} columns."
        )
        column_weights[ndvi_mask] = ndvi_weight

    logging.info(
        f"[{label}] Total samples: {X_all.shape[0]} | "
        f"features per sample: {X_all.shape[1]}"
    )

    # Train/test split
    stratify = years_all if len(np.unique(years_all)) > 1 else None
    split = train_test_split(
        X_all,
        y_all,
        years_all,
        sample_weights,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )
    X_train, X_test, y_train, y_test, years_train, years_test, w_train, w_test = split

    logging.info(
        f"[{label}] Train samples: {X_train.shape[0]} | "
        f"Test samples: {X_test.shape[0]}"
    )

    # XGBoost parameters
    model_params = dict(
        n_estimators=140,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_seed,
        n_jobs=8,
    )

    # Train model
    model = xgb.XGBRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        feature_weights=column_weights,
    )

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Compute metrics
    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)
    logging.info(
        f"[{label}] Train R²={train_metrics['r2']:.4f} | "
        f"RMSE={train_metrics['rmse']:.4f}"
    )
    logging.info(
        f"[{label}] Test  R²={test_metrics['r2']:.4f} | "
        f"RMSE={test_metrics['rmse']:.4f}"
    )

    # Per-year metrics
    per_year_metrics = {}
    for year in cfg.years:
        mask = years_test == year
        if np.any(mask):
            per_year_metrics[year] = compute_metrics(y_test[mask], test_pred[mask])
            logging.info(
                f"[{label}] Year {year} | samples={int(mask.sum())} | "
                f"R²={per_year_metrics[year]['r2']:.4f} | "
                f"RMSE={per_year_metrics[year]['rmse']:.4f}"
            )

    # Residuals
    train_residuals = train_pred - y_train
    test_residuals = test_pred - y_test
    residual_summary = {
        "train": summarize_residuals(train_residuals),
        "test": summarize_residuals(test_residuals)
    }

    # Cross-validation
    logging.info(f"[{label}] Running {cv_folds}-fold cross-validation...")
    fold_metrics = run_cross_validation(
        X_all,
        y_all,
        years_all,
        model_params,
        cv_folds,
        column_weights,
        sample_weights,
    )
    cv_summary = summarize_cv(fold_metrics)

    # Save metrics
    metrics_report = {
        "label": label,
        "months": list(months),
        "variables": cfg.variables,
        "feature_labels": feature_labels,
        "sample_counts": {
            "total": int(X_all.shape[0]),
            "train": int(X_train.shape[0]),
            "test": int(X_test.shape[0]),
            "per_year": {str(year): int(per_year_counts.get(year, 0)) for year in cfg.years},
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_metrics_per_year": {str(year): metrics for year, metrics in per_year_metrics.items()},
        "residual_summary": residual_summary,
        "cross_validation": {
            "folds": len(fold_metrics),
            "per_fold": fold_metrics,
            "summary": cv_summary,
        },
        "model_params": model_params,
        "random_seed": random_seed,
        "ndvi_weight": ndvi_weight,
        "sample_weighting": {
            "drought_threshold": drought_threshold,
            "drought_weight": drought_weight,
            "drought_samples": drought_count,
        },
    }

    metrics_path = output_dir / METRICS_FILENAME
    with metrics_path.open("w", encoding="utf8") as fh:
        json.dump(metrics_report, fh, indent=2)
    logging.info(f"[{label}] Saved metrics report to {metrics_path}")

    # Generate plots
    if not skip_plots:
        plot_feature_importance(model, feature_labels, output_dir / PLOT_FEATURE_IMPORTANCE)
        plot_parity(y_train, train_pred, y_test, test_pred, output_dir / PLOT_PARITY)
        plot_residual_hist(train_residuals, test_residuals, output_dir / PLOT_RESIDUALS)
        plot_residuals_vs_pred(test_pred, test_residuals, output_dir / PLOT_RESID_VS_PRED)
        if per_year_metrics:
            plot_per_year_metrics(per_year_metrics, output_dir / PLOT_PER_YEAR)

    # Train final model on all data
    logging.info(f"[{label}] Training final model on all samples...")
    final_model = xgb.XGBRegressor(**model_params)
    final_model.fit(
        X_all,
        y_all,
        sample_weight=sample_weights,
        feature_weights=column_weights,
    )
    booster = final_model.get_booster()
    booster.feature_names = list(feature_labels)
    model_path = output_dir / MODEL_FILENAME
    final_model.save_model(model_path)
    logging.info(f"[{label}] Saved model to {model_path}")

    return {
        "label": label,
        "test_r2": test_metrics.get("r2"),
        "test_rmse": test_metrics.get("rmse"),
        "samples": int(X_all.shape[0])
    }


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

    cfg, default_months = resolve_paths(args)
    enforce_optional_ndvi(cfg)
    experiments = parse_month_groups(args.month_group, default_months)

    summaries = []
    for label, months in experiments:
        experiment_dir = experiment_directory(cfg.output_dir, label)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"==== Running experiment '{label}' with months {months} ====")
        summary = train_and_evaluate(
            cfg,
            months,
            label,
            experiment_dir,
            args.test_size,
            args.random_seed,
            args.cv_folds,
            args.skip_plots,
            args.ndvi_weight,
            args.drought_threshold,
            args.drought_weight,
        )
        summaries.append(summary)

    # Print summary
    if len(summaries) > 1:
        logging.info("==== Experiment summary ====")
        for entry in summaries:
            logging.info(
                f"[{entry['label']}] samples={entry['samples']} | "
                f"test R²={entry['test_r2']:.4f} | "
                f"test RMSE={entry['test_rmse']:.4f}"
            )

    logging.info("All experiments completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
