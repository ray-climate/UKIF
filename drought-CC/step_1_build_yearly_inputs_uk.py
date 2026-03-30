#!/usr/bin/env python3
"""Build yearly UK-wide tensors from monthly GeoTIFF inputs.

This script aligns Sentinel-2 NDVI, VIIRS LST, and ERA5-Land soil evaporation
and precipitation GeoTIFFs onto a common 1 km grid, clips to the UK bounding box,
and saves compressed NumPy archives for downstream drought modelling.

Usage:
    python step_1_build_yearly_inputs_uk.py --years 2020 2021 2022
"""

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import calendar
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds, transform as window_transform

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "prepared_inputs_uk"
CLIMATOLOGY_FILENAME = "X_climatology.npz"
DEFAULT_ROI = (-10.5, 49.5, 2.5, 60.8)  # UK bounding box in WGS84
SENTINEL = np.float32(-9999.0)
VARIABLE_CHOICES = ("ndvi", "lst", "soil_evap", "precip")
DEFAULT_VARIABLE_SELECTION = ("lst", "soil_evap", "precip")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VariableConfig:
    """Configuration for a single variable (NDVI, LST, etc.)."""
    key: str
    directory: Path
    cleaner: Callable[[np.ndarray], np.ndarray]
    resampling: Resampling = Resampling.bilinear
    apply_nodata: bool = True

    def find_file(self, year: int, month: int) -> Path:
        """Find GeoTIFF file for given year and month."""
        pattern = f"*{year}_{month:02d}*.tif"
        matches = sorted(self.directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No GeoTIFF for {self.key.upper()} in {self.directory} "
                f"matching {pattern}"
            )
        if len(matches) > 1:
            logging.debug(
                f"Multiple matches for {self.key} {year}-{month:02d}, "
                f"using {matches[0].name}"
            )
        return matches[0]


# ============================================================================
# Data Cleaning Functions
# ============================================================================

def clean_ndvi(arr: np.ndarray) -> np.ndarray:
    """Clean NDVI data: convert to float32, mask invalid values."""
    arr = arr.astype(np.float32, copy=False)
    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < -1.0) | (arr > 1.0)] = np.nan
    return arr


def clean_lst(arr: np.ndarray) -> np.ndarray:
    """Clean LST data: convert to float32, mask invalid fill values."""
    arr = arr.astype(np.float32, copy=False)
    arr[~np.isfinite(arr)] = np.nan
    arr[arr < 200] = np.nan  # mask invalid fill values
    return arr


def clean_precip(arr: np.ndarray) -> np.ndarray:
    """Clean precipitation data: convert m to mm, ensure non-negative."""
    arr = arr.astype(np.float32, copy=False)
    arr[~np.isfinite(arr)] = np.nan
    arr *= 1000.0  # convert metres to mm
    arr[arr < 0] = 0.0
    return arr


def clean_soil_evap(arr: np.ndarray) -> np.ndarray:
    """Clean soil evaporation: convert m to mm, flip sign (ERA5 is negative)."""
    arr = arr.astype(np.float32, copy=False)
    arr[~np.isfinite(arr)] = np.nan
    arr *= -1000.0  # ERA5 evap is negative (downward flux); convert to +mm
    arr[arr < 0] = 0.0
    return arr


def build_variable_configs(base_dir: Path) -> dict[str, VariableConfig]:
    """Build configuration for all variables."""
    return {
        "ndvi": VariableConfig(
            key="ndvi",
            directory=base_dir / "data_s2_ndvi_uk",
            cleaner=clean_ndvi,
        ),
        "lst": VariableConfig(
            key="lst",
            directory=base_dir / "data_viirs_lst_uk",
            cleaner=clean_lst,
        ),
        "soil_evap": VariableConfig(
            key="soil_evap",
            directory=base_dir / "data_era5_bare_soil_evap_uk",
            cleaner=clean_soil_evap,
        ),
        "precip": VariableConfig(
            key="precip",
            directory=base_dir / "data_era5_total_precip_uk",
            cleaner=clean_precip,
        ),
    }


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022],
        help="List of years to process (default: 2020 2021 2022)."
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=list(range(1, 13)),
        help="Months to include (accepts numbers or names)."
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        choices=VARIABLE_CHOICES,
        default=list(DEFAULT_VARIABLE_SELECTION),
        help="Variables to stack (default: lst soil_evap precip)."
    )
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("min_lon", "min_lat", "max_lon", "max_lat"),
        help="Bounding box in WGS84 for cropping (default: UK extent)."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional GeoTIFF to use as the reference grid."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store yearly tensors."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Root folder containing data_* directories."
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip months with missing inputs instead of failing."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing yearly outputs."
    )
    parser.add_argument(
        "--climatology-path",
        type=Path,
        help=(
            "Optional path to read/write the monthly climatology used for anomalies. "
            "If provided and exists, the stored baseline is reused. If not provided, "
            "a file named X_climatology.npz is created inside --output-dir."
        )
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)."
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


def validate_months(months: Sequence[int | str]) -> list[int]:
    """Validate and sort months."""
    return sorted({parse_month(month) for month in months})


def validate_years(years: Sequence[int]) -> list[int]:
    """Validate and sort years."""
    return sorted({int(y) for y in years})


# ============================================================================
# Geospatial Utilities
# ============================================================================

def is_wgs84(crs: CRS) -> bool:
    """Check if CRS is WGS84."""
    try:
        wkt = crs.to_wkt().upper()
        return "WGS 84" in wkt or "WGS84" in wkt or "4326" in wkt
    except Exception:
        return False


def bbox_to_reference(
    bbox_wgs84: tuple[float, float, float, float],
    ref_crs: CRS
) -> tuple[float, float, float, float]:
    """Transform WGS84 bbox to reference CRS if needed."""
    if is_wgs84(ref_crs):
        return bbox_wgs84
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", ref_crs, min_lon, min_lat, max_lon, max_lat, densify_pts=21
    )
    return left, bottom, right, top


def compute_roi_window(
    ref_transform: Affine,
    ref_crs: CRS,
    ref_shape: tuple[int, int],
    bbox_wgs84: Optional[tuple[float, float, float, float]]
) -> tuple[Window, Affine, tuple[int, int]]:
    """Compute ROI window from bounding box."""
    height, width = ref_shape
    if bbox_wgs84 is None:
        window = Window(col_off=0, row_off=0, width=width, height=height)
        return window, ref_transform, (height, width)

    left, bottom, right, top = bbox_to_reference(bbox_wgs84, ref_crs)
    window = from_bounds(left, bottom, right, top, transform=ref_transform)
    window = window.round_offsets().round_shape()
    full = Window(col_off=0, row_off=0, width=width, height=height)
    window = window.intersection(full)

    if window.width <= 0 or window.height <= 0:
        raise ValueError("ROI does not intersect the reference grid.")

    roi_transform = window_transform(window, ref_transform)
    roi_shape = (int(window.height), int(window.width))
    return window, roi_transform, roi_shape


# ============================================================================
# Reference Grid Selection
# ============================================================================

def choose_reference_file(
    explicit: Optional[Path],
    variable_configs: dict[str, VariableConfig],
    selected_keys: Sequence[str]
) -> Path:
    """Choose a reference grid GeoTIFF."""
    if explicit:
        ref = explicit.expanduser().resolve()
        if not ref.exists():
            raise FileNotFoundError(f"Reference file not found: {ref}")
        return ref

    # Try selected variables first, then others
    priority = list(selected_keys) + [
        key for key in VARIABLE_CHOICES if key not in selected_keys
    ]
    for key in priority:
        cfg = variable_configs[key]
        candidates = sorted(cfg.directory.glob("*.tif"))
        if candidates:
            logging.info(
                f"Using {candidates[0].name} as reference grid (variable: {key.upper()})"
            )
            return candidates[0]

    raise FileNotFoundError(
        "Could not locate a reference grid. Supply one with --reference."
    )


def read_reference(
    path: Path
) -> tuple[CRS, Affine, tuple[int, int], tuple[float, float, float, float]]:
    """Read reference grid metadata."""
    with rasterio.open(path) as src:
        if not src.crs:
            raise ValueError(
                f"Reference {path} lacks CRS metadata; "
                "provide a GeoTIFF with CRS defined."
            )
        crs = src.crs
        transform = src.transform
        shape = (src.height, src.width)
        bounds = (src.bounds.left, src.bounds.bottom,
                 src.bounds.right, src.bounds.top)
    return crs, transform, shape, bounds


# ============================================================================
# Raster Loading and Alignment
# ============================================================================

def load_aligned_raster(
    path: Path,
    dst_shape: tuple[int, int],
    dst_transform: Affine,
    dst_crs: CRS,
    resampling: Resampling,
    apply_nodata: bool
) -> np.ndarray:
    """Load and reproject a raster to match destination grid."""
    destination = np.full(dst_shape, SENTINEL, dtype=np.float32)

    with rasterio.open(path) as src:
        if apply_nodata:
            data = src.read(1, masked=True)
            if isinstance(data, np.ma.MaskedArray):
                source = data.filled(SENTINEL).astype(np.float32, copy=False)
                src_nodata = SENTINEL
            else:
                source = data.astype(np.float32, copy=False)
                src_nodata = SENTINEL
        else:
            source = src.read(1).astype(np.float32, copy=False)
            src_nodata = None

        # Clean extreme values
        source[np.abs(source) > 1e6] = SENTINEL
        source[~np.isfinite(source)] = SENTINEL

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=SENTINEL,
        )

    destination[destination == SENTINEL] = np.nan
    return destination


# ============================================================================
# Climatology Helpers
# ============================================================================

def initialize_accumulators(
    shape: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialise running sum/count arrays for climatology."""
    sums = np.zeros(shape, dtype=np.float64)
    sums_sq = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.uint16)
    return sums, sums_sq, counts


def update_accumulators(
    sums: np.ndarray,
    sums_sq: np.ndarray,
    counts: np.ndarray,
    stack: np.ndarray
) -> None:
    """Update climatology running statistics in-place."""
    valid = np.isfinite(stack)
    contrib = np.where(valid, stack, 0.0)
    np.add(sums, contrib, out=sums)
    np.add(sums_sq, contrib * contrib, out=sums_sq)
    counts += valid.astype(counts.dtype, copy=False)


def finalize_climatology(
    sums: np.ndarray,
    sums_sq: np.ndarray,
    counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std climatology from accumulators."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.divide(sums, counts, where=counts > 0)
        mean_sq = np.divide(sums_sq, counts, where=counts > 0)
    mean[counts == 0] = np.nan
    variance = mean_sq - np.square(mean)
    variance[counts == 0] = np.nan
    variance[variance < 0] = 0.0
    std = np.sqrt(variance)
    std[counts < 2] = np.nan
    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)


def save_climatology(
    path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    months: Sequence[int],
    variable_keys: Sequence[str],
    roi_transform: Affine,
    roi_crs: CRS,
    roi_bounds: tuple[float, float, float, float]
) -> None:
    """Persist climatology to disk for later reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        climatology_mean=mean,
        climatology_std=std,
        months=np.array(months, dtype=np.int16),
        variables=np.array(variable_keys, dtype="U"),
        transform=np.array(roi_transform.to_gdal(), dtype=np.float64),
        crs=str(roi_crs),
        bounds=np.array(roi_bounds, dtype=np.float64),
    )
    logging.info(f"Saved climatology to {path}")


def load_climatology(path: Path) -> dict:
    """Load climatology NPZ archive."""
    with np.load(path) as data:
        required = {
            "climatology_mean",
            "climatology_std",
            "months",
            "variables",
            "transform",
            "crs",
            "bounds",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Climatology file {path} missing required arrays: {sorted(missing)}"
            )
        mean = data["climatology_mean"].astype(np.float32, copy=False)
        std = data["climatology_std"].astype(np.float32, copy=False)
        months = [int(val) for val in data["months"]]
        variables = [str(val) for val in data["variables"]]
        transform = Affine.from_gdal(*data["transform"])
        crs_raw = data["crs"]
        if isinstance(crs_raw, np.ndarray):
            crs_str = str(crs_raw.item()) if crs_raw.shape == () else str(crs_raw.tolist())
        else:
            crs_str = str(crs_raw)
        crs = CRS.from_string(crs_str)
        bounds = tuple(float(val) for val in data["bounds"].tolist())
    return {
        "mean": mean,
        "std": std,
        "months": months,
        "variables": variables,
        "transform": transform,
        "crs": crs,
        "bounds": bounds,
    }


# ============================================================================
# Temporary Storage Utilities
# ============================================================================

def save_raw_cache(cache_dir: Path, year: int, stack: np.ndarray) -> Path:
    """Save raw stack to temporary cache for post-processing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"raw_{year}.npy"
    np.save(cache_path, stack.astype(np.float32, copy=False))
    return cache_path


def load_raw_cache(cache_path: Path) -> np.ndarray:
    """Load raw stack from cache."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached stack missing: {cache_path}")
    return np.load(cache_path)

# ============================================================================
# Year Stack Building
# ============================================================================

def build_year_stack(
    year: int,
    months: Sequence[int],
    variables: Sequence[VariableConfig],
    roi_shape: tuple[int, int],
    roi_transform: Affine,
    roi_crs: CRS,
    skip_missing: bool
) -> tuple[np.ndarray, list[int]]:
    """Build a year stack of all variables for all months."""
    month_arrays: list[np.ndarray] = []
    processed_months: list[int] = []

    for month in months:
        monthly_layers: list[np.ndarray] = []
        missing = False

        for cfg in variables:
            try:
                file_path = cfg.find_file(year, month)
            except FileNotFoundError as exc:
                logging.warning(str(exc))
                missing = True
                break

            aligned = load_aligned_raster(
                file_path,
                roi_shape,
                roi_transform,
                roi_crs,
                cfg.resampling,
                cfg.apply_nodata,
            )
            cleaned = cfg.cleaner(aligned)
            monthly_layers.append(cleaned.astype(np.float32, copy=False))

        if missing:
            if skip_missing:
                logging.info(f"Skipping {year}-{month:02d} due to missing inputs.")
                continue
            raise FileNotFoundError(
                f"Missing inputs for {year}-{month:02d}; "
                "rerun with --skip-missing to ignore."
            )

        if not monthly_layers:
            continue

        # Stack variables for this month: shape (H, W, n_variables)
        month_stack = np.stack(monthly_layers, axis=-1)
        month_arrays.append(month_stack)
        processed_months.append(month)

        logging.debug(
            f"Stacked {', '.join(cfg.key.upper() for cfg in variables)} "
            f"for {year}-{month:02d} with shape {month_stack.shape}"
        )

    if not month_arrays:
        raise RuntimeError(
            f"No monthly data available for {year} after filtering."
        )

    # Stack months: shape (n_months, H, W, n_variables)
    full_stack = np.stack(month_arrays, axis=0).astype(np.float32, copy=False)
    return full_stack, processed_months


# ============================================================================
# Output Saving
# ============================================================================

def save_outputs(
    stack: np.ndarray,
    months: Sequence[int],
    variable_keys: Sequence[str],
    year: int,
    output_dir: Path,
    roi_transform: Affine,
    roi_crs: CRS,
    roi_bounds: tuple[float, float, float, float],
    overwrite: bool,
    climatology_path: Optional[Path],
    climatology_years: Sequence[int]
) -> None:
    """Save year stack and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"X_{year}.npz"

    if npz_path.exists() and not overwrite:
        logging.info(
            f"Output exists, skipping (use --overwrite to replace): {npz_path}"
        )
        return

    np.savez_compressed(
        npz_path,
        X=stack,
        months=np.array(months, dtype=np.int16),
        variables=np.array(variable_keys, dtype="U"),
        transform=np.array(roi_transform.to_gdal(), dtype=np.float64),
        crs=str(roi_crs),
    )
    logging.info(f"Saved {npz_path.name} (shape={stack.shape})")

    # Save metadata as JSON
    meta = {
        "year": year,
        "months": list(months),
        "variables": variable_keys,
        "shape": list(stack.shape),
        "transform": list(roi_transform.to_gdal()),
        "crs": str(roi_crs),
        "bounds": list(roi_bounds),
        "anomaly_reference": str(climatology_path) if climatology_path else None,
        "anomaly_years": [int(y) for y in climatology_years],
    }
    meta_path = output_dir / f"X_{year}_meta.json"
    with meta_path.open("w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=2)
    logging.debug(f"Wrote metadata {meta_path.name}")


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

    # Setup paths
    base_dir = args.base_dir.expanduser().resolve()
    variable_configs = build_variable_configs(base_dir)

    # Validate variables
    selected_keys = [key.lower() for key in args.variables]
    for key in selected_keys:
        cfg = variable_configs[key]
        if not cfg.directory.exists():
            raise FileNotFoundError(f"Data directory not found: {cfg.directory}")

    # Choose reference grid
    reference_file = choose_reference_file(
        args.reference, variable_configs, selected_keys
    )
    ref_crs, ref_transform, ref_shape, ref_bounds = read_reference(reference_file)
    logging.info(
        f"Reference grid: {reference_file.name} "
        f"(shape={ref_shape}, crs={ref_crs})"
    )

    # Compute ROI window
    roi_bbox = tuple(args.roi) if args.roi else DEFAULT_ROI
    roi_window, roi_transform, roi_shape = compute_roi_window(
        ref_transform, ref_crs, ref_shape, roi_bbox
    )
    roi_bounds = array_bounds(roi_shape[0], roi_shape[1], roi_transform)
    logging.info(
        f"ROI window rows {int(roi_window.row_off)}:"
        f"{int(roi_window.row_off + roi_window.height)} "
        f"cols {int(roi_window.col_off)}:"
        f"{int(roi_window.col_off + roi_window.width)} | "
        f"shape={roi_shape} bounds={roi_bounds}"
    )

    # Parse years and months
    years = validate_years(args.years)
    months = validate_months(args.months)
    selected_variables = [variable_configs[key] for key in selected_keys]
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = output_dir / "_raw_cache"

    if args.skip_missing:
        logging.warning(
            "Anomaly mode requires identical months for all years; "
            "--skip-missing may cause failures if data are absent."
        )

    # Prepare climatology options
    climatology_path = (
        args.climatology_path.expanduser().resolve()
        if args.climatology_path else output_dir / CLIMATOLOGY_FILENAME
    )

    reuse_climatology = climatology_path.exists()
    climatology_info: Optional[dict] = None
    climatology_mean: Optional[np.ndarray] = None
    climatology_std: Optional[np.ndarray] = None
    if reuse_climatology:
        climatology_info = load_climatology(climatology_path)
        logging.info(f"Loaded existing climatology from {climatology_path}")
        climatology_mean = climatology_info["mean"]
        climatology_std = climatology_info["std"]

    expected_shape = (
        len(months),
        roi_shape[0],
        roi_shape[1],
        len(selected_variables),
    )
    sums = sums_sq = counts = None
    if not reuse_climatology:
        sums, sums_sq, counts = initialize_accumulators(expected_shape)
        logging.info(
            "Computing climatology from years: %s",
            ", ".join(str(year) for year in years)
        )
    else:
        assert climatology_info is not None and climatology_mean is not None
        if climatology_mean.shape != expected_shape:
            raise ValueError(
                "Climatology shape does not match the requested stack dimensions."
            )
        if climatology_info["months"] != months:
            raise ValueError("Climatology months differ from requested months.")
        if climatology_info["variables"] != [cfg.key for cfg in selected_variables]:
            raise ValueError("Climatology variables differ from requested variables.")
        if climatology_info["transform"] != roi_transform or climatology_info["crs"] != ref_crs:
            raise ValueError(
                "Climatology grid/CRS differ from the requested ROI."
            )

    cached_years: list[tuple[int, Path, list[int]]] = []

    # Process each year (first pass)
    for year in years:
        logging.info(f"Processing year {year}")
        stack, covered_months = build_year_stack(
            year,
            months,
            selected_variables,
            roi_shape,
            roi_transform,
            ref_crs,
            args.skip_missing,
        )
        if covered_months != months:
            raise ValueError(
                "Monthly coverage mismatch detected; rerun without --skip-missing to "
                "ensure anomalies can be computed."
            )

        cache_path = save_raw_cache(cache_dir, year, stack)
        cached_years.append((year, cache_path, covered_months))

        if not reuse_climatology and sums is not None and counts is not None and sums_sq is not None:
            update_accumulators(sums, sums_sq, counts, stack)

    if not cached_years:
        raise RuntimeError("No yearly stacks were generated.")

    if reuse_climatology:
        assert climatology_mean is not None and climatology_std is not None
    else:
        assert sums is not None and counts is not None and sums_sq is not None
        climatology_mean, climatology_std = finalize_climatology(sums, sums_sq, counts)
        save_climatology(
            climatology_path,
            climatology_mean,
            climatology_std,
            months,
            [cfg.key for cfg in selected_variables],
            roi_transform,
            ref_crs,
            roi_bounds,
        )

    raw_variable_keys = [cfg.key for cfg in selected_variables]
    anomaly_keys = [f"{key}_anom" for key in raw_variable_keys]
    ndvi_derived_keys: list[str] = []
    if "ndvi" in raw_variable_keys:
        ndvi_derived_keys = [
            "ndvi_zscore",
            "ndvi_deficit",
            "ndvi_integral",
        ]
    combined_keys = raw_variable_keys + anomaly_keys + ndvi_derived_keys

    # Second pass: append anomalies and write outputs
    for year, cache_path, covered_months in cached_years:
        stack = load_raw_cache(cache_path)
        assert climatology_mean is not None and climatology_std is not None
        if stack.shape != climatology_mean.shape:
            raise ValueError(
                f"Cached stack shape {stack.shape} does not match climatology {climatology_mean.shape}"
            )
        anomaly = stack - climatology_mean
        anomaly[~np.isfinite(climatology_mean)] = np.nan

        ndvi_features = []
        if "ndvi" in raw_variable_keys:
            ndvi_idx = raw_variable_keys.index("ndvi")
            ndvi_stack = stack[..., ndvi_idx]
            ndvi_mean = climatology_mean[..., ndvi_idx]
            ndvi_std = climatology_std[..., ndvi_idx]
            ndvi_finite = np.isfinite(ndvi_stack)
            ndvi_any_valid = np.any(ndvi_finite, axis=0, keepdims=True)
            valid_mask_full = np.broadcast_to(ndvi_any_valid, ndvi_stack.shape)

            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi_z = (ndvi_stack - ndvi_mean) / ndvi_std
            invalid_std = (~np.isfinite(ndvi_std)) | (np.abs(ndvi_std) < 1e-6)
            ndvi_z[invalid_std] = np.nan
            ndvi_z[~ndvi_finite] = np.nan
            ndvi_z[~valid_mask_full] = np.nan
            ndvi_features.append(ndvi_z[..., np.newaxis])

            with np.errstate(all="ignore"):
                peak = np.nanmax(ndvi_stack, axis=0, keepdims=True)
            invalid_peak = ~ndvi_any_valid
            peak[invalid_peak] = np.nan
            ndvi_deficit = peak - ndvi_stack
            ndvi_deficit[~ndvi_finite] = np.nan
            ndvi_deficit[np.broadcast_to(invalid_peak, ndvi_stack.shape)] = np.nan
            ndvi_features.append(ndvi_deficit[..., np.newaxis])

            ndvi_integral = np.nancumsum(np.where(ndvi_finite, ndvi_stack, 0.0), axis=0)
            counts = np.cumsum(ndvi_finite, axis=0)
            ndvi_integral[counts == 0] = np.nan
            ndvi_features.append(ndvi_integral[..., np.newaxis])

        extra_features = np.concatenate(ndvi_features, axis=-1) if ndvi_features else None

        parts = [stack, anomaly]
        if extra_features is not None:
            parts.append(extra_features)
        full_stack = np.concatenate(parts, axis=-1)

        save_outputs(
            full_stack,
            covered_months,
            combined_keys,
            year,
            output_dir,
            roi_transform,
            ref_crs,
            roi_bounds,
            args.overwrite,
            climatology_path,
            years,
        )

    # Remove cache directory
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    logging.info("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
