#!/usr/bin/env python3
"""Extract UK-wide SPEI targets aligned with feature tensors from step_1.

This script reads SPEI (Standardized Precipitation-Evapotranspiration Index)
NetCDF files, reprojects them to match the grid from step_1, and saves both
continuous SPEI values and binary drought classifications.

Usage:
    python step_2_extract_spei_targets_uk.py --years 2020 2021 2022 --month 8
"""

import argparse
import calendar
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
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
DEFAULT_OUTPUT_DIR = BASE_DIR / "prepared_targets_uk"
DEFAULT_ROI = (-10.5, 49.5, 2.5, 60.8)  # UK bounding box in WGS84
DEFAULT_SPEI_DIR = BASE_DIR.parent / "UKCEH_ml_UK" / "data_UKCEH" / "data"
SENTINEL = np.float32(-9999.0)
NODATA_THRESHOLD = -1e20

# British National Grid WKT (EPSG:27700)
BNG_WKT = (
    'PROJCS["OSGB 1936 / British National Grid",'
    'GEOGCS["OSGB 1936",DATUM["OSGB_1936",SPHEROID["Airy 1830",6377563.396,299.3249646],'
    'TOWGS84[375,-111,431,0,0,0,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",49],'
    'PARAMETER["central_meridian",-2],PARAMETER["scale_factor",0.999601272],'
    'PARAMETER["false_easting",400000],PARAMETER["false_northing",-100000],'
    'UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)


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
        help="Calendar years to process."
    )
    parser.add_argument(
        "--month",
        type=int,
        default=8,
        help="1-based month index from the SPEI NetCDF (default: 8 for August)."
    )
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("min_lon", "min_lat", "max_lon", "max_lat"),
        help="Optional bounding box in WGS84 (default: UK extent)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="SPEI threshold for drought classification (default: -1.0)."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional raster/NetCDF defining the output grid."
    )
    parser.add_argument(
        "--feature-meta",
        type=Path,
        help="Metadata JSON from step_1 to enforce identical grid/transform."
    )
    parser.add_argument(
        "--spei-dir",
        type=Path,
        default=DEFAULT_SPEI_DIR,
        help="Directory containing SPEI NetCDF files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write SPEI targets."
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Disable PNG plot outputs."
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Do not save binary drought masks."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing yearly targets."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity."
    )
    return parser.parse_args(argv)


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
    bbox: Optional[tuple[float, float, float, float]]
) -> tuple[Window, Affine, tuple[int, int]]:
    """Compute ROI window from bounding box."""
    height, width = ref_shape
    if bbox is None:
        window = Window(col_off=0, row_off=0, width=width, height=height)
        return window, ref_transform, (height, width)

    left, bottom, right, top = bbox_to_reference(bbox, ref_crs)
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
# Reference Grid Selection and Reading
# ============================================================================

def choose_reference_file(
    explicit: Optional[Path],
    dataset_dirs: Sequence[Path]
) -> Path:
    """Choose a reference grid GeoTIFF."""
    if explicit:
        ref = explicit.expanduser().resolve()
        if not ref.exists():
            raise FileNotFoundError(f"Reference file not found: {ref}")
        return ref

    for directory in dataset_dirs:
        candidates = sorted(directory.glob("*.tif"))
        if candidates:
            logging.info(f"Using {candidates[0].name} as the reference grid")
            return candidates[0]

    raise FileNotFoundError(
        "Failed to locate a raster to use as the reference grid. "
        "Provide one via --reference."
    )


def read_reference(path: Path) -> tuple[CRS, Affine, tuple[int, int]]:
    """Read reference grid metadata."""
    with rasterio.open(path) as src:
        if not src.crs:
            raise ValueError(f"Reference {path} lacks CRS metadata.")
        crs = src.crs
        transform = src.transform
        shape = (src.height, src.width)
    return crs, transform, shape


def read_feature_meta(
    path: Path
) -> tuple[CRS, Affine, tuple[int, int], tuple[float, float, float, float]]:
    """Read grid definition from step_1 feature metadata JSON."""
    with path.open() as fh:
        meta = json.load(fh)

    shape = meta.get("shape")
    if not shape or len(shape) < 3:
        raise ValueError(f"Feature metadata {path} is missing a valid shape entry.")
    rows, cols = shape[1], shape[2]

    transform_vals = meta.get("transform")
    if not transform_vals or len(transform_vals) != 6:
        raise ValueError(f"Feature metadata {path} is missing the Affine transform.")
    transform = Affine.from_gdal(*transform_vals)

    crs_str = meta.get("crs")
    if not crs_str:
        raise ValueError(f"Feature metadata {path} has no CRS field.")
    crs = CRS.from_string(crs_str)

    bounds = tuple(meta.get("bounds")) if meta.get("bounds") else array_bounds(rows, cols, transform)
    return crs, transform, (rows, cols), bounds


# ============================================================================
# SPEI Data Processing
# ============================================================================

def find_spei_file(spei_dir: Path, year: int) -> Path:
    """Find SPEI NetCDF file for given year."""
    pattern = f"spei01*_mon_{year}01-{year}12.nc"
    matches = sorted(spei_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No SPEI NetCDF found for year {year} in {spei_dir}"
        )
    if len(matches) > 1:
        logging.debug(f"Multiple SPEI files for {year}, using {matches[0].name}")
    return matches[0]


def reproject_spei_band(
    dataset_path: Path,
    month_index: int,
    roi_shape: tuple[int, int],
    roi_transform: Affine,
    roi_crs: CRS
) -> np.ndarray:
    """Read and reproject a SPEI band to target grid."""
    with rasterio.open(dataset_path) as src:
        src_crs = src.crs
        if not src_crs:
            logging.warning(
                f"{dataset_path.name} has no CRS metadata; "
                "assuming EPSG:27700 (British National Grid)."
            )
            src_crs = CRS.from_wkt(BNG_WKT)

        try:
            band = src.read(month_index)
        except IndexError as exc:
            raise IndexError(
                f"Month index {month_index} unavailable in {dataset_path.name}"
            ) from exc

        # Clean data
        band = band.astype(np.float32, copy=False)
        band[band <= NODATA_THRESHOLD] = SENTINEL

        # Reproject
        destination = np.full(roi_shape, SENTINEL, dtype=np.float32)
        reproject(
            source=band,
            destination=destination,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=roi_transform,
            dst_crs=roi_crs,
            resampling=Resampling.bilinear,
            src_nodata=SENTINEL,
            dst_nodata=SENTINEL,
        )

    destination[destination == SENTINEL] = np.nan
    return destination


# ============================================================================
# Statistics and Visualization
# ============================================================================

def summarise_array(arr: np.ndarray) -> dict[str, Optional[float]]:
    """Compute summary statistics for array."""
    finite = np.isfinite(arr)
    if not finite.any():
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "nan_count": int(arr.size),
            "valid_count": 0
        }

    valid = arr[finite]
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
        "nan_count": int(arr.size - valid.size),
        "valid_count": int(valid.size),
    }


def save_plot(data: np.ndarray, output_path: Path, title: str) -> None:
    """Create and save a visualization of SPEI data."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use RdYlBu colormap: red for negative SPEI, blue for positive
    im = ax.imshow(data, cmap="RdYlBu", vmin=-2.5, vmax=2.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis("off")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="SPEI")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved plot {output_path.name}")


# ============================================================================
# Output Saving
# ============================================================================

def save_outputs(
    data: np.ndarray,
    output_dir: Path,
    year: int,
    month_index: int,
    roi_transform: Affine,
    roi_crs: CRS,
    roi_bounds: tuple[float, float, float, float],
    threshold: float,
    include_classification: bool,
    save_plot_png: bool
) -> None:
    """Save SPEI data, classification, metadata, and optional plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    month_name = calendar.month_name[month_index] if 1 <= month_index <= 12 else f"Month_{month_index}"

    # Save continuous SPEI values
    array_path = output_dir / f"y_{year}.npy"
    np.save(array_path, data.astype(np.float32, copy=False))
    logging.info(f"Saved {array_path.name}")

    # Save binary classification if requested
    classification_stats = None
    if include_classification:
        cls_path = output_dir / f"y_{year}_cls.npy"
        cls = np.full(data.shape, 255, dtype=np.uint8)
        finite = np.isfinite(data)
        cls[finite] = (data[finite] < threshold).astype(np.uint8)
        np.save(cls_path, cls)
        logging.info(f"Saved {cls_path.name}")

        drought_pixels = int(np.count_nonzero(cls == 1))
        valid_pixels = int(np.count_nonzero(finite))
        classification_stats = {
            "drought_pixels": drought_pixels,
            "valid_pixels": valid_pixels,
            "drought_percentage": (drought_pixels / valid_pixels * 100.0) if valid_pixels > 0 else 0.0
        }

    # Save metadata
    stats = summarise_array(data)
    meta = {
        "year": year,
        "month_index": month_index,
        "month_name": month_name,
        "threshold": threshold,
        "shape": list(data.shape),
        "transform": list(roi_transform.to_gdal()),
        "crs": str(roi_crs),
        "bounds": list(roi_bounds),
        "statistics": stats,
        "classification": classification_stats,
    }
    meta_path = output_dir / f"y_{year}_meta.json"
    with meta_path.open("w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=2)
    logging.debug(f"Saved metadata {meta_path.name}")

    # Save plot if requested
    if save_plot_png and np.isfinite(data).any():
        plot_path = output_dir / f"y_{year}_plot.png"
        save_plot(data, plot_path, f"SPEI - {month_name} {year}")


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

    # Validate month
    if args.month < 1 or args.month > 12:
        raise ValueError("--month must be between 1 and 12 inclusive")

    # Setup SPEI directory
    spei_dir = args.spei_dir.expanduser().resolve()
    if not spei_dir.exists():
        raise FileNotFoundError(f"SPEI directory not found: {spei_dir}")

    years = sorted({int(year) for year in args.years})

    # Determine target grid
    if args.feature_meta:
        logging.info(f"Reading grid definition from {args.feature_meta}")
        roi_crs, roi_transform, roi_shape, roi_bounds = read_feature_meta(args.feature_meta)
        logging.info(f"Grid from feature meta: shape={roi_shape} bounds={roi_bounds}")
    else:
        # Use reference file or find one
        reference_dirs = [
            BASE_DIR.parent / "UKCEH_ml_UK" / "data_s2_ndvi_uk",
            BASE_DIR.parent / "UKCEH_ml_UK" / "data_viirs_lst_uk"
        ]
        reference_file = choose_reference_file(args.reference, reference_dirs)
        ref_crs, ref_transform, ref_shape = read_reference(reference_file)
        roi_bbox = tuple(args.roi) if args.roi else DEFAULT_ROI
        roi_window, roi_transform, roi_shape = compute_roi_window(
            ref_transform, ref_crs, ref_shape, roi_bbox
        )
        roi_bounds = array_bounds(roi_shape[0], roi_shape[1], roi_transform)
        roi_crs = ref_crs
        logging.info(
            f"ROI window rows {int(roi_window.row_off)}:"
            f"{int(roi_window.row_off + roi_window.height)} "
            f"cols {int(roi_window.col_off)}:"
            f"{int(roi_window.col_off + roi_window.width)} | "
            f"shape={roi_shape} bounds={roi_bounds}"
        )

    logging.info(f"Target grid shape={roi_shape} crs={roi_crs}")

    # Setup output directory
    output_dir = args.output_dir.expanduser().resolve()

    # Process each year
    for year in years:
        array_path = output_dir / f"y_{year}.npy"
        cls_path = output_dir / f"y_{year}_cls.npy"

        # Check if output exists
        if not args.overwrite and array_path.exists():
            logging.info(
                f"Target for {year} exists, skipping (use --overwrite to replace)"
            )
            continue
        if not args.overwrite and (not args.skip_classification) and cls_path.exists():
            logging.info(
                f"Classification for {year} exists, skipping (use --overwrite to replace)"
            )
            continue

        # Find and process SPEI file
        spei_file = find_spei_file(spei_dir, year)
        logging.info(f"Processing {year} from {spei_file.name}")

        data = reproject_spei_band(
            spei_file,
            args.month,
            roi_shape,
            roi_transform,
            roi_crs
        )

        save_outputs(
            data,
            output_dir,
            year,
            args.month,
            roi_transform,
            roi_crs,
            roi_bounds,
            args.threshold,
            include_classification=not args.skip_classification,
            save_plot_png=not args.skip_plot,
        )

    logging.info("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
