# TsHARP SPEI Downscaling — Delivery Package

Downscale a 1 km SPEI drought map to **10 m** using Sentinel-2 NDVI via the TsHARP method.

## What you need to provide

Monthly Sentinel-2 NDVI GeoTIFFs at **10 m resolution** for **May, June and July** of your target year.
Everything else (regression model, residual, climatology) is pre-computed and included in `precomputed/`.

## Method summary

TsHARP splits the 1 km SPEI signal into two parts:

```
SPEI_10m = f(NDVI_10m)  +  upsample(R_1km)
```

| Term | Meaning |
|---|---|
| `f(NDVI_10m)` | Polynomial regression (NDVI → SPEI) applied at 10 m |
| `R_1km` | Residual = SPEI_1km − f(NDVI_1km): the part of SPEI not explained by NDVI |

The residual is pre-computed at 1 km and upsampled to 10 m. This guarantees that when the 10 m output is aggregated back to 1 km, it recovers the original 1 km SPEI (**mass conservation**).

The regression was trained on `ndvi_anom` (NDVI anomaly relative to the multi-year mean), so your raw NDVI is automatically converted to anomaly space using the supplied climatology before the regression is applied.

## Pre-computed files (`precomputed/`)

| File | Description |
|---|---|
| `tsHARP_regression_model.json` | Polynomial coefficients and metadata (trained on 2020–2022, months May–Jul) |
| `residual_1km.tif` | 1 km SPEI residual GeoTIFF for the target year |
| `climatology_ndvi_1km.tif` | Multi-year mean NDVI at 1 km (May–Jul average, 2020–2022), used to compute NDVI anomaly |

## Installation

```bash
pip install numpy rasterio matplotlib
```

Or with conda:

```bash
conda install -c conda-forge numpy rasterio matplotlib
```

## Usage

```bash
python apply_tsHARP_10m.py \
    --ndvi-files NDVI_May.tif NDVI_Jun.tif NDVI_Jul.tif \
    --year 2022 \
    --output-dir ./output_2022
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--ndvi-files` | Yes | 10 m NDVI GeoTIFFs — one per month, in order (May, Jun, Jul) |
| `--year` | Yes | Target year (used for output file names) |
| `--output-dir` | No | Output directory (default: `./tsHARP_output`) |
| `--regression-model` | No | Override path to regression JSON |
| `--residual-tif` | No | Override path to residual GeoTIFF |
| `--climatology-tif` | No | Override path to climatology GeoTIFF |

### Outputs

| File | Description |
|---|---|
| `spei_10m_YEAR.tif` | Downscaled SPEI at 10 m (GeoTIFF, float32) |
| `spei_10m_YEAR.npy` | Same array as NumPy binary |
| `spei_10m_YEAR_map.png` | Quick-look map (NDVI anomaly + SPEI side by side) |

## Notes

- All NDVI files must share the same grid (extent, resolution, CRS).
- The regression expects **3 monthly NDVI files** matching the training months (May, June, July). Providing files in a different order will silently produce wrong results.
- SPEI values below −1.0 indicate moderate-to-severe drought.
