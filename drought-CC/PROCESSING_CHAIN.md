# Drought Prediction Processing Chain

## Overview

A UK-wide drought prediction pipeline combining satellite data (Sentinel-2 NDVI, VIIRS LST), climate reanalysis (ERA5-Land), and ground-truth drought indices (SPEI) to produce 1 km predictions via XGBoost, then optionally downscale to 10 m using high-resolution NDVI.

```
Step 1: Build Yearly Inputs       (raw data → 1 km aligned feature cubes)
    ↓
Step 2: Extract SPEI Targets      (SPEI NetCDF → 1 km target maps)
    ↓
Step 3: Flatten & Train XGBoost   (feature matrix → trained model)
    ↓
Step 4: Predict SPEI Maps         (model + features → 1 km SPEI predictions)
    ↓
    ├─ [Option A] Beta Correction      (Steps 5+6: residual-based downscaling)
    └─ [Option B] TsHARP Downscaling   (Step 7: regression + residual)
```

---

## Step 1: Build Yearly Inputs (`step_1_build_yearly_inputs_uk.py`)

**Purpose**: Align multi-source satellite and climate data onto a common 1 km WGS84 grid, compute climatological anomalies, and generate NDVI-derived features.

### Input data sources

| Source | Variable | Native resolution |
|--------|----------|-------------------|
| Sentinel-2 | NDVI | ~10 m (aggregated to 1 km) |
| VIIRS | LST (Land Surface Temperature) | ~1 km |
| ERA5-Land | Soil evaporation | ~9 km (resampled to 1 km) |
| ERA5-Land | Total precipitation | ~9 km (resampled to 1 km) |

### Processing

1. **Reference grid**: 1 km WGS84 grid over UK (lon [-10.5, 1.77], lat [49.9, 60.8])
2. **Data cleaning per variable**:
   - NDVI: mask values outside [-1, 1]
   - LST: mask values < 200 K
   - Precipitation: convert m → mm, ensure non-negative
   - Soil evaporation: convert m → mm, flip sign (ERA5 convention)
3. **Reproject** each GeoTIFF to target 1 km grid (bilinear resampling)
4. **Climatology computation** (first pass): pixel-wise mean and std over all years
5. **Anomaly computation** (second pass):
   - `var_anom = var_raw - climatology_mean`
6. **NDVI-derived features**:
   - `ndvi_zscore = (NDVI - mean) / std` — standardised anomaly
   - `ndvi_deficit = max(NDVI_all_months) - NDVI_current` — distance below seasonal peak
   - `ndvi_integral = cumsum(NDVI)` — integrated vegetation productivity

### Output

```
prepared_inputs_uk/
  ├── X_{year}.npz           # (n_months, 1214, 1366, 11) float32
  ├── X_{year}_meta.json     # metadata: months, variables, CRS, transform
  └── X_climatology_*.npz    # climatology mean and std arrays
```

**Variables (11)**: ndvi, lst, soil_evap, precip, ndvi_anom, lst_anom, soil_evap_anom, precip_anom, ndvi_zscore, ndvi_deficit, ndvi_integral

### Grid properties

| Property | Value |
|----------|-------|
| CRS | EPSG:4326 (WGS84) |
| Grid size | 1214 rows × 1366 cols |
| Pixel size | 0.008983° (~1 km at 55°N) |
| Extent | UK landmass + surrounding water |
| Valid coverage | ~31% of grid (rest is ocean/coast) |

---

## Step 2: Extract SPEI Targets (`step_2_extract_spei_targets_uk.py`)

**Purpose**: Load SPEI (Standardised Precipitation-Evapotranspiration Index) from NetCDF, reproject to match Step 1 grid, and classify drought.

### Processing

1. Load SPEI from NetCDF (source CRS: British National Grid EPSG:27700)
2. Extract target month band (default: month 8 = August)
3. Mask fill values (< -1e20) as NaN
4. Reproject from BNG to WGS84 1 km grid (bilinear)
5. Binary classification: `drought = 1 if SPEI ≤ -1.0, else 0`

### Output

```
prepared_targets_uk/
  ├── y_{year}.npy            # (1214, 1366) float32 continuous SPEI
  ├── y_{year}_cls.npy        # (1214, 1366) uint8 binary drought class
  └── y_{year}_meta.json      # statistics, classification counts
```

### SPEI characteristics

- Range: typically [-3, +3]
- Negative values = drought; positive = wet
- Classification threshold: SPEI ≤ -1.0 → drought
- Example 2022 (severe drought year): 55.7% of valid pixels classified as drought

---

## Step 3: Train XGBoost Model (`step_3_flatten_inputs_uk.py`)

**Purpose**: Flatten feature cubes to sample vectors, train XGBoost regression model to predict SPEI from satellite/climate features.

### Processing

1. **Feature loading**: for each year, load `X_{year}.npz`, subset to training months (May-Jul) and selected variables
2. **Flatten**: reshape (months, H, W, vars) → (H×W, months×vars) per year, concatenate across years
3. **Validity filter**: keep pixels with finite target and at least one finite feature
4. **Train/test split**: 80/20, stratified by year
5. **Sample weighting**: optional emphasis on drought pixels (SPEI ≤ threshold)
6. **Train XGBoost** regression (`reg:squarederror`)
7. **Cross-validate** (k-fold) for stability assessment
8. **Final model**: retrain on all data

### Model configuration

```
n_estimators=140, max_depth=5, learning_rate=0.08,
subsample=0.9, colsample_bytree=0.9
```

### Feature labels

Each feature is named `{Month}_{VARIABLE}`, e.g. `May_LST`, `Jun_NDVI_ANOM`, `Jul_PRECIP`. Total features = n_months × n_variables.

### Output

```
prepared_inputs_uk/{month_group}/
  ├── trained_xgb_model.json       # serialised XGBoost Booster
  ├── trained_xgb_metrics.json     # R², RMSE, MAE, CV results, feature importance
  ├── feature_importance_gain.png
  ├── predicted_vs_observed.png
  └── residuals_*.png
```

### Expected performance

| Metric | Train | Test |
|--------|------:|-----:|
| R² | ~0.68 | ~0.64 |
| RMSE | ~0.42 | ~0.46 |
| MAE | ~0.31 | ~0.35 |

Cross-validation std ~0.01 (stable across folds).

---

## Step 4: Predict SPEI Maps (`step_4_predict_map_uk.py`)

**Purpose**: Apply trained XGBoost model to feature arrays to generate spatial SPEI prediction maps at 1 km.

### Processing

1. Load trained model and its metadata (months, variables)
2. Load feature archive for prediction year, subset to matching months/variables
3. Flatten to (H×W, n_features), mask invalid pixels
4. Predict SPEI via `xgb.Booster.predict()`
5. Reshape back to (H, W) spatial grid
6. Optionally compare with observed targets

### Output

```
{model_dir}/
  ├── predicted_spei_{year}.npy      # (1214, 1366) float32
  ├── predicted_spei_{year}_map.png
  └── comparison_spei_{year}.png     # if observed targets available
```

This is the **1 km SPEI prediction** — the base product before downscaling.

---

## Downscaling Option A: Beta Correction (Steps 5 + 6)

### Step 5: Calibrate Beta (`calibrate_ndvi_beta.py`)

**Purpose**: Relate XGBoost prediction residuals to NDVI changes, fitting a scalar factor β for spatial downscaling.

**Model**:
```
residual = SPEI_predicted - SPEI_observed ≈ β × ΔNDVI
```

**Processing**:
1. Recompute XGBoost predictions on training data
2. Compute residuals: `R = predicted - observed`
3. Compute NDVI metric (delta/mean/last) per pixel
4. Least-squares fit: `β = Σ(ΔNDVI × R) / Σ(ΔNDVI²)`
5. Optional dual calibration: separate β for global vs stress pixels

**Output**: `beta_calibration.json` containing β value, correlation, sample counts.

### Step 6: Apply Correction (`apply_beta_ndvi_correction.py`)

**Purpose**: Downscale 1 km SPEI to 10 m using high-resolution Sentinel-2 NDVI and the calibrated β.

**Method**:
```
NDVI_mean_1km = aggregate(NDVI_10m → 1 km)
NDVI_delta    = NDVI_10m - upsample(NDVI_mean_1km → 10 m)
SPEI_10m      = upsample(SPEI_1km → 10 m) + β × NDVI_delta
```

Each 10 m pixel's SPEI = coarse prediction + β × (local NDVI deviation from 1 km mean).

**Limitation**: Near-zero correlation between NDVI deviations and model residuals (r ≈ -0.05 to +0.06 in practice), meaning the β correction adds minimal spatial information.

---

## Downscaling Option B: TsHARP (`tsHARP_downscale.py`)

**Purpose**: Alternative downscaling using the TsHARP (Temperature Sharpening) method — fit a direct NDVI→SPEI relationship at 1 km, apply it at 10 m, and add an upsampled residual for mass conservation.

### Method

```
At 1 km:  SPEI_hat = f(NDVI_1km)                    (polynomial regression)
          R_1km    = SPEI_observed - SPEI_hat         (residual)

At 10 m:  SPEI_10m = f(NDVI_anom_10m) + R_1km↑10m   (regression + upsampled residual)
```

The residual guarantees that aggregating the 10 m product back to 1 km recovers the observed SPEI.

### Five phases

#### Phase 1: Diagnostic

For each NDVI variable (ndvi, ndvi_anom, ndvi_deficit), compute Pearson r with observed SPEI at 1 km across all training years. Auto-select the variable with highest |r|.

Typical results:

| Variable | r | R² |
|----------|---:|---:|
| ndvi | +0.20 | 0.041 |
| ndvi_anom | +0.25 | 0.062 |
| ndvi_deficit | -0.17 | 0.028 |

#### Phase 2: Fit regression at 1 km

Pool all training years. Fit polynomial (linear or quadratic) NDVI → SPEI via `np.polyfit`. Report R², RMSE, coefficients.

Example (linear, ndvi_anom): `SPEI ≈ 3.15 × ndvi_anom − 0.27`

#### Phase 3: Compute 1 km residual

For the prediction year:
- `SPEI_hat = f(NDVI_1km)`
- `R_1km = SPEI_observed - SPEI_hat`

Save residual as GeoTIFF + NPY.

#### Phase 4: Downscale to 10 m

1. Load 10 m Sentinel-2 NDVI, compute temporal metric (e.g. mean of May-Jul)
2. **Anomaly conversion** (if using ndvi_anom):
   - Load 1 km NDVI climatology from Step 1
   - Compute sensor calibration offset: `offset = mean(NDVI_10m aggregated to 1km) - climatology`
   - Convert: `NDVI_anom_10m = (NDVI_raw_10m - offset) - climatology_upsampled_10m`
3. Apply regression: `SPEI_reg_10m = f(NDVI_anom_10m)`
4. Upsample residual: `R_1km → R_10m` via bilinear interpolation
5. Combine: `SPEI_10m = SPEI_reg_10m + R_10m`

#### Phase 5: Validation

- Aggregate 10 m SPEI back to 1 km via average resampling
- Compare with observed 1 km SPEI: Pearson r, RMSE, MAE, bias
- Generate zoom comparison panels (NDVI | SPEI 1km | TsHARP SPEI 10m)

Typical results (2022, UK-wide S2 NDVI):

| Metric | Value |
|--------|------:|
| Pearson r | 0.92 |
| RMSE | 0.16 |
| MAE | 0.11 |
| Mean bias | -0.02 |

### Output

```
tsHARP_output/
  ├── diagnostic_correlations.json
  ├── diagnostic_{var}.png              # hexbin scatterplots
  ├── tsHARP_regression_model.json      # polynomial coefficients
  ├── tsHARP_regression_fit.png
  ├── spei_observed_1km_{year}.tif      # observed SPEI (input)
  ├── spei_regression_1km_{year}.tif    # f(NDVI) estimate
  ├── residual_1km_{year}.tif           # R = observed - regression
  ├── residual_1km_{year}.npy
  ├── residual_map_{year}.png
  ├── tsHARP_spei_10m_{year}.tif        # final 10 m SPEI product
  ├── tsHARP_spei_10m_{year}.npy
  ├── tsHARP_regression_10m_{year}.tif
  ├── residual_upsampled_10m_{year}.tif
  ├── tsHARP_validation_report.json
  ├── mass_conservation_check.png
  └── zoom_comparison.png
```

---

## Comparison: Beta Correction vs TsHARP

| Aspect | Beta Correction | TsHARP |
|--------|----------------|--------|
| Relationship modelled | residual ≈ β × ΔNDVI | SPEI = f(NDVI) directly |
| Requires trained XGBoost model | Yes (for residuals) | No (uses observed SPEI) |
| NDVI-residual correlation | Near zero (r ≈ ±0.05) | N/A |
| Mass conservation | Not guaranteed | Guaranteed by residual term |
| Validation r (agg 10m→1km) | — | 0.92 |
| Bias | — | -0.02 |
| Sensor calibration handling | None | Offset correction built-in |

TsHARP is the recommended approach for this dataset.

---

## Execution Workflow

```bash
# Step 1: Build feature cubes
python step_1_build_yearly_inputs_uk.py \
  --years 2020 2021 2022 --months 5 6 7 8

# Step 2: Extract SPEI targets
python step_2_extract_spei_targets_uk.py \
  --years 2020 2021 2022 --month 8 \
  --feature-meta prepared_inputs_uk/X_2020_meta.json

# Step 3: Train XGBoost
python step_3_flatten_inputs_uk.py \
  --years 2020 2021 2022 --months 5 6 7

# Step 4: Predict 1 km SPEI
python step_4_predict_map_uk.py \
  --year 2022 \
  --model-path prepared_inputs_uk/may_jun_jul/trained_xgb_model.json

# Step 7: TsHARP downscaling to 10 m
python tsHARP_downscale.py \
  --years 2020 2021 2022 \
  --prediction-year 2022 \
  --ndvi-months 5 6 7 \
  --ndvi-metric mean \
  --regression-type linear \
  --ndvi-files S2_NDVI_10m/uk_ndvi_2022_05.tif \
               S2_NDVI_10m/uk_ndvi_2022_06.tif \
               S2_NDVI_10m/uk_ndvi_2022_07.tif \
  --output-dir tsHARP_output
```

---

## Feature Engineering Summary

| Feature | Source | Description |
|---------|--------|-------------|
| ndvi | Sentinel-2 | Normalised Difference Vegetation Index [-1, 1] |
| lst | VIIRS | Land Surface Temperature (K) |
| soil_evap | ERA5-Land | Bare soil evaporation (mm/month) |
| precip | ERA5-Land | Total precipitation (mm/month) |
| *_anom | Derived | Raw value minus climatological mean |
| ndvi_zscore | Derived | (NDVI - mean) / std |
| ndvi_deficit | Derived | Seasonal peak NDVI minus current NDVI |
| ndvi_integral | Derived | Cumulative sum of NDVI over months |

**Target**: SPEI-1 (1-month Standardised Precipitation-Evapotranspiration Index), August. Drought threshold: SPEI ≤ -1.0.
