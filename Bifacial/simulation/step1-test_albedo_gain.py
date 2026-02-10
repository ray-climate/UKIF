#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test_albedo_gain.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        21/08/2025 14:58

# STEP-1: Quick BG(α) curves at one latitude using pvlib+pvfactors
# - Fixed-tilt array modeled via pvfactors "tracker that never moves" trick
# - Months: Jan, Apr, Jul, Oct (edit as you like)
# - Output: plot of bifacial gain vs albedo per month

import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import calendar
import pvlib
import os

output_dir = './figures'  # where to save figures (if needed)
# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -------------------- QUIET OLD-PACKAGE NOISE (pvlib 0.9.x + pvfactors) --------------------
# Silence Shapely 2.0 deprecations triggered by old pvfactors
try:
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except Exception:
    pass  # older Shapely may not expose this class

# Silence divide-by-zero/invalid warnings from pvfactors viewfactor internals
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r".*pvfactors\.viewfactors\.vfmethods.*",
)
# Optional: hush pvfactors logger if present
logging.getLogger("pvfactors").setLevel(logging.ERROR)

# -------------------- USER SETTINGS (edit if desired) --------------------
lat = 53.0                # representative UK latitude (deg)
lon = -2.0                # any central UK lon is fine; doesn't affect BG materially
tz  = 'Europe/London'

tilt = 35.0               # deg
surface_azimuth = 180.0   # south-facing
axis_azimuth = (surface_azimuth - 90) % 360  # fixed-tilt trick

pvrow_width  = 2.0        # m (row-normal module/row width)
pitch        = 6.0        # m (row-to-row spacing)
gcr          = pvrow_width / pitch
pvrow_height = 1.0        # m (lower edge height)
bifaciality  = 0.70

months = [1, 4, 7, 10]    # test months
albedo_grid = np.linspace(0.05, 0.60, 12)  # sweep range (extend if you expect snow)
year = 2019               # non-leap; only solar geometry & clear-sky used

# Clear-sky options:
clearsky_model = 'ineichen'  # 'ineichen' (default) or 'haurwitz'

# -------------------- HELPERS --------------------
def month_timerange(y, m, tz):
    start = pd.Timestamp(f'{y}-{m:02d}-01', tz=tz)
    end = start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
    return pd.date_range(start, end, freq='1h', tz=tz)

def run_month_BG(lat, lon, tz, y, m, albedo_val):
    loc = pvlib.location.Location(lat, lon, tz=tz, altitude=100)
    times = month_timerange(y, m, tz)

    solpos = loc.get_solarposition(times)
    if clearsky_model == 'ineichen':
        cs = loc.get_clearsky(times, model='ineichen')
    elif clearsky_model == 'haurwitz':
        # simple Haürwitz GHI + DNI via DIRINT for completeness
        cs_ghi = pvlib.clearsky.haurwitz(solpos['apparent_zenith'])
        cs_ghi.index = times
        dni = pvlib.irradiance.dirint(cs_ghi['ghi'], solpos['zenith'], times)
        dhi = cs_ghi['ghi'] - pvlib.irradiance.beam_component(
            tilt=0, surface_azimuth=0,
            solar_zenith=solpos['zenith'],
            solar_azimuth=solpos['azimuth'],
            dni=dni
        )
        cs = pd.DataFrame({'dni': dni, 'ghi': cs_ghi['ghi'], 'dhi': dhi})
    else:
        raise ValueError("Unknown clearsky_model")

    # Be slightly stricter than 90° to avoid numerically degenerate sun-near-horizon cases
    mask = (solpos['apparent_zenith'] < 89.5)
    if mask.sum() == 0:
        return 0.0, 0.0

    # ---- NOTE: pvlib 0.9.x returns 4 arrays (not a DataFrame) ----
    with np.errstate(all='ignore'):  # suppress transient NaNs/infs from vfmethods
        poa_front, poa_back, poa_front_abs, poa_back_abs = pvlib.bifacial.pvfactors.pvfactors_timeseries(
            solar_azimuth=solpos.loc[mask, 'azimuth'],
            solar_zenith=solpos.loc[mask, 'apparent_zenith'],
            surface_azimuth=surface_azimuth,
            surface_tilt=tilt,
            axis_azimuth=axis_azimuth,
            timestamps=times[mask],
            dni=cs.loc[mask, 'dni'],
            dhi=cs.loc[mask, 'dhi'],
            gcr=gcr,
            pvrow_height=pvrow_height,
            pvrow_width=pvrow_width,
            albedo=pd.Series(float(albedo_val), index=times[mask]),
            n_pvrows=3, index_observed_pvrow=1
        )

    # Integrate with dt from the timestamps
    t = pd.DatetimeIndex(times[mask])
    dt_h = t.to_series().diff().dt.total_seconds().fillna(method='bfill') / 3600.0

    A_front = (pd.Series(poa_front_abs, index=t) * dt_h).sum()
    A_back  = (pd.Series(poa_back_abs,  index=t) * dt_h).sum()

    if A_front <= 0 or not np.isfinite(A_front):
        return 0.0, 0.0

    A_bi = A_front + bifaciality * A_back
    BG = A_bi / A_front - 1.0
    return float(BG), float(A_front)

# -------------------- MAIN: sweep albedo and plot --------------------
print("pvlib version:", pvlib.__version__)
try:
    import solarfactors
    print("solarfactors version:", getattr(solarfactors, '__version__', '(installed)'))
except Exception:
    print("Note: pvfactors backend 'solarfactors' must be installed. pip install solarfactors or pip install pvlib[optional]")

results = {}  # month -> np.array(BG) aligned to albedo_grid
for m in months:
    BGs = []
    for a in albedo_grid:
        BG, Amono = run_month_BG(lat, lon, tz, year, m, a)
        BGs.append(BG)
    results[m] = np.array(BGs)

# Fit simple BG = k0 + k1*alpha per month (helps sanity-check linearity)
coeffs = {m: np.polyfit(albedo_grid, results[m], deg=1) for m in months}
for m in months:
    k1, k0 = coeffs[m]
    print(f"Month {m:02d}: BG(alpha) ≈ {k0:+.4f} + {k1:+.4f}·alpha")

# -------------------- Plot --------------------
cmap = cm.get_cmap("tab10", len(months))  # 10 distinct colors

plt.figure(figsize=(8, 4))
for i, m in enumerate(months):
    month_name = calendar.month_name[m]
    plt.plot(
        albedo_grid, results[m],
        label=month_name,
        color=cmap(i), linewidth=2, marker='o'
    )

plt.xlabel('Albedo', fontsize=16)
plt.ylabel('Bifacial Gain Factor', fontsize=16)
plt.title('Bifacial gain at %.1f°' % lat, fontsize=16)

plt.legend(fontsize=16)  # month names only
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bifacial_gain_vs_albedo.png'), dpi=300)
