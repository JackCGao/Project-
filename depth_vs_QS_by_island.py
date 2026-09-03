#!/usr/bin/env python3
"""
Erosional depth (d) vs Stream Power Proxy (Q x S) -- one log-log plot per
island (not a single all-islands-combined plot like depth_vs_QS.py).

Deliberately does NOT go through maxflux_stream_points.py's output: no
hydro-proximity-to-Streams_reprojected.shp refinement, no pre-built
combined points shapefile. Instead, Q comes from each island's weighted
flow accumulation raster in Island Data/weighted_flow/ (D8 max-flux,
already masked to stream pixels -- nodata everywhere else), which also
doubles as the mask: S and d are only sampled wherever that raster has a
valid (non-nodata) value, with no separate streams==1 raster needed.

QS is the simple product Q*S with no separate exponents fit on Q and S
individually (i.e. the single-exponent form d = k*(QS)^m, same method as
recreation.py's Fit 1 / depth_vs_QS.py -- not the multivariate
Q^c*S^n form in weathering_stream_power_regression.py or
depth_over_age_vs_QS.py's second plot).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from usefulfunctions import load_clean

# Derived from this file's own location rather than hardcoded, since the
# project folder has been renamed before (it used to be "Summer Work 2026",
# with spaces) and a stale hardcoded path here silently no-ops every path
# built from it -- same caveat as recreation.py's PROJECT_DIR.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = f'{base_dir}/Temp Output Placements'

# Per-island rasters live under Island Data/ (see recreation.py's note on
# the same reorg).
island_data_dir = f'{base_dir}/Island Data'

# Weighted flow accumulation (D8 max-flux, already masked to stream pixels)
# -- one raster per island, not under each island's own folder.
weighted_flow_dir = f'{island_data_dir}/weighted_flow'

_hawaii_dir    = f'{island_data_dir}/hawaii'
_kahoolawe_dir = f'{island_data_dir}/kahoolawe'
_oahu_dir      = f'{island_data_dir}/oahu/new'

island_files = {
    'hawaii': {
        'flow_accum': f'{weighted_flow_dir}/hawaii_d8maxflux_masked_streams_albers.tif',
        'slope':      f'{_hawaii_dir}/hawaii_slope_nans.tif',
        'erosion':    f'{_hawaii_dir}/hawaii_erosion_nans.tif',
    },
    'kahoolawe': {
        'flow_accum': f'{weighted_flow_dir}/kahoolawe_d8maxflux_masked_streams_albers.tif',
        'slope':      f'{_kahoolawe_dir}/kahoolawe_slope_nans.tif',
        'erosion':    f'{_kahoolawe_dir}/kahoolawe_erosion_nans.tif',
    },
    'oahu': {
        'flow_accum': f'{weighted_flow_dir}/oahu_d8maxflux_masked_streams_albers.tif',
        'slope':      f'{_oahu_dir}/oahu_slope_nans.tif',
        'erosion':    f'{_oahu_dir}/oahu_erosion_nans.tif',
    },
    'kauai': {
        'flow_accum': f'{weighted_flow_dir}/kauai_d8maxflux_masked_streams_albers.tif',
        'slope':      f'{island_data_dir}/kauai/new/kauai_slope.tif',
        'erosion':    f'{island_data_dir}/kauai/new/kauai_erosion_nans.tif',
    },
    'lanai': {
        'flow_accum': f'{weighted_flow_dir}/lanai_d8maxflux_masked_streams_albers.tif',
        'slope':      f'{island_data_dir}/lanai/new/lanai_slope.tif',
        'erosion':    f'{island_data_dir}/lanai/new/lanai_erosion_nans.tif',
    },
    'molokai': {
        'flow_accum': f'{weighted_flow_dir}/molokai_d8maxflux_masked_streams_albers.tif',
        'slope':      f'{island_data_dir}/molokai/new/molokai_slope_nans.tif',
        'erosion':    f'{island_data_dir}/molokai/new/molokai_erosion_nans.tif',
    },
    'maui': {
        'flow_accum': f'{weighted_flow_dir}/maui_d8maxflux_masked_streams_albers.tif',
        'slope':      f'{island_data_dir}/maui/new/maui_slope_nans.tif',
        'erosion':    f'{island_data_dir}/maui/new/maui_erosion_nans.tif',
    },
}


def get_island_QS_d(island):
    """Q (weighted flow accumulation, already stream-masked) doubles as the
    mask: S and d are only kept where Q itself is finite (i.e. where the
    weighted_flow raster has a valid, non-nodata pixel) -- no separate
    streams==1 raster and no maxflux_stream_points.py involvement."""
    paths = island_files[island]

    flow_accum = load_clean(paths['flow_accum'])
    slope_rast = load_clean(paths['slope'])
    erosion    = load_clean(paths['erosion'])

    valid_mask = (np.isfinite(flow_accum) & np.isfinite(slope_rast) &
                  np.isfinite(erosion))

    Q = flow_accum[valid_mask]
    S = slope_rast[valid_mask]
    d = erosion[valid_mask]

    pos_mask = (Q > 0) & (S > 0) & (d >= 1)
    return Q[pos_mask] * S[pos_mask], d[pos_mask]


def plot_island(island, QS, d_vals):
    log_QS = np.log10(QS)
    log_d = np.log10(d_vals)

    m, b, r, p, se = stats.linregress(log_QS, log_d)
    r2 = r ** 2
    rho, _ = stats.spearmanr(QS, d_vals)
    print(f"[{island}]  log10(d) = {m:.6f} * log10(QS) + {b:.6f}   "
          f"(k = 10^b = {10**b:.6g}, R2 = {r2:.4f}, p = {p:.3g}, "
          f"slope SE = {se:.3g}, rho = {rho:.4f}, n = {len(d_vals):,})")

    x_fit = np.logspace(log_QS.min(), log_QS.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(QS, d_vals, s=3, alpha=0.115, edgecolors='none', color='#888888')
    ax.plot(x_fit, (10 ** b) * (x_fit ** m), color='#3a6ea5', linewidth=1.8,
            label='OLS fit')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel(r'Stream Power Proxy ($Q \times S$)')
    ax.set_ylabel('Erosional Depth (d) [m]')
    ax.set_title(island.capitalize())

    textstr = (f"Spearman's rho = {rho:.4f}\n"
               f"Log-Log R2 = {r2:.4f}\n"
               f"k = {10**b:.4g},  m = {m:.4f}")
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    out_path = f'{output_dir}/{island}_depth_vs_QS_no_streampoints.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    for island in island_files:
        QS, d_vals = get_island_QS_d(island)
        if len(QS) < 10:
            print(f"[{island}] too few valid pixels ({len(QS)}), skipping")
            continue
        plot_island(island, QS, d_vals)


if __name__ == '__main__':
    main()
