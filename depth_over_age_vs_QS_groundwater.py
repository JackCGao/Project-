#!/usr/bin/env python3
"""
Combined plot: (Erosional Depth / Median Volcanic-Province Age) vs Net
Stream Power Proxy (Q_net x S), for all islands overlaid in one log-log
scatter -- groundwater-aware version of depth_over_age_vs_QS.py.

Point locations (and the raw D8 max-flux Q used only for the same
positivity/masking check as before) come from maxflux_stream_points.py's
per-island output (watershed_outputs/{island}/maxflux_stream_points.shp) --
stream-pixel points already restricted to the island's own stream mask and
within 50 px of the statewide Streams_reprojected.shp hydrography layer --
rather than loading the raw flow_accum raster and masking it here.

Q_net comes from each island's mass_flux raster (WhiteboxTools d8_mass_flux
output: precipitation routed downstream minus groundwater recharge
absorption -- see recreation.py's compute_and_save_net_precip_accum),
sampled at those same points, so this factors in groundwater recharge
instead of using raw flow accumulation.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats

from usefulfunctions import sample_raster_at_points
from maxflux_stream_points import OUTPUT_DIR

base_dir = '/Users/jackgao/Summer Work 2026'
output_dir = '/Users/jackgao/Summer Work 2026/Temp Output Placements'

MEDIAN_AGE_CSV = '/Users/jackgao/Summer Work 2026/Project-/volcanic_province_median_age.csv'

_hawaii_dir    = f'{base_dir}/new (1)'
_kahoolawe_dir = f'{base_dir}/new'
_oahu_dir      = f'{base_dir}/oahu/new'

island_files = {
    'hawaii': {
        'slope':      f'{_hawaii_dir}/hawaii_slope_nans.tif',
        'erosion':    f'{_hawaii_dir}/hawaii_erosion_nans.tif',
        'dem':        f'{_hawaii_dir}/hawaii_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/hawaii_precip_net_accum.tif',
    },
    'kahoolawe': {
        'slope':      f'{_kahoolawe_dir}/kahoolawe_slope_nans.tif',
        'erosion':    f'{_kahoolawe_dir}/kahoolawe_erosion_nans.tif',
        'dem':        f'{_kahoolawe_dir}/kahoolawe_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/kahoolawe_precip_net_accum.tif',
    },
    'oahu': {
        'slope':      f'{_oahu_dir}/oahu_slope_nans.tif',
        'erosion':    f'{_oahu_dir}/oahu_erosion_nans.tif',
        'dem':        f'{_oahu_dir}/oahu_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/oahu_precip_net_accum.tif',
    },
    'kauai': {
        'slope':      f'{base_dir}/kauai/new/kauai_slope.tif',
        'erosion':    f'{base_dir}/kauai/new/kauai_erosion_nans.tif',
        'dem':        f'{base_dir}/kauai/new/kauai_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/kauai_precip_net_accum.tif',
    },
    'lanai': {
        'slope':      f'{base_dir}/lanai/new/lanai_slope.tif',
        'erosion':    f'{base_dir}/lanai/new/lanai_erosion_nans.tif',
        'dem':        f'{base_dir}/lanai/new/lanai_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/lanai_precip_net_accum.tif',
    },
    'molokai': {
        'slope':      f'{base_dir}/molokai/new/molokai_slope_nans.tif',
        'erosion':    f'{base_dir}/molokai/new/molokai_erosion_nans.tif',
        'dem':        f'{base_dir}/molokai/new/molokai_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/molokai_precip_net_accum.tif',
    },
    'maui': {
        'slope':      f'{base_dir}/maui/new/maui_slope_nans.tif',
        'erosion':    f'{base_dir}/maui/new/maui_erosion_nans.tif',
        'dem':        f'{base_dir}/maui/new/maui_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/maui_precip_net_accum.tif',
    },
}

ISLAND_COLORS = {
    'hawaii':    '#d62728',
    'kahoolawe': '#9467bd',
    'oahu':      '#2ca02c',
    'kauai':     '#1f77b4',
    'lanai':     '#8c564b',
    'molokai':   '#e377c2',
    'maui':      '#ff7f0e',
}

def get_island_QnetS_E(island):
    """Q (maxflux, used only for the same positivity/masking check as
    before) and its point locations come from maxflux_stream_points.py's
    cached output; slope, erosion, dem, and mass_flux are sampled at those
    same point locations. Mass_flux (net of groundwater recharge) is used
    as the discharge proxy in place of raw flow accumulation, same as
    recreation.py's process_island() Fit 6."""
    points_path = f'{OUTPUT_DIR}{island}/maxflux_stream_points.shp'
    if not os.path.exists(points_path):
        return None

    paths = island_files[island]
    pts = gpd.read_file(points_path)
    xs, ys = pts.geometry.x.values, pts.geometry.y.values
    Q = pts['maxflux'].values

    S = sample_raster_at_points(paths['slope'], xs, ys, pts.crs)
    E = sample_raster_at_points(paths['erosion'], xs, ys, pts.crs)
    dem = sample_raster_at_points(paths['dem'], xs, ys, pts.crs)
    # mass_flux is a whitebox_workflows d8_mass_flux output -- its CRS tag is
    # mislabeled EPSG:4269 (same wbw quirk documented in watershed.py's
    # _ensure_raster_crs), but its transform/grid is untouched and matches
    # the other (correctly-labeled) island rasters exactly. Pass crs=None so
    # sample_raster_at_points uses xs/ys directly instead of reprojecting
    # them into the bogus CRS.
    MF = sample_raster_at_points(paths['mass_flux'], xs, ys, crs=None)

    valid_mask = np.isfinite(Q) & np.isfinite(S) & np.isfinite(E) & (dem > 1)
    Q, S, E, MF = Q[valid_mask], S[valid_mask], E[valid_mask], MF[valid_mask]

    pos_mask = (Q > 0) & (S > 0) & (E >= 1)
    S_pos, E_pos, MF_pos = S[pos_mask], E[pos_mask], MF[pos_mask]

    mf_mask = np.isfinite(MF_pos) & (MF_pos > 0)
    Q_net = MF_pos[mf_mask]
    E_net = E_pos[mf_mask]
    S_net = S_pos[mf_mask]

    return Q_net * S_net, E_net

def main():
    median_age = pd.read_csv(MEDIAN_AGE_CSV, index_col=0)['median_age_yr']

    fig, ax = plt.subplots(figsize=(7, 6))

    all_QS, all_E_norm = [], []
    for island in island_files:
        island_label = island.capitalize()
        if island_label not in median_age.index:
            print(f"[{island}] no median age found, skipping")
            continue
        age = median_age[island_label]

        result = get_island_QnetS_E(island)
        if result is None:
            print(f"[{island}] no maxflux_stream_points.shp found "
                  f"(run maxflux_stream_points.py first), skipping")
            continue
        QS_net, E = result
        if len(QS_net) < 10:
            print(f"[{island}] too few valid mass-flux pixels ({len(QS_net)}), skipping")
            continue
        E_norm = E / age

        all_QS.append(QS_net)
        all_E_norm.append(E_norm)

        ax.scatter(QS_net, E_norm, s=3, alpha=0.05, edgecolors='none',
                   color=ISLAND_COLORS[island], label=island_label)
        print(f"[{island}] n={len(QS_net)}  median_age={age:,.0f} yr")

    all_QS = np.concatenate(all_QS)
    all_E_norm = np.concatenate(all_E_norm)

    log_QS = np.log10(all_QS)
    log_EN = np.log10(all_E_norm)
    m, b, r, _, _ = stats.linregress(log_QS, log_EN)
    r2 = r ** 2
    rho, _ = stats.spearmanr(all_QS, all_E_norm)

    x_fit = np.logspace(log_QS.min(), log_QS.max(), 200)
    ax.plot(x_fit, (10 ** b) * (x_fit ** m), color='black', linewidth=1.8,
             linestyle='--', label='combined OLS fit')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel(r'Net Stream Power Proxy ($Q_\mathrm{net} \times S$, '
                  r'Recharge Accounted)')
    ax.set_ylabel(r'Erosional Depth / Median Volcanic-Province Age  '
                  r'($E$ / Age)  [m/yr]')
    ax.set_title('Depth-Normalized-by-Age vs Net Stream Power '
                 '(Groundwater-Accounted), All Islands')

    textstr = (f"Spearman's rho = {rho:.4f}\n"
               f"Log-Log R2 = {r2:.4f}")
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))

    leg = ax.legend(fontsize=8, markerscale=3, loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    out_path = f'{output_dir}/all_islands_E_over_age_vs_QnetS_groundwater.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved {out_path}")

if __name__ == '__main__':
    main()
    