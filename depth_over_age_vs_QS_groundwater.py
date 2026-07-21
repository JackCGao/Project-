#!/usr/bin/env python3
"""
Combined plot: (Erosional Depth / Median Volcanic-Province Age) vs Net
Stream Power Proxy (Q_net x S), for all islands overlaid in one log-log
scatter -- groundwater-aware version of depth_over_age_vs_QS.py.

Q_net comes from each island's mass_flux raster (WhiteboxTools d8_mass_flux
output: precipitation routed downstream minus groundwater recharge
absorption -- see recreation.py's compute_and_save_net_precip_accum), so
this factors in groundwater recharge instead of using raw flow
accumulation.
"""

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

base_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao'
output_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Temp Output Placements'

MEDIAN_AGE_CSV = '/Users/jackgao/Summer Work 2026/Project-/volcanic_province_median_age.csv'

_hawaii_dir    = f'{base_dir}/new (1)'
_kahoolawe_dir = f'{base_dir}/new'
_oahu_dir      = f'{base_dir}/oahu/new'

island_files = {
    'hawaii': {
        'flow_accum': f'{_hawaii_dir}/hawaii_d8maxflux_nans.tif',
        'slope':      f'{_hawaii_dir}/hawaii_slope_nans.tif',
        'erosion':    f'{_hawaii_dir}/hawaii_erosion_nans.tif',
        'dem':        f'{_hawaii_dir}/hawaii_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/hawaii_precip_net_accum.tif',
    },
    'kahoolawe': {
        'flow_accum': f'{_kahoolawe_dir}/kahoolawe_d8maxflux_nans.tif',
        'slope':      f'{_kahoolawe_dir}/kahoolawe_slope_nans.tif',
        'erosion':    f'{_kahoolawe_dir}/kahoolawe_erosion_nans.tif',
        'dem':        f'{_kahoolawe_dir}/kahoolawe_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/kahoolawe_precip_net_accum.tif',
    },
    'oahu': {
        'flow_accum': f'{_oahu_dir}/oahu_d8maxflux_nans.tif',
        'slope':      f'{_oahu_dir}/oahu_slope_nans.tif',
        'erosion':    f'{_oahu_dir}/oahu_erosion_nans.tif',
        'dem':        f'{_oahu_dir}/oahu_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/oahu_precip_net_accum.tif',
    },
    'kauai': {
        'flow_accum': f'{base_dir}/kauai/new/kauai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/kauai/new/kauai_slope.tif',
        'erosion':    f'{base_dir}/kauai/new/kauai_erosion_nans.tif',
        'dem':        f'{base_dir}/kauai/new/kauai_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/kauai_precip_net_accum.tif',
    },
    'lanai': {
        'flow_accum': f'{base_dir}/lanai/new/lanai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/lanai/new/lanai_slope.tif',
        'erosion':    f'{base_dir}/lanai/new/lanai_erosion_nans.tif',
        'dem':        f'{base_dir}/lanai/new/lanai_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/lanai_precip_net_accum.tif',
    },
    'molokai': {
        'flow_accum': f'{base_dir}/molokai/new/molokai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/molokai/new/molokai_slope_nans.tif',
        'erosion':    f'{base_dir}/molokai/new/molokai_erosion_nans.tif',
        'dem':        f'{base_dir}/molokai/new/molokai_dem_enforced_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/molokai_precip_net_accum.tif',
    },
    'maui': {
        'flow_accum': f'{base_dir}/maui/new/maui_d8maxflux_nans.tif',
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


def load_clean(path):
    with rasterio.open(path) as src:
        arr = src.read([1])[0].astype(np.float64, copy=False)
        nd = src.nodata
        if nd is not None:
            if np.isnan(nd):
                arr[~np.isfinite(arr)] = np.nan
            else:
                arr[np.isclose(arr, nd, rtol=0, atol=1e-6)] = np.nan
        return arr


def get_island_QnetS_E(island):
    """Same masking as recreation.py's process_island() Fit 6 (mass_flux
    used as the discharge proxy in place of raw flow accumulation)."""
    paths = island_files[island]
    flow_accum = load_clean(paths['flow_accum'])
    slope_rast = load_clean(paths['slope'])
    erosion    = load_clean(paths['erosion'])
    dem        = load_clean(paths['dem'])
    mass_flux  = load_clean(paths['mass_flux'])

    valid_mask = (np.isfinite(flow_accum) & np.isfinite(slope_rast) &
                  np.isfinite(erosion) & (dem > 1))
    final_valid = valid_mask & np.isfinite(flow_accum * slope_rast)

    S = slope_rast[final_valid]
    E = erosion[final_valid]
    Q = flow_accum[final_valid]
    MF = mass_flux[final_valid]

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

        QS_net, E = get_island_QnetS_E(island)
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
