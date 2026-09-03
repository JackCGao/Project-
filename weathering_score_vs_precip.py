#!/usr/bin/env python3
"""
Chemical Weathering Score (see weathering_score_vs_age.py for the formula
and why freeiron/ptotal were dropped for lack of coverage), at stream
points across all islands, vs. local precipitation instead of substrate
age.

Reuses the mukey-level scoring pipeline from weathering_score_vs_age.py
(that module guards its own execution behind `if __name__ == '__main__'`,
so importing it here is side-effect-free) and the precipitation
resampling helper from clay_vs_precip.py, since the precipitation rasters
are on a coarser grid (~242 m) than the streams/DEM grid (~10 m) and need
reprojecting first.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from weathering_score_vs_age import (
    base_dir, output_dir, load_clean, load_weathering_gdf, build_weathering_kdtree,
)
from clay_vs_precip import resample_to_ref

_kahoolawe_dir = f'{base_dir}/new'
_oahu_dir      = f'{base_dir}/oahu/new'

island_files = {
    'kahoolawe': {
        'streams': f'{_kahoolawe_dir}/kahoolawe_streams_unweighted_albers.tif',
        'dem':     f'{_kahoolawe_dir}/kahoolawe_dem_enforced_qgis_albers.tif',
        'precip':  f'{_kahoolawe_dir}/kahoolawe_precip_qgis_albers.tif',
    },
    'oahu': {
        'streams': f'{_oahu_dir}/oahu_streams_unweighted_albers.tif',
        'dem':     f'{_oahu_dir}/oahu_dem_enforced_qgis_albers.tif',
        'precip':  f'{_oahu_dir}/oahu_precip_qgis_albers.tif',
    },
    'kauai': {
        'streams': f'{base_dir}/kauai/new/kauai_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/kauai/new/kauai_dem_enforced_qgis_albers.tif',
        'precip':  f'{base_dir}/kauai/new/kauai_precip_qgis_albers.tif',
    },
    'lanai': {
        'streams': f'{base_dir}/lanai/new/lanai_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/lanai/new/lanai_dem_enforced_qgis_albers.tif',
        'precip':  f'{base_dir}/lanai/new/lanai_precip_qgis_albers.tif',
    },
    'molokai': {
        'streams': f'{base_dir}/molokai/new/molokai_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/molokai/new/molokai_dem_enforced_qgis_albers.tif',
        'precip':  f'{base_dir}/molokai/new/molokai_precip_qgis_albers.tif',
    },
    'maui': {
        'streams': f'{base_dir}/maui/new/maui_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/maui/new/maui_dem_enforced_qgis_albers.tif',
        'precip':  f'{base_dir}/maui/new/maui_precip_qgis_albers.tif',
    },
}

ISLAND_COLORS = {
    'kahoolawe': '#9467bd',
    'oahu':      '#2ca02c',
    'kauai':     '#1f77b4',
    'lanai':     '#8c564b',
    'molokai':   '#e377c2',
    'maui':      '#ff7f0e',
}


def get_island_weathering_precip(island, tree, w_vals):
    paths = island_files[island]
    streams = load_clean(paths['streams'])
    dem     = load_clean(paths['dem'])

    with rasterio.open(paths['streams']) as ref_src:
        ref_transform, ref_crs, ref_shape = ref_src.transform, ref_src.crs, ref_src.shape

    precip = resample_to_ref(paths['precip'], ref_transform, ref_crs, ref_shape)

    on_stream = (streams == 1)
    valid_mask = on_stream & np.isfinite(dem) & (dem > 1) & np.isfinite(precip) & (precip > 0)

    rows, cols = np.where(valid_mask)
    xs, ys = rasterio.transform.xy(ref_transform, rows, cols)
    stream_points = np.column_stack([xs, ys])

    dist, idx = tree.query(stream_points, k=1)
    matched_w = w_vals[idx]
    precip_vals = precip[valid_mask]
    return matched_w, precip_vals, dist


def main():
    gdf = load_weathering_gdf()
    print(f"{len(gdf)} map-unit polygons carry a weathering score, "
          f"CRS={gdf.crs.name}")
    tree, w_vals = build_weathering_kdtree(gdf)

    fig, ax = plt.subplots(figsize=(7, 6))

    all_precip, all_w = [], []
    for island in island_files:
        w_matched, precip_vals, dist = get_island_weathering_precip(island, tree, w_vals)
        if len(w_matched) == 0:
            print(f"[{island}] no valid stream points, skipping")
            continue

        all_precip.append(precip_vals)
        all_w.append(w_matched)

        ax.scatter(precip_vals, w_matched, s=3, alpha=0.115, edgecolors='none',
                   color=ISLAND_COLORS[island], label=island.capitalize())
        print(f"[{island}] n={len(w_matched)}  "
              f"nearest-neighbor dist (m): mean={dist.mean():.1f} max={dist.max():.1f}  "
              f"W mean={w_matched.mean():.2f}  precip mean={precip_vals.mean():.1f} mm/yr")

    all_precip = np.concatenate(all_precip)
    all_w = np.concatenate(all_w)

    log_precip = np.log10(all_precip)
    m, b, r, _, _ = stats.linregress(log_precip, all_w)
    r2 = r ** 2
    rho, _ = stats.spearmanr(all_precip, all_w)

    print(f"\nAll islands pooled: Spearman's rho = {rho:.4f}, "
          f"R2 (W vs log10 precip) = {r2:.4f}, slope = {m:.4f}, n = {len(all_precip):,}")

    x_fit = np.logspace(log_precip.min(), log_precip.max(), 200)
    ax.plot(x_fit, m * np.log10(x_fit) + b, color='black', linewidth=1.8,
            linestyle='--', label='OLS fit (W vs log10 precip)')

    ax.set_xscale('log')
    ax.minorticks_on()
    ax.set_xlabel('Precipitation (mm/yr)')
    ax.set_ylabel('Chemical Weathering Score (W)')
    ax.set_title('Weathering Score vs. Precipitation at Stream Pixels, All Islands')

    textstr = (f"Spearman's rho = {rho:.4f}\n"
               f"R2 (W vs log10 precip) = {r2:.4f}\n"
               f"n = {len(all_precip):,}")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))

    leg = ax.legend(fontsize=8, markerscale=3, loc='lower right')
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    out_path = f'{output_dir}/all_islands_weathering_score_vs_precip.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
