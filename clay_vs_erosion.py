#!/usr/bin/env python3
"""
Combined plot: Clay Content (%) vs Erosional Depth (E), at stream points
only, for all islands overlaid in one log-log scatter with an OLS fit.

Clay content comes from gSSURGO_HI.gdb: for each map unit (MUKEY), the
representative surface-horizon clay% (claytotal_r, topmost hzdept_r per
component) is averaged across components weighted by comppct_r. Each
stream pixel (same *_streams_unweighted_albers.tif mask used by
comparegauges.py -- restricting to the actual channel network rather than
all land pixels) is converted to a point and matched to its NEAREST clay
map-unit polygon (by centroid distance via a KD-tree), rather than
rasterizing/painting clay values onto the grid by polygon containment.
"""

import rasterio
import numpy as np
import pyogrio
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree

base_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao'
output_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Temp Output Placements'

GSSURGO_GDB = (f'{base_dir}/Jack, Ze-Wen summer project files/gSSURGO_HI.gdb')

_hawaii_dir    = f'{base_dir}/new (1)'
_kahoolawe_dir = f'{base_dir}/new'
_oahu_dir      = f'{base_dir}/oahu/new'

island_files = {
    'hawaii': {
        'erosion': f'{_hawaii_dir}/hawaii_erosion_nans.tif',
        'streams': f'{_hawaii_dir}/hawaii_streams_unweighted_albers.tif',
        'dem':     f'{_hawaii_dir}/hawaii_dem_enforced_qgis_albers.tif',
    },
    'kahoolawe': {
        'erosion': f'{_kahoolawe_dir}/kahoolawe_erosion_nans.tif',
        'streams': f'{_kahoolawe_dir}/kahoolawe_streams_unweighted_albers.tif',
        'dem':     f'{_kahoolawe_dir}/kahoolawe_dem_enforced_qgis_albers.tif',
    },
    'oahu': {
        'erosion': f'{_oahu_dir}/oahu_erosion_nans.tif',
        'streams': f'{_oahu_dir}/oahu_streams_unweighted_albers.tif',
        'dem':     f'{_oahu_dir}/oahu_dem_enforced_qgis_albers.tif',
    },
    'kauai': {
        'erosion': f'{base_dir}/kauai/new/kauai_erosion_nans.tif',
        'streams': f'{base_dir}/kauai/new/kauai_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/kauai/new/kauai_dem_enforced_qgis_albers.tif',
    },
    'lanai': {
        'erosion': f'{base_dir}/lanai/new/lanai_erosion_nans.tif',
        'streams': f'{base_dir}/lanai/new/lanai_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/lanai/new/lanai_dem_enforced_qgis_albers.tif',
    },
    'molokai': {
        'erosion': f'{base_dir}/molokai/new/molokai_erosion_nans.tif',
        'streams': f'{base_dir}/molokai/new/molokai_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/molokai/new/molokai_dem_enforced_qgis_albers.tif',
    },
    'maui': {
        'erosion': f'{base_dir}/maui/new/maui_erosion_nans.tif',
        'streams': f'{base_dir}/maui/new/maui_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/maui/new/maui_dem_enforced_qgis_albers.tif',
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


def build_mukey_clay_table():
    """Weighted-average surface-horizon clay% (claytotal_r) per MUKEY,
    weighted across components by comppct_r."""
    comp = pyogrio.read_dataframe(GSSURGO_GDB, layer='component',
                                   columns=['mukey', 'cokey', 'comppct_r'],
                                   read_geometry=False)
    hor = pyogrio.read_dataframe(GSSURGO_GDB, layer='chorizon',
                                  columns=['cokey', 'hzdept_r', 'claytotal_r'],
                                  read_geometry=False)

    surf = hor.loc[hor.groupby('cokey')['hzdept_r'].idxmin()]
    surf = surf.dropna(subset=['claytotal_r'])

    merged = surf.merge(comp, on='cokey', how='inner')
    weighted = (
        merged.groupby('mukey')
        .apply(lambda g: (g['claytotal_r'] * g['comppct_r']).sum() / g['comppct_r'].sum())
        .rename('clay_pct')
    )
    return weighted


def load_clay_gdf():
    """MUPOLYGON joined to per-mukey clay%, reprojected once to the
    islands' common CRS (ESRI:102007)."""
    clay_by_mukey = build_mukey_clay_table()

    mupoly = pyogrio.read_dataframe(GSSURGO_GDB, layer='MUPOLYGON',
                                     columns=['MUKEY'])
    mupoly = mupoly.merge(clay_by_mukey, left_on='MUKEY', right_index=True, how='inner')
    mupoly = mupoly[mupoly['clay_pct'] > 0].copy()

    return mupoly.to_crs('ESRI:102007')


def build_clay_kdtree(clay_gdf):
    """KD-tree over clay map-unit polygon centroids, for nearest-neighbor
    matching of stream points to a clay% value."""
    centroids = clay_gdf.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])
    tree = cKDTree(coords)
    clay_vals = clay_gdf['clay_pct'].values
    return tree, clay_vals


def get_island_clay_erosion(island, tree, clay_vals):
    paths = island_files[island]
    erosion = load_clean(paths['erosion'])
    streams = load_clean(paths['streams'])
    dem     = load_clean(paths['dem'])

    on_stream = (streams == 1)
    valid_mask = on_stream & np.isfinite(erosion) & (erosion >= 1) & (dem > 1)

    with rasterio.open(paths['erosion']) as src:
        transform = src.transform

    rows, cols = np.where(valid_mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    stream_points = np.column_stack([xs, ys])

    dist, idx = tree.query(stream_points, k=1)
    matched_clay = clay_vals[idx]

    erosion_vals = erosion[valid_mask]
    return matched_clay, erosion_vals, dist


def main():
    clay_gdf = load_clay_gdf()
    print(f"Loaded clay data for {len(clay_gdf)} map-unit polygons "
          f"(clay% > 0), CRS={clay_gdf.crs.name}")
    tree, clay_vals_arr = build_clay_kdtree(clay_gdf)

    fig, ax = plt.subplots(figsize=(7, 6))

    all_erosion, all_clay = [], []
    for island in island_files:
        clay_vals, erosion_vals, dist = get_island_clay_erosion(island, tree, clay_vals_arr)
        if len(clay_vals) == 0:
            print(f"[{island}] no valid stream points, skipping")
            continue

        all_erosion.append(erosion_vals)
        all_clay.append(clay_vals)

        ax.scatter(erosion_vals, clay_vals, s=3, alpha=0.115, edgecolors='none',
                   color=ISLAND_COLORS[island], label=island.capitalize())
        print(f"[{island}] n={len(clay_vals)}  "
              f"nearest-neighbor dist (m): mean={dist.mean():.1f} max={dist.max():.1f}  "
              f"clay% mean={clay_vals.mean():.1f}  erosion mean={erosion_vals.mean():.2f} m")

    all_erosion = np.concatenate(all_erosion)
    all_clay = np.concatenate(all_clay)

    log_x = np.log10(all_erosion)
    log_y = np.log10(all_clay)
    m, b, r, _, _ = stats.linregress(log_x, log_y)
    r2 = r ** 2
    rho, _ = stats.spearmanr(all_erosion, all_clay)

    x_fit = np.logspace(log_x.min(), log_x.max(), 200)
    ax.plot(x_fit, (10 ** b) * (x_fit ** m), color='black', linewidth=1.8,
            linestyle='--', label='log-log OLS fit')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel('Erosional Depth (E) [m]')
    ax.set_ylabel('Clay Content (%)')
    ax.set_title('Clay Content vs Erosional Depth at Stream Pixels, All Islands')

    textstr = (f"Spearman's rho = {rho:.4f}\n"
               f"Log-Log R2 = {r2:.4f}\n"
               f"n = {len(all_erosion):,}")
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))

    leg = ax.legend(fontsize=8, markerscale=3, loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    out_path = f'{output_dir}/all_islands_clay_vs_erosion.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
