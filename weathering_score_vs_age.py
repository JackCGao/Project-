#!/usr/bin/env python3
"""
Chemical Weathering Score, at stream points across all islands, vs. each
point's representative volcano age (usefulfunctions.representative_age).

True CIA/CIW require Al2O3/CaO/Na2O/K2O whole-rock oxide geochemistry,
which does not exist anywhere in gSSURGO_HI.gdb or this project (checked
every table). This instead builds a weathering score from gSSURGO
chemistry fields that actually have usable coverage in Hawaii.

The originally proposed formula was
    W = z(freeiron) - z(CEC7) - z(ECEC) - z(pH) - z(ptotal)
but in gSSURGO_HI.gdb's chorizon table, freeiron_r is populated in only
0.7% of horizons and ptotal_r in 0% -- both unusable. Dropping them:

    W = -z(pH, 1:1 H2O) - z(CEC7) - z(ECEC)

All three decline as Hawaiian volcanic soils weather: base-cation leaching
lowers pH, and exchange capacity drops as clay mineralogy shifts from
high-activity 2:1 clays/allophane to low-activity kaolinite/gibbsite/Fe-Al
oxides. So higher W = more weathered, same sign convention as CIA/CIW.
Coverage of the three retained fields (Hawaii chorizon horizons):
    ph1to1h2o_r  83.9%
    cec7_r       75.8%
    ecec_r       51.7%
A map unit needs ALL THREE present (complete-case) to get a score.

The weighting/signs are a plain dict (WEATHERING_FIELDS) -- edit it to
add/drop/reweight variables without touching the rest of the pipeline.
"""

import rasterio
import numpy as np
import pandas as pd
import pyogrio
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree

from usefulfunctions import representative_age

base_dir   = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao'
output_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Temp Output Placements'

GSSURGO_GDB = (f'{base_dir}/Jack, Ze-Wen summer project files/gSSURGO_HI.gdb')

_hawaii_dir    = f'{base_dir}/new (1)'
_kahoolawe_dir = f'{base_dir}/new'
_oahu_dir      = f'{base_dir}/oahu/new'

island_files = {
    'hawaii': {
        'streams': f'{_hawaii_dir}/hawaii_streams_unweighted_albers.tif',
        'dem':     f'{_hawaii_dir}/hawaii_dem_enforced_qgis_albers.tif',
    },
    'kahoolawe': {
        'streams': f'{_kahoolawe_dir}/kahoolawe_streams_unweighted_albers.tif',
        'dem':     f'{_kahoolawe_dir}/kahoolawe_dem_enforced_qgis_albers.tif',
    },
    'oahu': {
        'streams': f'{_oahu_dir}/oahu_streams_unweighted_albers.tif',
        'dem':     f'{_oahu_dir}/oahu_dem_enforced_qgis_albers.tif',
    },
    'kauai': {
        'streams': f'{base_dir}/kauai/new/kauai_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/kauai/new/kauai_dem_enforced_qgis_albers.tif',
    },
    'lanai': {
        'streams': f'{base_dir}/lanai/new/lanai_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/lanai/new/lanai_dem_enforced_qgis_albers.tif',
    },
    'molokai': {
        'streams': f'{base_dir}/molokai/new/molokai_streams_unweighted_albers.tif',
        'dem':     f'{base_dir}/molokai/new/molokai_dem_enforced_qgis_albers.tif',
    },
    'maui': {
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

# field -> sign (+1 if it rises with weathering, -1 if it falls)
WEATHERING_FIELDS = {
    'ph1to1h2o_r': -1,
    'cec7_r':      -1,
    'ecec_r':      -1,
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


def build_mukey_weathering_table():
    """Weighted-average surface-horizon chemistry per MUKEY (weighted
    across components by comppct_r), for each field in WEATHERING_FIELDS."""
    fields = list(WEATHERING_FIELDS)
    comp = pyogrio.read_dataframe(GSSURGO_GDB, layer='component',
                                   columns=['mukey', 'cokey', 'comppct_r'],
                                   read_geometry=False)
    hor = pyogrio.read_dataframe(GSSURGO_GDB, layer='chorizon',
                                  columns=['cokey', 'hzdept_r'] + fields,
                                  read_geometry=False)

    surf = hor.loc[hor.groupby('cokey')['hzdept_r'].idxmin()]
    merged = surf.merge(comp, on='cokey', how='inner')

    def weighted_mean(g, col):
        sub = g.dropna(subset=[col])
        if sub.empty or sub['comppct_r'].sum() == 0:
            return np.nan
        return (sub[col] * sub['comppct_r']).sum() / sub['comppct_r'].sum()

    rows = []
    for mukey, g in merged.groupby('mukey'):
        row = {'mukey': mukey}
        for col in fields:
            row[col] = weighted_mean(g, col)
        rows.append(row)

    return pd.DataFrame(rows).set_index('mukey')


def compute_weathering_score(table):
    """Complete-case z-score combination: table -> Series of W, indexed by
    mukey, only for map units with every WEATHERING_FIELDS column present."""
    fields = list(WEATHERING_FIELDS)
    complete = table.dropna(subset=fields)
    z = (complete - complete.mean()) / complete.std()
    score = sum(WEATHERING_FIELDS[c] * z[c] for c in fields)
    return score.rename('weathering_score')


def load_weathering_gdf():
    table = build_mukey_weathering_table()
    complete_n = table.dropna(subset=list(WEATHERING_FIELDS)).shape[0]
    print(f"{len(table)} map units have >=1 weathering field; "
          f"{complete_n} have all {len(WEATHERING_FIELDS)} (complete-case, used below)")

    score = compute_weathering_score(table)

    mupoly = pyogrio.read_dataframe(GSSURGO_GDB, layer='MUPOLYGON', columns=['MUKEY'])
    mupoly = mupoly.merge(score, left_on='MUKEY', right_index=True, how='inner')
    return mupoly.to_crs('ESRI:102007')


def build_weathering_kdtree(gdf):
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])
    tree = cKDTree(coords)
    return tree, gdf['weathering_score'].values


def get_island_weathering_age(island, tree, w_vals):
    paths = island_files[island]
    streams = load_clean(paths['streams'])
    dem     = load_clean(paths['dem'])

    with rasterio.open(paths['dem']) as src:
        transform = src.transform
        age_rast = representative_age(src)

    on_stream = (streams == 1)
    valid_mask = (on_stream & np.isfinite(dem) & (dem > 1) &
                  np.isfinite(age_rast) & (age_rast > 0))

    rows, cols = np.where(valid_mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    stream_points = np.column_stack([xs, ys])

    dist, idx = tree.query(stream_points, k=1)
    matched_w = w_vals[idx]
    age_vals = age_rast[valid_mask]
    return matched_w, age_vals, dist


def main():
    gdf = load_weathering_gdf()
    print(f"{len(gdf)} map-unit polygons carry a weathering score, "
          f"CRS={gdf.crs.name}")
    tree, w_vals = build_weathering_kdtree(gdf)

    fig, ax = plt.subplots(figsize=(7, 6))

    all_age, all_w = [], []
    for island in island_files:
        w_vals_matched, age_vals, dist = get_island_weathering_age(island, tree, w_vals)
        if len(w_vals_matched) == 0:
            print(f"[{island}] no valid stream points, skipping")
            continue

        all_age.append(age_vals)
        all_w.append(w_vals_matched)

        ax.scatter(age_vals, w_vals_matched, s=3, alpha=0.115, edgecolors='none',
                   color=ISLAND_COLORS[island], label=island.capitalize())
        print(f"[{island}] n={len(w_vals_matched)}  "
              f"nearest-neighbor dist (m): mean={dist.mean():.1f} max={dist.max():.1f}  "
              f"W mean={w_vals_matched.mean():.2f}  age mean={age_vals.mean():,.0f} yr")

    all_age = np.concatenate(all_age)
    all_w = np.concatenate(all_w)

    log_age = np.log10(all_age)
    m, b, r, _, _ = stats.linregress(log_age, all_w)
    r2 = r ** 2
    rho, _ = stats.spearmanr(all_age, all_w)

    print(f"\nAll islands pooled: Spearman's rho = {rho:.4f}, "
          f"R2 (W vs log10 age) = {r2:.4f}, slope = {m:.4f}, n = {len(all_age):,}")

    x_fit = np.logspace(log_age.min(), log_age.max(), 200)
    ax.plot(x_fit, m * np.log10(x_fit) + b, color='black', linewidth=1.8,
            linestyle='--', label='OLS fit (W vs log10 age)')

    ax.set_xscale('log')
    ax.minorticks_on()
    ax.set_xlabel('Representative Volcano Age (yr)')
    ax.set_ylabel('Chemical Weathering Score (W)')
    ax.set_title('Weathering Score vs. Substrate Age at Stream Pixels, All Islands')

    textstr = (f"Spearman's rho = {rho:.4f}\n"
               f"R2 (W vs log10 age) = {r2:.4f}\n"
               f"n = {len(all_age):,}")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))

    leg = ax.legend(fontsize=8, markerscale=3, loc='lower right')
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    out_path = f'{output_dir}/all_islands_weathering_score_vs_age.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
