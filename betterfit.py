#!/usr/bin/env python3
"""
Multivariate power-law regression at stream points, across all islands:

    E = k * C^c * Q^m * s^n

  E = erosional depth (m)
  C = clay content (%), nearest-neighbor matched from gSSURGO_HI.gdb
      (same approach as clay_vs_erosion.py)
  Q = flow-accumulation / discharge proxy (d8maxflux_nans.tif, m^3/yr)
  s = slope (degrees)

Fit is done by ordinary least squares on the log-log-linearized form:

    log(E) = log(k) + c*log(C) + m*log(Q) + n*log(s)

i.e. a multiple linear regression with predictors [log C, log Q, log s],
solved via numpy.linalg.lstsq.
"""

import rasterio
import numpy as np
import pyogrio
from scipy.spatial import cKDTree

base_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao'
output_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Temp Output Placements'

GSSURGO_GDB = (f'{base_dir}/Jack, Ze-Wen summer project files/gSSURGO_HI.gdb')

_hawaii_dir    = f'{base_dir}/new (1)'
_kahoolawe_dir = f'{base_dir}/new'
_oahu_dir      = f'{base_dir}/oahu/new'

island_files = {
    'hawaii': {
        'flow_accum': f'{_hawaii_dir}/hawaii_d8maxflux_nans.tif',
        'slope':      f'{_hawaii_dir}/hawaii_slope_nans.tif',
        'erosion':    f'{_hawaii_dir}/hawaii_erosion_nans.tif',
        'streams':    f'{_hawaii_dir}/hawaii_streams_unweighted_albers.tif',
    },
    'kahoolawe': {
        'flow_accum': f'{_kahoolawe_dir}/kahoolawe_d8maxflux_nans.tif',
        'slope':      f'{_kahoolawe_dir}/kahoolawe_slope_nans.tif',
        'erosion':    f'{_kahoolawe_dir}/kahoolawe_erosion_nans.tif',
        'streams':    f'{_kahoolawe_dir}/kahoolawe_streams_unweighted_albers.tif',
    },
    'oahu': {
        'flow_accum': f'{_oahu_dir}/oahu_d8maxflux_nans.tif',
        'slope':      f'{_oahu_dir}/oahu_slope_nans.tif',
        'erosion':    f'{_oahu_dir}/oahu_erosion_nans.tif',
        'streams':    f'{_oahu_dir}/oahu_streams_unweighted_albers.tif',
    },
    'kauai': {
        'flow_accum': f'{base_dir}/kauai/new/kauai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/kauai/new/kauai_slope.tif',
        'erosion':    f'{base_dir}/kauai/new/kauai_erosion_nans.tif',
        'streams':    f'{base_dir}/kauai/new/kauai_streams_unweighted_albers.tif',
    },
    'lanai': {
        'flow_accum': f'{base_dir}/lanai/new/lanai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/lanai/new/lanai_slope.tif',
        'erosion':    f'{base_dir}/lanai/new/lanai_erosion_nans.tif',
        'streams':    f'{base_dir}/lanai/new/lanai_streams_unweighted_albers.tif',
    },
    'molokai': {
        'flow_accum': f'{base_dir}/molokai/new/molokai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/molokai/new/molokai_slope_nans.tif',
        'erosion':    f'{base_dir}/molokai/new/molokai_erosion_nans.tif',
        'streams':    f'{base_dir}/molokai/new/molokai_streams_unweighted_albers.tif',
    },
    'maui': {
        'flow_accum': f'{base_dir}/maui/new/maui_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/maui/new/maui_slope_nans.tif',
        'erosion':    f'{base_dir}/maui/new/maui_erosion_nans.tif',
        'streams':    f'{base_dir}/maui/new/maui_streams_unweighted_albers.tif',
    },
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

    mupoly = pyogrio.read_dataframe(GSSURGO_GDB, layer='MUPOLYGON', columns=['MUKEY'])
    mupoly = mupoly.merge(clay_by_mukey, left_on='MUKEY', right_index=True, how='inner')
    mupoly = mupoly[mupoly['clay_pct'] > 0].copy()

    return mupoly.to_crs('ESRI:102007')


def build_clay_kdtree(clay_gdf):
    centroids = clay_gdf.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])
    tree = cKDTree(coords)
    return tree, clay_gdf['clay_pct'].values


def get_island_data(island, tree, clay_vals_arr):
    """Masked by slope validity + the channel mask (*_streams_unweighted_
    albers.tif, same stream definition used in comparegauges.py) -- no DEM
    mask -- with nearest-neighbor-matched clay%."""
    paths = island_files[island]
    flow_accum = load_clean(paths['flow_accum'])
    slope_rast = load_clean(paths['slope'])
    erosion    = load_clean(paths['erosion'])
    streams    = load_clean(paths['streams'])

    on_stream = (streams == 1)
    valid_mask = (on_stream & np.isfinite(slope_rast) & (slope_rast > 0) &
                  np.isfinite(flow_accum) & np.isfinite(erosion))

    Q = flow_accum[valid_mask]
    slope_v = slope_rast[valid_mask]
    E = erosion[valid_mask]

    pos_mask = (Q > 0) & (E >= 1)
    Q_pos, slope_pos, E_pos = Q[pos_mask], slope_v[pos_mask], E[pos_mask]

    with rasterio.open(paths['erosion']) as src:
        transform = src.transform
    rows, cols = np.where(valid_mask)
    rows, cols = rows[pos_mask], cols[pos_mask]
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    _, idx = tree.query(np.column_stack([xs, ys]), k=1)
    C_pos = clay_vals_arr[idx]

    # Explicit final check: every variable must be finite and nonzero (clay
    # is already guaranteed >0 by load_clay_gdf's clay_pct > 0 filter before
    # the KD-tree is built, but re-checking here makes it self-evident
    # rather than relying on that upstream guarantee).
    final = (np.isfinite(Q_pos) & (Q_pos > 0) &
             np.isfinite(slope_pos) & (slope_pos > 0) &
             np.isfinite(E_pos) & (E_pos > 0) &
             np.isfinite(C_pos) & (C_pos > 0))

    return Q_pos[final], slope_pos[final], E_pos[final], C_pos[final]


def main():
    clay_gdf = load_clay_gdf()
    tree, clay_vals_arr = build_clay_kdtree(clay_gdf)
    print(f"Loaded clay data for {len(clay_gdf)} map-unit polygons (clay% > 0)")

    all_C, all_Q, all_s, all_E = [], [], [], []
    for island in island_files:
        Q, slope_v, E, C = get_island_data(island, tree, clay_vals_arr)
        all_C.append(C); all_Q.append(Q)
        all_s.append(slope_v); all_E.append(E)
        print(f"[{island}] n={len(Q)}")

    C = np.concatenate(all_C)          # clay %
    Q = np.concatenate(all_Q)          # discharge proxy (flow accumulation, m^3/yr)
    s = np.concatenate(all_s)          # slope (degrees)
    E = np.concatenate(all_E)

    # --- Multivariate OLS in log-log space ---
    # log(E) = log(k) + c*log(C) + m*log(Q) + n*log(s)
    log_C, log_Q, log_s, log_E = (
        np.log10(C), np.log10(Q), np.log10(s), np.log10(E)
    )

    X = np.column_stack([np.ones_like(log_C), log_C, log_Q, log_s])
    coeffs, residuals_ss, rank, sv = np.linalg.lstsq(X, log_E, rcond=None)
    log_k, c, m, n = coeffs
    k = 10 ** log_k

    log_E_pred = X @ coeffs
    resid = log_E - log_E_pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((log_E - log_E.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print("\n--- Fit: E = k * C(clay)^c * Q^m * s(slope)^n ---")
    print(f"  k = {k:.4g}")
    print(f"  c (clay% exponent)     = {c:.4f}")
    print(f"  m (discharge exponent) = {m:.4f}")
    print(f"  n (slope exponent)     = {n:.4f}")
    print(f"  R^2 (log-log)          = {r2:.4f}")
    print(f"  n points               = {len(E):,}")

    return {'k': k, 'c': c, 'm': m, 'n': n, 'r2': r2, 'n_points': len(E)}


if __name__ == '__main__':
    main()
