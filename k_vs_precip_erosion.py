#!/usr/bin/env python3
"""
Local erodibility (k) vs. local precipitation and erosion, across all islands.

Stream power law:  d = k * Q^m * S^n
  d = erosional depth (m)
  Q = discharge proxy (flow accumulation, *_d8maxflux_nans.tif, m^3/yr)
  S = slope (degrees)
  k = erodibility coefficient

Two-stage fit:
  1. m and n are fit ONCE, pooling every island, via OLS on the
     log-log-linearized form log10(d) = log10(k) + m*log10(Q) + n*log10(S)
     (same lstsq approach as betterfit.py, just without the clay term).
  2. Holding those global m, n fixed, k is solved per-pixel:
         k = d / (Q^m * S^n)
     This local k is then the response in a multiple linear regression
     (a plane fit) against local precipitation and local erosion:
         k = b0 + b1*precip + b2*erosion

Note: erosion (d) is used both to solve for local k in stage 2 and as a
predictor in the stage-3 regression, so a positive erosion coefficient is
expected largely by construction (k scales directly with d). The
precipitation coefficient is the one that speaks to whether rainfall drives
erodibility independent of the Q/S stream-power scaling already captured by
m, n.
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rasterio')

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
import geopandas as gpd
import pandas as pd
from rasterio.features import rasterize as rio_rasterize
import os

from usefulfunctions import load_clean

base_dir   = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao'
output_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Temp Output Placements'

_recharge_base = (f'{base_dir}/Jack, Ze-Wen summer project files/Groundwater Recharge Files')
RECHARGE_SHPS = {
    'kauai':   f'{_recharge_base}/Kauai Water Budget Components 2020/Kauai_water_budget_components_subarea_inches.shp',
    'lanai':   f'{_recharge_base}/Lanai Water Budget Components 2020/Lanai_water_budget_components_subarea_inches.shp',
    'molokai': f'{_recharge_base}/Molokai Water Budget Components 2020/Molokai_water_budget_components_subarea_inches.shp',
    'maui':    f'{_recharge_base}/Maui Water Budget Components 2020/Maui_water_budget_components_subarea_inches.shp',
    'oahu':    f'{_recharge_base}/Oahu Water Budget Components 2020/Oahu_water_budget_components_subarea_inches.shp',
    'hawaii':  [f'{_recharge_base}/Hawaii Water Budget Components Part 1/Hawaii_water_budget_components_subarea_inches_P1.shp',
                f'{_recharge_base}/Hawaii Water Budget Components Part 2/Hawaii_water_budget_components_subarea_inches_P2.shp'],
}
PRECIP_FIELD = 'S1_Rain'   # rainfall field in the water-budget shapefiles (in/yr)

_recharge_gdfs = {}

def _load_recharge_shp(island):
    if island not in _recharge_gdfs:
        paths = RECHARGE_SHPS[island]
        if isinstance(paths, str):
            paths = [paths]

        parts = []
        for path in paths:
            gdf = gpd.read_file(path)
            cols = ['geometry']
            if PRECIP_FIELD in gdf.columns:
                cols.append(PRECIP_FIELD)
            parts.append(gdf[cols])

        if len(parts) > 1:
            crs_set = {p.crs for p in parts}
            if len(crs_set) > 1:
                parts = [p.to_crs(parts[0].crs) for p in parts]
            combined = pd.concat(parts, ignore_index=True)
            combined = gpd.GeoDataFrame(combined, geometry='geometry', crs=parts[0].crs)
        else:
            combined = parts[0]

        _recharge_gdfs[island] = combined.copy()
    return _recharge_gdfs[island]

def rasterize_precipitation(ref_path, island):
    """Rasterize S1_Rain (rainfall, in/yr) onto the reference raster grid."""
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs       = src.crs
        width     = src.width
        height    = src.height

    if island not in RECHARGE_SHPS:
        return np.full((height, width), np.nan)

    gdf = _load_recharge_shp(island).to_crs(crs)
    if PRECIP_FIELD not in gdf.columns:
        print(f"[{island}] precipitation field '{PRECIP_FIELD}' not found; "
              f"available: {list(gdf.columns)}")
        return np.full((height, width), np.nan)

    shapes = (
        (geom, val)
        for geom, val in zip(gdf.geometry, gdf[PRECIP_FIELD])
        if geom is not None and not geom.is_empty
    )
    return rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype=np.float64,
    )

_hawaii_dir    = f'{base_dir}/new (1)'
_kahoolawe_dir = f'{base_dir}/new'
_oahu_dir      = f'{base_dir}/oahu/new'

island_files = {
    'hawaii': {
        'flow_accum': f'{_hawaii_dir}/hawaii_d8maxflux_nans.tif',
        'slope':      f'{_hawaii_dir}/hawaii_slope_nans.tif',
        'erosion':    f'{_hawaii_dir}/hawaii_erosion_nans.tif',
    },
    'kahoolawe': {
        'flow_accum': f'{_kahoolawe_dir}/kahoolawe_d8maxflux_nans.tif',
        'slope':      f'{_kahoolawe_dir}/kahoolawe_slope_nans.tif',
        'erosion':    f'{_kahoolawe_dir}/kahoolawe_erosion_nans.tif',
    },
    'oahu': {
        'flow_accum': f'{_oahu_dir}/oahu_d8maxflux_nans.tif',
        'slope':      f'{_oahu_dir}/oahu_slope_nans.tif',
        'erosion':    f'{_oahu_dir}/oahu_erosion_nans.tif',
    },
    'kauai': {
        'flow_accum': f'{base_dir}/kauai/new/kauai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/kauai/new/kauai_slope.tif',
        'erosion':    f'{base_dir}/kauai/new/kauai_erosion_nans.tif',
    },
    'lanai': {
        'flow_accum': f'{base_dir}/lanai/new/lanai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/lanai/new/lanai_slope.tif',
        'erosion':    f'{base_dir}/lanai/new/lanai_erosion_nans.tif',
    },
    'molokai': {
        'flow_accum': f'{base_dir}/molokai/new/molokai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/molokai/new/molokai_slope_nans.tif',
        'erosion':    f'{base_dir}/molokai/new/molokai_erosion_nans.tif',
    },
    'maui': {
        'flow_accum': f'{base_dir}/maui/new/maui_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/maui/new/maui_slope_nans.tif',
        'erosion':    f'{base_dir}/maui/new/maui_erosion_nans.tif',
    },
}
islands = list(island_files.keys())


def load_island_data(island):
    """Q (discharge proxy), S (slope), E (erosion), P (local precipitation),
    co-masked to this island's valid, positive-valued pixels."""
    paths = island_files[island]
    Q_rast = load_clean(paths['flow_accum'])
    S_rast = load_clean(paths['slope'])
    E_rast = load_clean(paths['erosion'])
    P_rast = rasterize_precipitation(paths['flow_accum'], island)

    valid_mask = (np.isfinite(Q_rast) & np.isfinite(S_rast) &
                  np.isfinite(E_rast) & np.isfinite(P_rast))

    Q = Q_rast[valid_mask]
    S = S_rast[valid_mask]
    E = E_rast[valid_mask]
    P = P_rast[valid_mask]

    pos_mask = (Q > 0) & (S > 0) & (E >= 1) & (P > 0)
    return {'island': island, 'Q': Q[pos_mask], 'S': S[pos_mask],
            'E': E[pos_mask], 'P': P[pos_mask]}


def fit_global_stream_power_exponents(island_data):
    """Pool every island and fit log10(E) = log10(k) + m*log10(Q) + n*log10(S)
    via OLS (numpy.linalg.lstsq), same method as betterfit.py."""
    all_Q = np.concatenate([d['Q'] for d in island_data])
    all_S = np.concatenate([d['S'] for d in island_data])
    all_E = np.concatenate([d['E'] for d in island_data])

    log_Q, log_S, log_E = np.log10(all_Q), np.log10(all_S), np.log10(all_E)

    X = np.column_stack([np.ones_like(log_Q), log_Q, log_S])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_E, rcond=None)
    log_k, m, n = coeffs

    log_E_pred = X @ coeffs
    ss_res = np.sum((log_E - log_E_pred) ** 2)
    ss_tot = np.sum((log_E - log_E.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {'k_global': 10 ** log_k, 'm': m, 'n': n, 'r2': r2, 'n_points': len(all_E)}


def compute_local_k(data, m, n):
    """Per-pixel erodibility coefficient k = E / (Q^m * S^n)."""
    return data['E'] / (data['Q'] ** m * data['S'] ** n)


def fit_k_vs_precip_erosion(P, E, k):
    """Multiple linear regression (plane fit): k = b0 + b1*P + b2*E."""
    X = np.column_stack([np.ones_like(P), P, E])
    coeffs, _, _, _ = np.linalg.lstsq(X, k, rcond=None)
    b0, b1, b2 = coeffs

    k_pred = X @ coeffs
    ss_res = np.sum((k - k_pred) ** 2)
    ss_tot = np.sum((k - k.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {'intercept': b0, 'precip_coef': b1, 'erosion_coef': b2, 'r2': r2}


def plot_k_plane(label, P, E, k, fit):
    """3D scatter of local k above the (precipitation, erosion) plane, with
    the fitted regression plane overlaid."""
    idx = np.arange(len(P))
    if len(idx) > 20_000:
        idx = np.random.default_rng(0).choice(len(P), 20_000, replace=False)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[idx], E[idx], k[idx], s=3, alpha=0.25, color='#3a6ea5', edgecolors='none')

    p_grid = np.linspace(P.min(), P.max(), 25)
    e_grid = np.linspace(E.min(), E.max(), 25)
    PG, EG = np.meshgrid(p_grid, e_grid)
    KG = fit['intercept'] + fit['precip_coef'] * PG + fit['erosion_coef'] * EG
    ax.plot_surface(PG, EG, KG, color='#e06b3a', alpha=0.35, linewidth=0)

    ax.set_xlabel('Local Precipitation (S1_Rain, in/yr)')
    ax.set_ylabel('Local Erosional Depth (E) [m]')
    ax.set_zlabel('Local Erodibility k')
    ax.set_title(label.capitalize())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{label}_k_vs_precip_erosion.png'), dpi=200)
    plt.close(fig)


def main():
    print("Loading per-island Q/S/E/precipitation data...")
    island_data = [load_island_data(isl) for isl in islands]

    print("\nFitting global stream-power exponents  E = k * Q^m * S^n ...")
    global_fit = fit_global_stream_power_exponents(island_data)
    m, n = global_fit['m'], global_fit['n']
    print(f"  m (Q exponent) = {m:.4f}")
    print(f"  n (S exponent) = {n:.4f}")
    print(f"  global k       = {global_fit['k_global']:.4g}")
    print(f"  R^2 (log-log)  = {global_fit['r2']:.4f}")
    print(f"  n points       = {global_fit['n_points']:,}")

    results = []
    all_P, all_E, all_k = [], [], []
    for data in island_data:
        k_local = compute_local_k(data, m, n)
        valid = np.isfinite(k_local) & (k_local > 0)
        P, E, k_local = data['P'][valid], data['E'][valid], k_local[valid]

        if len(k_local) < 10:
            print(f"[{data['island']}] skipped -- only {len(k_local)} valid "
                  f"points (likely no precipitation data for this island)")
            continue

        fit = fit_k_vs_precip_erosion(P, E, k_local)
        plot_k_plane(data['island'], P, E, k_local, fit)

        print(f"[{data['island']}] k = {fit['intercept']:.4g} "
              f"+ {fit['precip_coef']:.4g}*precip + {fit['erosion_coef']:.4g}*erosion, "
              f"R^2 = {fit['r2']:.4f}  (n = {len(k_local):,})")

        results.append({'island': data['island'], **fit, 'n': len(k_local)})
        all_P.append(P); all_E.append(E); all_k.append(k_local)

    all_P = np.concatenate(all_P)
    all_E = np.concatenate(all_E)
    all_k = np.concatenate(all_k)
    pooled_fit = fit_k_vs_precip_erosion(all_P, all_E, all_k)
    plot_k_plane('all_islands', all_P, all_E, all_k, pooled_fit)
    print(f"\n[all_islands] k = {pooled_fit['intercept']:.4g} "
          f"+ {pooled_fit['precip_coef']:.4g}*precip + {pooled_fit['erosion_coef']:.4g}*erosion, "
          f"R^2 = {pooled_fit['r2']:.4f}  (n = {len(all_k):,})")

    return {'global_stream_power_fit': global_fit, 'per_island': results, 'pooled': pooled_fit}


if __name__ == '__main__':
    main()
