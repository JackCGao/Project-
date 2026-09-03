#!/usr/bin/env python3
"""
Multivariate power-law regression at stream points, across all islands:

    E = k * W * Q^m * S^n

  E = erosional depth (m)
  W = chemical weathering index (see below), nearest-neighbor matched by
      map-unit centroid -- same matching approach as betterfit.py's clay%.
      No exponent is fit for W -- it enters linearly (exponent fixed at 1),
      so log(E/W) is what actually gets regressed below.
  Q = flow-accumulation / discharge proxy (d8maxflux_nans.tif, m^3/yr)
  S = slope (degrees)

Fit is done by ordinary least squares on the log-log-linearized form:

    log(E/W) = log(k) + m*log(Q) + n*log(S)

i.e. a multiple linear regression with predictors [log Q, log S], solved
via numpy.linalg.lstsq -- same method as betterfit.py, run once per island
and once pooling every island together.

Why W still needs rescaling even without a fitted exponent: the weathering
score from weathering_score_vs_age.py is W_raw = -z(pH) - z(CEC7) - z(ECEC),
a sum of signed z-scores that can be negative or zero, which breaks
log(E/W) directly. It's shifted to strictly positive here
(W = W_raw - min(W_raw) + 1, shift computed once across the pooled dataset
so it's the same constant everywhere).

Spearman's rank correlation reported below is between observed E/W and the
model's fitted E/W (in log space, though Spearman is rank-invariant to
that), i.e. a non-parametric companion to R^2 for the fit itself -- same
role Spearman plays for the bivariate fits elsewhere in this project.
"""

import rasterio
import numpy as np
from scipy import stats

from weathering_score_vs_age import (
    base_dir, load_clean, load_weathering_gdf, build_weathering_kdtree,
)

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


def get_island_data(island, tree, w_vals):
    """Masked by slope validity + the channel mask (*_streams_unweighted_
    albers.tif, same stream definition used in betterfit.py/comparegauges.py)
    -- no DEM mask -- with nearest-neighbor-matched weathering index."""
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
    W_pos = w_vals[idx]

    final = (np.isfinite(Q_pos) & (Q_pos > 0) &
             np.isfinite(slope_pos) & (slope_pos > 0) &
             np.isfinite(E_pos) & (E_pos > 0) &
             np.isfinite(W_pos) & (W_pos > 0))

    return Q_pos[final], slope_pos[final], E_pos[final], W_pos[final]


def fit_power_law(Q, S, E, W):
    """log(E) = log(k) + d*log(W) + m*log(Q) + n*log(S), via lstsq."""
    log_Q, log_S, log_E, log_W = np.log10(Q), np.log10(S), np.log10(E), np.log10(W)
    log_EW = log_E - log_W   # W's exponent fixed at 1, so it's divided out beforehand

    X = np.column_stack([np.ones_like(log_Q), log_Q, log_S])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_EW, rcond=None)
    log_k, m, n = coeffs

    log_EW_pred = X @ coeffs
    ss_res = np.sum((log_EW - log_EW_pred) ** 2)
    ss_tot = np.sum((log_EW - log_EW.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    rho, _ = stats.spearmanr(log_EW, log_EW_pred)

    return {'k': 10 ** log_k, 'm': m, 'n': n,
            'r2': r2, 'rho': rho, 'n_points': len(E)}


def fit_fixed_exponents(Q, S, E, W, m=0.5, n=1.0):
    """Same E = k*W*Q^m*S^n model, but m and n are fixed at the classic
    detachment-limited stream-power values (m=0.5, n=1) instead of fit by
    OLS. Only k is solved for, as the mean log-space residual (same method
    as recreation.py's k_sp for its Spearman-optimal-c fit)."""
    log_Q, log_S, log_E, log_W = np.log10(Q), np.log10(S), np.log10(E), np.log10(W)
    log_EW = log_E - log_W

    resid = log_EW - (m * log_Q + n * log_S)
    log_k = resid.mean()
    log_EW_pred = log_k + m * log_Q + n * log_S

    ss_res = np.sum((log_EW - log_EW_pred) ** 2)
    ss_tot = np.sum((log_EW - log_EW.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    rho, _ = stats.spearmanr(log_EW, log_EW_pred)

    return {'k': 10 ** log_k, 'm': m, 'n': n,
            'r2': r2, 'rho': rho, 'n_points': len(E)}


def print_fit(label, fit):
    print(f"[{label}]  E = {fit['k']:.4g} * W * "
          f"Q^{fit['m']:.4f} * S^{fit['n']:.4f}")
    print(f"    R^2 = {fit['r2']:.4f}   "
          f"Spearman's rho (obs vs. fitted E/W) = {fit['rho']:.4f}   "
          f"n = {fit['n_points']:,}\n")


def main():
    print("Loading weathering-index map units...")
    gdf = load_weathering_gdf()
    tree, w_raw_vals = build_weathering_kdtree(gdf)

    shift = -np.nanmin(w_raw_vals) + 1.0
    w_pos_vals = w_raw_vals + shift
    print(f"Weathering score range: [{w_raw_vals.min():.3f}, {w_raw_vals.max():.3f}] "
          f"-> shifted by +{shift:.3f} so W > 0 everywhere (needed for the log-log fit)\n")

    island_data = {}
    for island in island_files:
        Q, S, E, W = get_island_data(island, tree, w_pos_vals)
        if len(Q) < 10:
            print(f"[{island}] skipped -- only {len(Q)} valid points\n")
            continue
        island_data[island] = (Q, S, E, W)

    print("=== OLS-fit m, n ===\n")
    for island, (Q, S, E, W) in island_data.items():
        print_fit(island, fit_power_law(Q, S, E, W))

    all_Q = np.concatenate([d[0] for d in island_data.values()])
    all_S = np.concatenate([d[1] for d in island_data.values()])
    all_E = np.concatenate([d[2] for d in island_data.values()])
    all_W = np.concatenate([d[3] for d in island_data.values()])

    print_fit('all_islands', fit_power_law(all_Q, all_S, all_E, all_W))

    print("\n=== Fixed m=0.5, n=1 (classic detachment-limited stream power) ===\n")
    for island, (Q, S, E, W) in island_data.items():
        print_fit(island, fit_fixed_exponents(Q, S, E, W, m=0.5, n=1.0))

    print_fit('all_islands', fit_fixed_exponents(all_Q, all_S, all_E, all_W, m=0.5, n=1.0))

    print("\n=== Fixed m=1, n=1 (recreation.py's implicit raw Q*S proxy, no exponent fit) ===\n")
    for island, (Q, S, E, W) in island_data.items():
        print_fit(island, fit_fixed_exponents(Q, S, E, W, m=1.0, n=1.0))

    print_fit('all_islands', fit_fixed_exponents(all_Q, all_S, all_E, all_W, m=1.0, n=1.0))


if __name__ == '__main__':
    main()
