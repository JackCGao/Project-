#!/usr/bin/env python3
"""
Combined plot: (Erosional Depth / Representative Volcano Age) vs Stream
Power Proxy (Q x S), for all islands overlaid in one log-log scatter.

Q comes from maxflux_stream_points.py's per-island output
(watershed_outputs/{island}/maxflux_stream_points.shp) -- D8 max-flux
sampled at stream-pixel points that are both on the island's own stream
mask and within 50 px of the statewide Streams_reprojected.shp hydrography
layer -- rather than loading the raw flow_accum raster and masking it here.

Normalizing E by each pixel's representative volcano age (usefulfunctions.
representative_age / nearest_volcano_age, per-volcano-region average from
ageprocessing.py) puts pixels from volcanoes of very different ages on a
comparable incision-rate-like footing before comparing against the Q*S
stream power proxy -- same age method as recreation.py's
process_island_erosion_rate.
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

from usefulfunctions import sample_raster_at_points, nearest_volcano_age
from maxflux_stream_points import OUTPUT_DIR

# Derived from this file's own location rather than hardcoded, since the
# project folder has been renamed before (it used to be "Summer Work 2026",
# with spaces) and a stale hardcoded path here silently no-ops every path
# built from it.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = f'{base_dir}/Temp Output Placements'

# Per-island slope/erosion rasters live under Island Data/ -- the project's
# earlier flat-per-island-folder layout was reorganized under this common
# parent (same fix as recreation.py/maxflux_stream_points.py).
island_data_dir = f'{base_dir}/Island Data'

_hawaii_dir    = f'{island_data_dir}/hawaii'
_kahoolawe_dir = f'{island_data_dir}/kahoolawe'
_oahu_dir      = f'{island_data_dir}/oahu/new'

island_files = {
    'hawaii': {
        'slope':   f'{_hawaii_dir}/hawaii_slope_nans.tif',
        'erosion': f'{_hawaii_dir}/hawaii_erosion_nans.tif',
    },
    'kahoolawe': {
        'slope':   f'{_kahoolawe_dir}/kahoolawe_slope_nans.tif',
        'erosion': f'{_kahoolawe_dir}/kahoolawe_erosion_nans.tif',
    },
    'oahu': {
        'slope':   f'{_oahu_dir}/oahu_slope_nans.tif',
        'erosion': f'{_oahu_dir}/oahu_erosion_nans.tif',
    },
    'kauai': {
        'slope':   f'{island_data_dir}/kauai/new/kauai_slope.tif',
        'erosion': f'{island_data_dir}/kauai/new/kauai_erosion_nans.tif',
    },
    'lanai': {
        'slope':   f'{island_data_dir}/lanai/new/lanai_slope.tif',
        'erosion': f'{island_data_dir}/lanai/new/lanai_erosion_nans.tif',
    },
    'molokai': {
        'slope':   f'{island_data_dir}/molokai/new/molokai_slope_nans.tif',
        'erosion': f'{island_data_dir}/molokai/new/molokai_erosion_nans.tif',
    },
    'maui': {
        'slope':   f'{island_data_dir}/maui/new/maui_slope_nans.tif',
        'erosion': f'{island_data_dir}/maui/new/maui_erosion_nans.tif',
    },
}

def get_island_QS_E(island):
    """Q (maxflux) and its point locations come from maxflux_stream_points.py's
    cached output; S and E are sampled at those same point locations, and
    age is looked up at those points via nearest_volcano_age (the point-level
    core of usefulfunctions.representative_age)."""
    points_path = f'{OUTPUT_DIR}{island}/maxflux_stream_points.shp'
    if not os.path.exists(points_path):
        return None

    paths = island_files[island]
    pts = gpd.read_file(points_path)
    xs, ys = pts.geometry.x.values, pts.geometry.y.values
    Q = pts['maxflux'].values

    S = sample_raster_at_points(paths['slope'], xs, ys, pts.crs)
    E = sample_raster_at_points(paths['erosion'], xs, ys, pts.crs)

    valid_mask = np.isfinite(Q) & np.isfinite(S) & np.isfinite(E)
    Q, S, E, xs, ys = Q[valid_mask], S[valid_mask], E[valid_mask], xs[valid_mask], ys[valid_mask]

    pos_mask = (Q > 0) & (S > 0) & (E >= 1) #USE THE OTHER STREAM MASK
    Q_pos, S_pos, E_pos = Q[pos_mask], S[pos_mask], E[pos_mask]
    xs_pos, ys_pos = xs[pos_mask], ys[pos_mask]

    age_pos, tag_pos = nearest_volcano_age(xs_pos, ys_pos, pts.crs, return_tags=True)

    age_ok = np.isfinite(age_pos) & (age_pos > 0)
    Q_pos, S_pos, E_pos = Q_pos[age_ok], S_pos[age_ok], E_pos[age_ok]
    age_pos, tag_pos = age_pos[age_ok], tag_pos[age_ok]

    return Q_pos, S_pos, E_pos, age_pos, tag_pos


def make_plots(Q, S, E_norm, age, label, file_prefix):
    """Produces both fits/figures (Plot 1: combined OLS on the Q*S proxy;
    Plot 2: multivariate power-law fit with separate Q/S exponents) for one
    dataset -- either all islands combined, or a single island -- and saves
    both. `label` is used in titles/print statements, `file_prefix` in the
    saved filenames."""
    age_kyr = age / 1000.0
    QS = Q * S

    # ---------- Plot 1: combined OLS on the Q*S proxy ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(QS, E_norm, c=age_kyr, cmap='Blues',
                     norm=LogNorm(vmin=age_kyr.min(), vmax=age_kyr.max()),
                     s=3, alpha=1, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Representative Volcano Age (kyr)')

    log_QS = np.log10(QS)
    log_EN = np.log10(E_norm)
    m, b, r, p, se = stats.linregress(log_QS, log_EN)
    r2 = r ** 2
    rho, _ = stats.spearmanr(QS, E_norm)
    print(f"\n[{label}] Fit: log10(E/age) = {m:.6f} * log10(QS) + {b:.6f}   "
          f"(k = 10^b = {10**b:.6g}, R2 = {r2:.4f}, p = {p:.3g}, slope SE = {se:.3g})")

    # --- Refit on points within 1 SD of the combined OLS fit's residuals ---
    resid = log_EN - (m * log_QS + b)
    within_1sd = np.abs(resid) <= resid.std()
    m2, b2, r2_, p2, se2 = stats.linregress(log_QS[within_1sd], log_EN[within_1sd])
    r2_2 = r2_ ** 2
    print(f"[{label}] Refit (within 1 SD, n={within_1sd.sum():,}/{len(resid):,}): "
          f"log10(E/age) = {m2:.6f} * log10(QS) + {b2:.6f}   "
          f"(k = 10^b = {10**b2:.6g}, R2 = {r2_2:.4f}, p = {p2:.3g}, slope SE = {se2:.3g})")

    x_fit = np.logspace(log_QS.min(), log_QS.max(), 200)
    ax.plot(x_fit, (10 ** b) * (x_fit ** m), color='black', linewidth=1.8,
             linestyle='--', label='combined OLS fit')
    ax.plot(x_fit, (10 ** b2) * (x_fit ** m2), color='red', linewidth=1.8,
             linestyle='-.', label='within-1SD refit')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel(r'Stream Power Proxy ($Q \times S$)')
    ax.set_ylabel(r'Erosional Depth / Representative Volcano Age  '
                  r'($E$ / Age)  [m/yr]')
    ax.set_title(f'{label}')

    textstr = (f"Spearman's rho = {rho:.4f}\n"
               f"Log-Log R2 (combined OLS) = {r2:.4f}\n"
               f"Log-Log R2 (within-1SD refit) = {r2_2:.4f}")
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=14, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))

    leg = ax.legend(fontsize=13, markerscale=3, loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    out_path = f'{output_dir}/{file_prefix}_E_over_age_vs_QS.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")

    # --- Plot 2: multivariate power-law fit, E/age = k * Q^c * S^n ---
    # (separate exponents for Q and S, instead of Plot 1's single shared
    # exponent on the combined Q*S proxy) -- OLS in log space:
    # log10(E/age) = log10(k) + c*log10(Q) + n*log10(S)
    log_Q = np.log10(Q)
    log_S = np.log10(S)

    n_pts = len(log_EN)
    A = np.column_stack([np.ones(n_pts), log_Q, log_S])
    beta, _, _, _ = np.linalg.lstsq(A, log_EN, rcond=None)
    b3, c3, n3 = beta
    k3 = 10 ** b3

    pred3 = A @ beta
    resid3 = log_EN - pred3
    dof = n_pts - A.shape[1]
    mse = np.sum(resid3 ** 2) / dof
    se_beta = np.sqrt(np.diag(mse * np.linalg.inv(A.T @ A)))
    tvals = beta / se_beta
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)

    ss_res = np.sum(resid3 ** 2)
    ss_tot = np.sum((log_EN - log_EN.mean()) ** 2)
    r2_3 = 1 - ss_res / ss_tot

    QcSn = (Q ** c3) * (S ** n3)
    rho3, _ = stats.spearmanr(QcSn, E_norm)

    print(f"\n[{label}] Fit: log10(E/age) = {c3:.6f} * log10(Q) + {n3:.6f} * log10(S) + {b3:.6f}   "
          f"(k = 10^b = {k3:.6g}, R2 = {r2_3:.4f})")
    print(f"  c (Q exponent) = {c3:.4f}  SE = {se_beta[1]:.3g}  p = {pvals[1]:.3g}")
    print(f"  n (S exponent) = {n3:.4f}  SE = {se_beta[2]:.3g}  p = {pvals[2]:.3g}")
    print(f"  Spearman's rho = {rho3:.4f}")

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sc2 = ax2.scatter(QcSn, E_norm, c=age_kyr, cmap='Blues',
                       norm=LogNorm(vmin=age_kyr.min(), vmax=age_kyr.max()),
                       s=3, alpha=1, edgecolors='none')
    cbar2 = fig2.colorbar(sc2, ax=ax2, pad=0.02)
    cbar2.set_label('Representative Volcano Age (kyr)')

    x_fit2 = np.logspace(np.log10(QcSn.min()), np.log10(QcSn.max()), 200)
    ax2.plot(x_fit2, k3 * x_fit2, color='black', linewidth=1.8, linestyle='--',
             label=f'fit: $k\\,Q^{{{c3:.3f}}}S^{{{n3:.3f}}}$')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.minorticks_on()
    ax2.set_xlabel(r'Discharge x slope ($Q^{c} S^{n}$)')
    ax2.set_ylabel(r'Erosional Rate [m/yr]')
    ax2.set_title(f'{label}')

    textstr2 = (f"Spearman's rho = {rho3:.4f}\n"
                f"Log-Log R2 = {r2_3:.4f}\n"
                f"k = {k3:.4g},  c = {c3:.4f},  n = {n3:.4f}")
    ax2.text(0.05, 0.05, textstr2, transform=ax2.transAxes,
             fontsize=14, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))

    leg2 = ax2.legend(fontsize=13, markerscale=3, loc='upper left')
    for lh in leg2.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    out_path2 = f'{output_dir}/{file_prefix}_E_over_age_vs_QcSn.png'
    plt.savefig(out_path2, dpi=200)
    plt.close(fig2)
    print(f"Saved {out_path2}")


def main():
    per_island = {}
    all_Q, all_S, all_E_norm, all_age = [], [], [], []
    print("Mean age by island (volcanic sector):")
    for island in island_files:
        island_label = island.capitalize()

        result = get_island_QS_E(island)
        if result is None:
            print(f"[{island}] no maxflux_stream_points.shp found "
                  f"(run maxflux_stream_points.py first), skipping")
            continue
        Q, S, E, age, tag = result
        if len(Q) < 10:
            print(f"[{island}] too few valid pixels with representative age ({len(Q)}), skipping")
            continue
        E_norm = E / age

        per_island[island] = (Q, S, E_norm, age)
        all_Q.append(Q)
        all_S.append(S)
        all_E_norm.append(E_norm)
        all_age.append(age)

        for t in np.unique(tag):
            sel = (tag == t)
            print(f"  {island_label} ({t}): {age[sel][0]:,.0f} yr")

    all_Q = np.concatenate(all_Q)
    all_S = np.concatenate(all_S)
    all_E_norm = np.concatenate(all_E_norm)
    all_age = np.concatenate(all_age)

    # draw oldest (darkest) points last so they aren't buried under the
    # much more numerous young pixels
    order = np.argsort(all_age)
    all_Q, all_S, all_E_norm, all_age = all_Q[order], all_S[order], all_E_norm[order], all_age[order]

    make_plots(all_Q, all_S, all_E_norm, all_age, 'All Islands', 'all_islands')

    # --- same two plots again, once per individual island ---
    for island, (Q, S, E_norm, age) in per_island.items():
        order = np.argsort(age)
        Q, S, E_norm, age = Q[order], S[order], E_norm[order], age[order]
        make_plots(Q, S, E_norm, age, island.capitalize(), island)


if __name__ == '__main__':
    main()
