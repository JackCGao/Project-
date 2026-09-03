#!/usr/bin/env python3
"""
(Erosional Depth / Representative Volcano Age) vs Stream Power Proxy --
two log-log plots per island, plus two all-islands-combined plots:

  1. Single shared exponent on the QS product: d/age = k*(QS)^m -- same
     single-exponent method as depth_vs_QS_by_island.py / depth_vs_QS.py.
  2. Multivariate power-law fit with separate Q/S exponents:
     d/age = k * Q^c * S^n -- same method as depth_over_age_vs_QS.py's
     second plot.

Deliberately does NOT go through maxflux_stream_points.py's output ("the
filter"), unlike depth_over_age_vs_QS.py: Q comes from each island's
weighted flow accumulation raster in Island Data/weighted_flow/ (D8
max-flux, already masked to stream pixels -- nodata everywhere else),
which also doubles as the mask -- S and d are only sampled wherever that
raster has a valid (non-nodata) value, with no separate streams==1
raster and no hydro-proximity refinement.

Age comes from usefulfunctions.nearest_volcano_age (the point-level core
of representative_age -- nearest volcano region's average age, from
ageprocessing.py's volcano_avg_age_regions.shp) evaluated at each valid
pixel's own coordinates, same age method as recreation.py's
process_island_erosion_rate / depth_over_age_vs_QS.py.
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

from usefulfunctions import load_clean, nearest_volcano_age

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


def get_island_QS_d_age(island):
    """Q (weighted flow accumulation, already stream-masked) doubles as the
    mask: S and d are only kept where Q itself is finite -- no separate
    streams==1 raster and no maxflux_stream_points.py involvement, same as
    depth_vs_QS_by_island.py. Age is looked up per surviving pixel at that
    pixel's own coordinates via nearest_volcano_age."""
    paths = island_files[island]

    with rasterio.open(paths['flow_accum']) as src:
        transform = src.transform
        crs = src.crs

    flow_accum = load_clean(paths['flow_accum'])
    slope_rast = load_clean(paths['slope'])
    erosion    = load_clean(paths['erosion'])

    valid_mask = (np.isfinite(flow_accum) & np.isfinite(slope_rast) &
                  np.isfinite(erosion))

    rows, cols = np.nonzero(valid_mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs, ys = np.asarray(xs), np.asarray(ys)

    Q = flow_accum[valid_mask]
    S = slope_rast[valid_mask]
    d = erosion[valid_mask]

    pos_mask = (Q > 0) & (S > 0) & (d >= 1)
    Q, S, d = Q[pos_mask], S[pos_mask], d[pos_mask]
    xs, ys = xs[pos_mask], ys[pos_mask]

    age = nearest_volcano_age(xs, ys, crs)
    age_ok = np.isfinite(age) & (age > 0)
    Q, S, d, age = Q[age_ok], S[age_ok], d[age_ok], age[age_ok]

    return Q, S, d / age, age


def make_plot(Q, S, d_over_age, age, label, file_prefix):
    age_kyr = age / 1000.0
    QS = Q * S
    log_QS = np.log10(QS)
    log_dN = np.log10(d_over_age)

    m, b, r, p, se = stats.linregress(log_QS, log_dN)
    r2 = r ** 2
    rho, _ = stats.spearmanr(QS, d_over_age)
    print(f"[{label}]  log10(d/age) = {m:.6f} * log10(QS) + {b:.6f}   "
          f"(k = 10^b = {10**b:.6g}, R2 = {r2:.4f}, p = {p:.3g}, "
          f"slope SE = {se:.3g}, rho = {rho:.4f}, n = {len(d_over_age):,})")

    x_fit = np.logspace(log_QS.min(), log_QS.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(QS, d_over_age, c=age_kyr, cmap='Blues',
                     norm=LogNorm(vmin=age_kyr.min(), vmax=age_kyr.max()),
                     s=3, alpha=1, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Representative Volcano Age (kyr)')

    ax.plot(x_fit, (10 ** b) * (x_fit ** m), color='black', linewidth=1.8,
            linestyle='--', label='OLS fit')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel(r'Stream Power Proxy ($Q \times S$)')
    ax.set_ylabel(r'Erosional Depth / Representative Volcano Age  '
                  r'($d$ / Age)  [m/yr]')
    ax.set_title(label)

    textstr = (f"Spearman's rho = {rho:.4f}\n"
               f"Log-Log R2 = {r2:.4f}\n"
               f"k = {10**b:.4g},  m = {m:.4f}")
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))

    leg = ax.legend(fontsize=9, markerscale=3, loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    out_path = f'{output_dir}/{file_prefix}_depth_over_age_vs_QS_no_streampoints.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def make_plot_QcSn(Q, S, d_over_age, age, label, file_prefix):
    """Multivariate power-law fit, d/age = k * Q^c * S^n (separate
    exponents for Q and S, instead of make_plot's single shared exponent
    on the combined Q*S proxy) -- OLS in log space:
    log10(d/age) = log10(k) + c*log10(Q) + n*log10(S)."""
    age_kyr = age / 1000.0
    log_Q = np.log10(Q)
    log_S = np.log10(S)
    log_dN = np.log10(d_over_age)

    n_pts = len(log_dN)
    A = np.column_stack([np.ones(n_pts), log_Q, log_S])
    beta, _, _, _ = np.linalg.lstsq(A, log_dN, rcond=None)
    b, c, n = beta
    k = 10 ** b

    pred = A @ beta
    resid = log_dN - pred
    dof = n_pts - A.shape[1]
    mse = np.sum(resid ** 2) / dof
    se_beta = np.sqrt(np.diag(mse * np.linalg.inv(A.T @ A)))
    tvals = beta / se_beta
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((log_dN - log_dN.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    QcSn = (Q ** c) * (S ** n)
    rho, _ = stats.spearmanr(QcSn, d_over_age)

    print(f"[{label}]  log10(d/age) = {c:.6f} * log10(Q) + {n:.6f} * log10(S) + {b:.6f}   "
          f"(k = 10^b = {k:.6g}, R2 = {r2:.4f}, rho = {rho:.4f}, n = {n_pts:,})")
    print(f"  c (Q exponent) = {c:.4f}  SE = {se_beta[1]:.3g}  p = {pvals[1]:.3g}")
    print(f"  n (S exponent) = {n:.4f}  SE = {se_beta[2]:.3g}  p = {pvals[2]:.3g}")

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(QcSn, d_over_age, c=age_kyr, cmap='Blues',
                     norm=LogNorm(vmin=age_kyr.min(), vmax=age_kyr.max()),
                     s=3, alpha=1, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Representative Volcano Age (kyr)')

    x_fit = np.logspace(np.log10(QcSn.min()), np.log10(QcSn.max()), 200)
    ax.plot(x_fit, k * x_fit, color='black', linewidth=1.8, linestyle='--',
            label=f'fit: $k\\,Q^{{{c:.3f}}}S^{{{n:.3f}}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel(r'Discharge x slope ($Q^{c} S^{n}$)')
    ax.set_ylabel(r'Erosional Depth / Representative Volcano Age  '
                  r'($d$ / Age)  [m/yr]')
    ax.set_title(label)

    textstr = (f"Spearman's rho = {rho:.4f}\n"
               f"Log-Log R2 = {r2:.4f}\n"
               f"k = {k:.4g},  c = {c:.4f},  n = {n:.4f}")
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))

    leg = ax.legend(fontsize=9, markerscale=3, loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    out_path = f'{output_dir}/{file_prefix}_depth_over_age_vs_QcSn_no_streampoints.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    per_island = {}
    all_Q, all_S, all_dN, all_age = [], [], [], []

    for island in island_files:
        Q, S, d_over_age, age = get_island_QS_d_age(island)
        if len(Q) < 10:
            print(f"[{island}] too few valid pixels with representative age ({len(Q)}), skipping")
            continue

        per_island[island] = (Q, S, d_over_age, age)
        all_Q.append(Q)
        all_S.append(S)
        all_dN.append(d_over_age)
        all_age.append(age)

    all_Q = np.concatenate(all_Q)
    all_S = np.concatenate(all_S)
    all_dN = np.concatenate(all_dN)
    all_age = np.concatenate(all_age)

    # draw oldest (darkest) points last so they aren't buried under the
    # much more numerous young pixels
    order = np.argsort(all_age)
    all_Q, all_S, all_dN, all_age = all_Q[order], all_S[order], all_dN[order], all_age[order]

    make_plot(all_Q, all_S, all_dN, all_age, 'All Islands', 'all_islands')
    make_plot_QcSn(all_Q, all_S, all_dN, all_age, 'All Islands', 'all_islands')

    for island, (Q, S, d_over_age, age) in per_island.items():
        order = np.argsort(age)
        Q, S, d_over_age, age = Q[order], S[order], d_over_age[order], age[order]
        make_plot(Q, S, d_over_age, age, island.capitalize(), island)
        make_plot_QcSn(Q, S, d_over_age, age, island.capitalize(), island)


if __name__ == '__main__':
    main()
