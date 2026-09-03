#!/usr/bin/env python3
"""
Erosional depth (d) and (d / representative volcano age) vs Stream Power
Proxy (Q x S) -- four log-log plots per island, plus four all-islands-
combined plots:

  1. d vs QS           -- single shared exponent: d = k*(QS)^m
  2. d vs Q^c*S^n       -- multivariate power-law, separate Q/S exponents
  3. d/age vs QS        -- single shared exponent: d/age = k*(QS)^m
  4. d/age vs Q^c*S^n   -- multivariate power-law, separate Q/S exponents

Plot types 2 and 4's Q exponent is named c (not m) to match the Q^c*S^n
convention used elsewhere for multivariate fits (e.g.
depth_over_age_vs_QS.py, weathering_stream_power_regression.py) -- m is
reserved for plot types 1/3's single shared exponent on the combined QS
proxy.
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy import stats

from usefulfunctions import sample_raster_at_points, nearest_volcano_age
from maxflux_stream_points import OUTPUT_DIR as MAXFLUX_OUTPUT_DIR

# Derived from this file's own location rather than hardcoded, since the
# project folder has been renamed before (it used to be "Summer Work 2026",
# with spaces) and a stale hardcoded path here silently no-ops every path
# built from it -- same caveat as recreation.py's PROJECT_DIR.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = f'{base_dir}/Temp Output Placements/flow_accumulation_plots'
os.makedirs(output_dir, exist_ok=True)

# One subfolder per fit type, so each of the 4 plot types' 8 (7 islands +
# all-islands) PNGs lands together instead of all 32 files sitting flat in
# output_dir.
FIT_SUBDIRS = {
    'depth_vs_QS':            f'{output_dir}/weighted_depth_vs_QS',
    'depth_vs_QcSn':          f'{output_dir}/weighted_depth_vs_QcSn',
    'depth_over_age_vs_QS':   f'{output_dir}/weighted_depth_over_age_vs_QS',
    'depth_over_age_vs_QcSn': f'{output_dir}/weighted_depth_over_age_vs_QcSn',
}
for _dir in FIT_SUBDIRS.values():
    os.makedirs(_dir, exist_ok=True)

# Per-island rasters live under Island Data/ (see recreation.py's note on
# the same reorg).
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

# maxflux_stream_points.py's combined, already-filtered point set -- the
# points it *included* (kept), not all_islands_maxflux_stream_points_removed.shp.
MAXFLUX_POINTS_PATH = MAXFLUX_OUTPUT_DIR + 'all_islands_maxflux_stream_points.shp'
_maxflux_points_gdf = None


def _load_maxflux_points():
    global _maxflux_points_gdf
    if _maxflux_points_gdf is None:
        _maxflux_points_gdf = gpd.read_file(MAXFLUX_POINTS_PATH)
    return _maxflux_points_gdf


def points_for_island(island):
    """Included stream-pixel points for `island` from maxflux_stream_points.py's
    combined output -- already restricted to the island's own stream mask
    and within PROXIMITY_PX pixels of Streams_reprojected.shp, so no further
    masking on Q itself is needed here."""
    gdf = _load_maxflux_points()
    subset = gdf[gdf['island'] == island]
    return subset if len(subset) > 0 else None


def get_island_Q_S_d(island):
    """Q = maxflux (precipitation-weighted) flow accumulation, taken from
    the included points in all_islands_maxflux_stream_points.shp (not
    resampled off the raw weighted_flow raster). S and d are sampled at
    those same point coordinates from the island's slope/erosion rasters
    and kept only where both are finite. Also returns each surviving
    point's own coordinates + CRS, for the age lookup used by the d/age
    plots."""
    paths = island_files[island]

    pts = points_for_island(island)
    if pts is None:
        return (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), None)

    xs, ys = pts.geometry.x.values, pts.geometry.y.values
    crs = pts.crs
    Q = pts['maxflux'].values

    slope_rast = sample_raster_at_points(paths['slope'], xs, ys, crs)
    erosion    = sample_raster_at_points(paths['erosion'], xs, ys, crs)

    valid_mask = np.isfinite(Q) & np.isfinite(slope_rast) & np.isfinite(erosion)

    Q, S, d = Q[valid_mask], slope_rast[valid_mask], erosion[valid_mask]
    xs, ys = xs[valid_mask], ys[valid_mask]

    pos_mask = (Q > 0) & (S > 0) & (d >= 1)
    return Q[pos_mask], S[pos_mask], d[pos_mask], xs[pos_mask], ys[pos_mask], crs


def add_age(Q, S, d, xs, ys, crs):
    """Looks up each point's representative volcano age and drops points
    with no valid (positive) age -- used only for the d/age plots."""
    age = nearest_volcano_age(xs, ys, crs)
    age_ok = np.isfinite(age) & (age > 0)
    return Q[age_ok], S[age_ok], d[age_ok], age[age_ok]


def _fit_annotate(ax, rho, r2, extra_lines=()):
    lines = [f"Spearman's rho = {rho:.4f}", f"Log-Log R2 = {r2:.4f}", *extra_lines]
    ax.text(0.05, 0.05, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))


# Up to 10 visually distinct colors -- 7 islands fits comfortably.
_ISLAND_KDE_COLORS = plt.cm.tab10.colors

# Same subsampling convention as recreation.py's _kde_plot -- gaussian KDE
# cost scales with n points x grid size, so evaluating it on a raw
# 200k+-point island (Hawaii, Kauai, ...) is far too slow. A 50k-point
# subsample gives an indistinguishable density estimate at a fraction of
# the cost; gridsize is also cut from seaborn's default 200 to 100 since
# only a coarse 3-level backdrop contour is needed here.
_KDE_N = 50_000


def _add_island_kde_backdrop(ax, island_xy, log_scale=True):
    """Draws a per-island KDE contour as an overlay on top of the main
    scatter on the 'All Islands' plots (zorder above the scatter), so it's
    visible where each island's own points sit within the combined
    distribution. island_xy: dict of
    island_label -> (x_values, y_values), already in the plot's own x/y
    units (e.g. QS and d). Returns legend handles (one per island actually
    drawn, skipping any with too few points) for the caller to fold into
    its own legend."""
    rng = np.random.default_rng(0)
    handles = []
    for i, (isl_label, (x, y)) in enumerate(island_xy.items()):
        if len(x) < 10:
            continue
        if len(x) > _KDE_N:
            idx = rng.choice(len(x), _KDE_N, replace=False)
            x, y = x[idx], y[idx]
        color = _ISLAND_KDE_COLORS[i % len(_ISLAND_KDE_COLORS)]
        try:
            sns.kdeplot(x=x, y=y, ax=ax, log_scale=log_scale, levels=3,
                        gridsize=100, color=color, linewidths=1.1, alpha=0.75,
                        zorder=4)
        except Exception as e:
            print(f"  (skipping KDE backdrop for {isl_label}: {e})")
            continue
        handles.append(Line2D([0], [0], color=color, lw=1.5, label=isl_label))
    return handles


# ---------- Plot type 1: d vs QS (single exponent) ----------
def make_plot_d_QS(Q, S, d, label, file_prefix, island_breakdown=None):
    QS = Q * S
    log_QS, log_d = np.log10(QS), np.log10(d)

    m, b, r, p, se = stats.linregress(log_QS, log_d)
    r2 = r ** 2
    rho, _ = stats.spearmanr(QS, d)
    print(f"[{label}]  log10(d) = {m:.6f} * log10(QS) + {b:.6f}   "
          f"(k = 10^b = {10**b:.6g}, R2 = {r2:.4f}, p = {p:.3g}, "
          f"slope SE = {se:.3g}, rho = {rho:.4f}, n = {len(d):,})")

    x_fit = np.logspace(log_QS.min(), log_QS.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 6))

    kde_handles = []
    if island_breakdown:
        island_xy = {isl: (Qi * Si, di) for isl, (Qi, Si, di) in island_breakdown.items()}
        kde_handles = _add_island_kde_backdrop(ax, island_xy)

    ax.scatter(QS, d, s=3, alpha=0.115, edgecolors='none', color='#888888', zorder=2)
    ax.plot(x_fit, (10 ** b) * (x_fit ** m), color='#3a6ea5', linewidth=1.8,
            label='OLS fit', zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel(r'Stream Power Proxy ($Q \times S$)')
    ax.set_ylabel('Erosional Depth (d) [m]')
    ax.set_title(label)

    _fit_annotate(ax, rho, r2, [f"k = {10**b:.4g},  m = {m:.4f}"])
    fit_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=fit_handles + kde_handles, fontsize=8, loc='upper left')

    plt.tight_layout()
    out_path = f"{FIT_SUBDIRS['depth_vs_QS']}/weighted_{file_prefix}_depth_vs_QS.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")

    return {'island': label, 'k': 10 ** b, 'm': m, 'r2': r2, 'rho': rho, 'n': len(d)}


# ---------- Plot type 2: d vs Q^c*S^n (multivariate) ----------
def make_plot_d_QcSn(Q, S, d, label, file_prefix, island_breakdown=None):
    log_Q, log_S, log_d = np.log10(Q), np.log10(S), np.log10(d)

    n_pts = len(log_d)
    M = np.column_stack([np.ones(n_pts), log_Q, log_S])
    beta, _, _, _ = np.linalg.lstsq(M, log_d, rcond=None)
    b, c, n = beta
    k = 10 ** b

    pred = M @ beta
    resid = log_d - pred
    dof = n_pts - M.shape[1]
    mse = np.sum(resid ** 2) / dof
    se_beta = np.sqrt(np.diag(mse * np.linalg.inv(M.T @ M)))
    tvals = beta / se_beta
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((log_d - log_d.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    QcSn = (Q ** c) * (S ** n)
    rho, _ = stats.spearmanr(QcSn, d)

    print(f"[{label}]  log10(d) = {c:.6f} * log10(Q) + {n:.6f} * log10(S) + {b:.6f}   "
          f"(k = 10^b = {k:.6g}, R2 = {r2:.4f}, rho = {rho:.4f}, n = {n_pts:,})")
    print(f"  c (Q exponent) = {c:.4f}  SE = {se_beta[1]:.3g}  p = {pvals[1]:.3g}")
    print(f"  n (S exponent) = {n:.4f}  SE = {se_beta[2]:.3g}  p = {pvals[2]:.3g}")

    x_fit = np.logspace(np.log10(QcSn.min()), np.log10(QcSn.max()), 200)

    fig, ax = plt.subplots(figsize=(7, 6))

    kde_handles = []
    if island_breakdown:
        island_xy = {isl: ((Qi ** c) * (Si ** n), di)
                     for isl, (Qi, Si, di) in island_breakdown.items()}
        kde_handles = _add_island_kde_backdrop(ax, island_xy)

    ax.scatter(QcSn, d, s=3, alpha=0.115, edgecolors='none', color='#888888', zorder=2)
    ax.plot(x_fit, k * x_fit, color='#3a6ea5', linewidth=1.8,
            label=f'fit: $k\\,Q^{{{c:.3f}}}S^{{{n:.3f}}}$', zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel(r'$Q^{c} S^{n}$')
    ax.set_ylabel('Erosional Depth (d) [m]')
    ax.set_title(label)

    _fit_annotate(ax, rho, r2, [f"k = {k:.4g},  c = {c:.4f},  n = {n:.4f}"])
    fit_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=fit_handles + kde_handles, fontsize=8, loc='upper left')

    plt.tight_layout()
    out_path = f"{FIT_SUBDIRS['depth_vs_QcSn']}/weighted_{file_prefix}_depth_vs_QcSn.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")

    return {'island': label, 'k': k, 'c': c, 'n': n, 'r2': r2, 'rho': rho, 'n_pts': n_pts}


# ---------- Plot type 3: d/age vs QS (single exponent) ----------
def make_plot_dage_QS(Q, S, d, age, label, file_prefix, island_breakdown=None):
    age_kyr = age / 1000.0
    d_over_age = d / age
    QS = Q * S
    log_QS, log_dN = np.log10(QS), np.log10(d_over_age)

    m, b, r, p, se = stats.linregress(log_QS, log_dN)
    r2 = r ** 2
    rho, _ = stats.spearmanr(QS, d_over_age)
    print(f"[{label}]  log10(d/age) = {m:.6f} * log10(QS) + {b:.6f}   "
          f"(k = 10^b = {10**b:.6g}, R2 = {r2:.4f}, p = {p:.3g}, "
          f"slope SE = {se:.3g}, rho = {rho:.4f}, n = {len(d_over_age):,})")

    x_fit = np.logspace(log_QS.min(), log_QS.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 6))

    kde_handles = []
    if island_breakdown:
        island_xy = {isl: (Qi * Si, di / agei)
                     for isl, (Qi, Si, di, agei) in island_breakdown.items()}
        kde_handles = _add_island_kde_backdrop(ax, island_xy)

    sc = ax.scatter(QS, d_over_age, c=age_kyr, cmap='Blues',
                     norm=LogNorm(vmin=age_kyr.min(), vmax=age_kyr.max()),
                     s=3, alpha=1, edgecolors='none', zorder=2)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Representative Volcano Age (kyr)')

    ax.plot(x_fit, (10 ** b) * (x_fit ** m), color='black', linewidth=1.8,
            linestyle='--', label='OLS fit', zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel(r'Stream Power Proxy ($Q \times S$)')
    ax.set_ylabel(r'Erosional Depth / Representative Volcano Age  '
                  r'($d$ / Age)  [m/yr]')
    ax.set_title(label)

    _fit_annotate(ax, rho, r2, [f"k = {10**b:.4g},  m = {m:.4f}"])
    leg = ax.legend(fontsize=9, markerscale=3, loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    if kde_handles:
        ax.add_artist(leg)
        ax.legend(handles=kde_handles, fontsize=8, loc='lower right')

    plt.tight_layout()
    out_path = f"{FIT_SUBDIRS['depth_over_age_vs_QS']}/weighted_{file_prefix}_depth_over_age_vs_QS.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")

    return {'island': label, 'k': 10 ** b, 'm': m, 'r2': r2, 'rho': rho, 'n': len(d_over_age)}


# ---------- Plot type 4: d/age vs Q^c*S^n (multivariate) ----------
def make_plot_dage_QcSn(Q, S, d, age, label, file_prefix, island_breakdown=None):
    age_kyr = age / 1000.0
    d_over_age = d / age
    log_Q, log_S, log_dN = np.log10(Q), np.log10(S), np.log10(d_over_age)

    n_pts = len(log_dN)
    M = np.column_stack([np.ones(n_pts), log_Q, log_S])
    beta, _, _, _ = np.linalg.lstsq(M, log_dN, rcond=None)
    b, c, n = beta
    k = 10 ** b

    pred = M @ beta
    resid = log_dN - pred
    dof = n_pts - M.shape[1]
    mse = np.sum(resid ** 2) / dof
    se_beta = np.sqrt(np.diag(mse * np.linalg.inv(M.T @ M)))
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

    kde_handles = []
    if island_breakdown:
        island_xy = {isl: ((Qi ** c) * (Si ** n), di / agei)
                     for isl, (Qi, Si, di, agei) in island_breakdown.items()}
        kde_handles = _add_island_kde_backdrop(ax, island_xy)

    sc = ax.scatter(QcSn, d_over_age, c=age_kyr, cmap='Blues',
                     norm=LogNorm(vmin=age_kyr.min(), vmax=age_kyr.max()),
                     s=3, alpha=1, edgecolors='none', zorder=2)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Representative Volcano Age (kyr)')

    x_fit = np.logspace(np.log10(QcSn.min()), np.log10(QcSn.max()), 200)
    ax.plot(x_fit, k * x_fit, color='black', linewidth=1.8, linestyle='--',
            label=f'fit: $k\\,Q^{{{c:.3f}}}S^{{{n:.3f}}}$', zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_xlabel(r'$Q^{c} S^{n}$')
    ax.set_ylabel(r'Erosional Depth / Representative Volcano Age  '
                  r'($d$ / Age)  [m/yr]')
    ax.set_title(label)

    _fit_annotate(ax, rho, r2, [f"k = {k:.4g},  c = {c:.4f},  n = {n:.4f}"])
    leg = ax.legend(fontsize=9, markerscale=3, loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    if kde_handles:
        ax.add_artist(leg)
        ax.legend(handles=kde_handles, fontsize=8, loc='lower right')

    plt.tight_layout()
    out_path = f"{FIT_SUBDIRS['depth_over_age_vs_QcSn']}/weighted_{file_prefix}_depth_over_age_vs_QcSn.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")

    return {'island': label, 'k': k, 'c': c, 'n': n, 'r2': r2, 'rho': rho, 'n_pts': n_pts}


def _print_table_single(title, rows):
    print(f"\n--- {title} ---")
    header = f"{'Island':<12} {'m (slope)':>10} {'R2':>8} {'rho':>8} {'n':>12}"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['island']:<12} {r['m']:>10.4f} {r['r2']:>8.4f} "
              f"{r['rho']:>8.4f} {r['n']:>12,}")


def _print_table_multivariate(title, rows):
    print(f"\n--- {title} ---")
    header = f"{'Island':<12} {'k':>12} {'c (Q exp)':>10} {'n (S exp)':>10} {'R2':>8} {'rho':>8} {'n pts':>12}"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['island']:<12} {r['k']:>12.4g} {r['c']:>10.4f} {r['n']:>10.4f} "
              f"{r['r2']:>8.4f} {r['rho']:>8.4f} {r['n_pts']:>12,}")


def main():
    per_island_noage = {}
    per_island_age = {}
    all_Q, all_S, all_d = [], [], []
    all_Q2, all_S2, all_d2, all_age2 = [], [], [], []

    for island in island_files:
        Q, S, d, xs, ys, crs = get_island_Q_S_d(island)
        if len(Q) < 10:
            print(f"[{island}] too few valid pixels ({len(Q)}), skipping")
            continue
        per_island_noage[island] = (Q, S, d)
        all_Q.append(Q); all_S.append(S); all_d.append(d)

        Q2, S2, d2, age2 = add_age(Q, S, d, xs, ys, crs)
        if len(Q2) < 10:
            print(f"[{island}] too few valid pixels with representative age ({len(Q2)}), skipping d/age plots")
            continue
        per_island_age[island] = (Q2, S2, d2, age2)
        all_Q2.append(Q2); all_S2.append(S2); all_d2.append(d2); all_age2.append(age2)

    all_Q, all_S, all_d = np.concatenate(all_Q), np.concatenate(all_S), np.concatenate(all_d)
    all_Q2 = np.concatenate(all_Q2)
    all_S2 = np.concatenate(all_S2)
    all_d2 = np.concatenate(all_d2)
    all_age2 = np.concatenate(all_age2)

    # draw oldest (darkest) points last on the age-colored plots so they
    # aren't buried under the much more numerous young pixels
    order2 = np.argsort(all_age2)
    all_Q2, all_S2, all_d2, all_age2 = all_Q2[order2], all_S2[order2], all_d2[order2], all_age2[order2]

    results_d_QS, results_d_QcSn = [], []
    results_dage_QS, results_dage_QcSn = [], []

    # Per-island breakdown for the "All Islands" plots' KDE backdrop, so
    # each island's own points can be seen against the combined distribution.
    noage_breakdown = {island.capitalize(): v for island, v in per_island_noage.items()}
    age_breakdown = {island.capitalize(): v for island, v in per_island_age.items()}

    results_d_QS.append(make_plot_d_QS(all_Q, all_S, all_d, 'All Islands', 'all_islands',
                                        island_breakdown=noage_breakdown))
    results_d_QcSn.append(make_plot_d_QcSn(all_Q, all_S, all_d, 'All Islands', 'all_islands',
                                            island_breakdown=noage_breakdown))
    results_dage_QS.append(make_plot_dage_QS(all_Q2, all_S2, all_d2, all_age2, 'All Islands', 'all_islands',
                                              island_breakdown=age_breakdown))
    results_dage_QcSn.append(make_plot_dage_QcSn(all_Q2, all_S2, all_d2, all_age2, 'All Islands', 'all_islands',
                                                  island_breakdown=age_breakdown))

    for island, (Q, S, d) in per_island_noage.items():
        label = island.capitalize()
        results_d_QS.append(make_plot_d_QS(Q, S, d, label, island))
        results_d_QcSn.append(make_plot_d_QcSn(Q, S, d, label, island))

    for island, (Q2, S2, d2, age2) in per_island_age.items():
        label = island.capitalize()
        order = np.argsort(age2)
        Q2, S2, d2, age2 = Q2[order], S2[order], d2[order], age2[order]
        results_dage_QS.append(make_plot_dage_QS(Q2, S2, d2, age2, label, island))
        results_dage_QcSn.append(make_plot_dage_QcSn(Q2, S2, d2, age2, label, island))

    _print_table_single('d vs QS -- OLS fit (d = k*(QS)^m)', results_d_QS)
    _print_table_multivariate('d vs Q^c*S^n -- OLS fit (d = k*Q^c*S^n)', results_d_QcSn)
    _print_table_single('d/age vs QS -- OLS fit (d/age = k*(QS)^m)', results_dage_QS)
    _print_table_multivariate('d/age vs Q^c*S^n -- OLS fit (d/age = k*Q^c*S^n)', results_dage_QcSn)


if __name__ == '__main__':
    main()
