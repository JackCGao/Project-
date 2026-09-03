import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rasterio')

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import geopandas as gpd
import seaborn as sns
import os

from usefulfunctions import nearest_age, nearest_volcano_age, sample_raster_at_points
from maxflux_stream_points import OUTPUT_DIR as MAXFLUX_OUTPUT_DIR

# Derived from this file's own location rather than hardcoded, since the
# project folder has been renamed before (it used to be "Summer Work 2026",
# with spaces) and a stale hardcoded path here silently no-ops every path
# built from it -- same caveat as maxflux_stream_points.py's PROJECT_DIR.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = f'{base_dir}/Temp Output Placements'

# Per-island source rasters (slope, erosion) live under Island Data/ -- the
# project's earlier flat-per-island-folder layout was reorganized under
# this common parent.
island_data_dir = f'{base_dir}/Island Data'

MAXFLUX_POINTS_PATH = MAXFLUX_OUTPUT_DIR + 'all_islands_maxflux_stream_points.shp'
_maxflux_points_gdf = None

def _load_maxflux_points():
    global _maxflux_points_gdf
    if _maxflux_points_gdf is None:
        _maxflux_points_gdf = gpd.read_file(MAXFLUX_POINTS_PATH)
    return _maxflux_points_gdf

def points_for_island(island):
    """Stream-pixel points for `island` from maxflux_stream_points.py's
    combined output -- already restricted to the island's own stream mask
    and within 50 px of Streams_reprojected.shp, so no further masking is
    needed here; every point in the file is used as-is."""
    gdf = _load_maxflux_points()
    subset = gdf[gdf['island'] == island]
    return subset if len(subset) > 0 else None

_KDE_N = 50_000

def _kde_plot(ax, x, y, log_scale=False, **kw):
    """KDE density overlay, subsampled to _KDE_N points for speed on large rasters."""
    if len(x) > _KDE_N:
        idx = np.random.default_rng(0).choice(len(x), _KDE_N, replace=False)
        x, y = x[idx], y[idx]
    opts = dict(levels=10, fill=True, cmap='YlOrRd', alpha=0.75, thresh=0.02)
    opts.update(kw)
    sns.kdeplot(x=x, y=y, ax=ax, log_scale=log_scale, **opts)

def _log_lim(arr):
    """Decade-rounded (lower, upper) limits for a positive array."""
    lo = 10 ** np.floor(np.log10(np.nanmin(arr[arr > 0])))
    hi = 10 ** np.ceil( np.log10(np.nanmax(arr)))
    return lo, hi

_hawaii_dir    = f'{island_data_dir}/hawaii'
_kahoolawe_dir = f'{island_data_dir}/kahoolawe'
_oahu_dir      = f'{island_data_dir}/oahu/new'

island_files = {
    'hawaii': {
        'slope':      f'{_hawaii_dir}/hawaii_slope_nans.tif',
        'erosion':    f'{_hawaii_dir}/hawaii_erosion_nans.tif',
    },
    'kahoolawe': {
        'slope':      f'{_kahoolawe_dir}/kahoolawe_slope_nans.tif',
        'erosion':    f'{_kahoolawe_dir}/kahoolawe_erosion_nans.tif',
    },
    'oahu': {
        'slope':      f'{_oahu_dir}/oahu_slope_nans.tif',
        'erosion':    f'{_oahu_dir}/oahu_erosion_nans.tif',
    },
    'kauai': {
        'slope':      f'{island_data_dir}/kauai/new/kauai_slope.tif',
        'erosion':    f'{island_data_dir}/kauai/new/kauai_erosion_nans.tif',
    },
    'lanai': {
        'slope':      f'{island_data_dir}/lanai/new/lanai_slope.tif',
        'erosion':    f'{island_data_dir}/lanai/new/lanai_erosion_nans.tif',
    },
    'molokai': {
        'slope':      f'{island_data_dir}/molokai/new/molokai_slope_nans.tif',
        'erosion':    f'{island_data_dir}/molokai/new/molokai_erosion_nans.tif',
    },
    'maui': {
        'slope':      f'{island_data_dir}/maui/new/maui_slope_nans.tif',
        'erosion':    f'{island_data_dir}/maui/new/maui_erosion_nans.tif',
    },
}
islands = list(island_files.keys())

def process_island(island):
    slopes_path      = island_files[island]['slope']
    erosion_path     = island_files[island]['erosion']

    pts = points_for_island(island)
    if pts is None:
        print(f"[{island}] no maxflux_stream_points found, skipping")
        return None
    xs, ys = pts.geometry.x.values, pts.geometry.y.values
    crs = pts.crs
    flow_accum = pts['maxflux'].values

    slope_rast = sample_raster_at_points(slopes_path, xs, ys, crs)
    erosion    = sample_raster_at_points(erosion_path, xs, ys, crs)

    valid_mask = np.isfinite(slope_rast) & np.isfinite(flow_accum) & np.isfinite(erosion)

    Q = flow_accum[valid_mask]
    S = slope_rast[valid_mask]
    E = erosion[valid_mask]
    xs, ys = xs[valid_mask], ys[valid_mask]

    pos_mask = (Q > 0) & (S > 0) & (E >= 1)
    Q_pos = Q[pos_mask]
    S_pos = S[pos_mask]
    E_pos = E[pos_mask]
    xs_pos, ys_pos = xs[pos_mask], ys[pos_mask]

    QS_pos = Q_pos * S_pos

    # --- OLS in log space: log10(d) = slope*log10(QS) + b  =>  d = 10^b * QS^slope ---
    log_QS = np.log10(QS_pos)
    log_E  = np.log10(E_pos)

    m, b, r1, _, _ = stats.linregress(log_QS, log_E)
    r2_1 = r1 ** 2

    rho, _ = stats.spearmanr(QS_pos, E_pos)

    x_fit = np.logspace(np.log10(QS_pos.min()), np.log10(QS_pos.max()), 200)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(QS_pos, E_pos, s=3, alpha=0.115, edgecolors='none', color='#888888')
    _kde_plot(ax, QS_pos, E_pos, log_scale=True)
    ax.plot(x_fit, (10**b) * (x_fit ** m), color='#3a6ea5', linewidth=1.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*_log_lim(QS_pos))
    ax.set_ylim(*_log_lim(E_pos))
    ax.minorticks_on()
    ax.set_xlabel(r'Stream Power Proxy ($Q \times S$)')
    ax.set_ylabel('Erosional Depth (d) [m]')
    ax.set_title(island.capitalize())
    textstr = (f"Spearman's rank correlation = {rho:.4f}\n"
               f"Log–Log R² = {r2_1:.4f}")
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{island}_E_vs_QS.png'), dpi=200)
    plt.close(fig)

    # --- Fit 2: d = Q^c * S  =>  log10(d/S) = c*log10(Q) + b2  =>  d/S = 10^b2 * Q^c ---
    ES_pos = E_pos / S_pos
    log_Q  = np.log10(Q_pos)
    log_ES = np.log10(ES_pos)

    # OLS in log space (same method as fit 1)
    c, b2, r2_raw, _, _ = stats.linregress(log_Q, log_ES)
    r2_2 = r2_raw ** 2

    rho2, _ = stats.spearmanr(Q_pos, ES_pos)

    # --- Plot 2: E/S vs Q ---
    x_fit2 = np.logspace(np.log10(Q_pos.min()), np.log10(Q_pos.max()), 200)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Q_pos, ES_pos, s=3, alpha=0.115, edgecolors='none', color='#888888')
    _kde_plot(ax, Q_pos, ES_pos, log_scale=True)
    ax.plot(x_fit2, (10**b2) * (x_fit2 ** c), color='#3a6ea5', linewidth=1.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*_log_lim(Q_pos))
    ax.set_ylim(*_log_lim(ES_pos))
    ax.minorticks_on()
    ax.set_xlabel(r'Discharge Proxy ($Q$)')
    ax.set_ylabel(r'$d \,/\, S$ [m]')
    ax.set_title(island.capitalize())
    textstr2 = (f"Spearman's rank correlation = {rho2:.4f}\n"
                f"Log–Log R² = {r2_2:.4f}\n"
                f"c = {c:.4f},  k = 10^{b2:.4f} = {10**b2:.4g}")
    ax.text(0.05, 0.05, textstr2, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{island}_E_vs_QcS.png'), dpi=200)
    plt.close(fig)

    
    # --- Plot 3: Spearman's rho vs c (sweep) ---
    c_vals   = np.linspace(0, 3, 300)
    rho_vals = np.array([stats.spearmanr(Q_pos**cv * S_pos, E_pos)[0] for cv in c_vals])

    # Spearman-optimal c: peak of the sweep
    argmax_idx = np.argmax(rho_vals)
    c_sp  = c_vals[argmax_idx]
    # k for c_sp: best-fit intercept given fixed c_sp (log-space mean)
    k_sp  = 10 ** np.mean(log_E - c_sp * log_Q)
    rho_sp, _ = stats.spearmanr(Q_pos**c_sp * S_pos, E_pos)
    
    rho_span = rho_vals.max() - rho_vals.min()
    at_boundary = argmax_idx in (0, len(c_vals) - 1)
    near_flat_at_boundary = (rho_span > 0) and (
        abs(rho_vals[-1] - rho_vals.max()) < 0.02 * rho_span
    )
    c_sp_unreliable = at_boundary or near_flat_at_boundary
    if c_sp_unreliable:
        reason = "argmax at search-range edge" if at_boundary else \
                 "sweep is flat near the edge -- true optimum may lie outside [0, 3]"
        print(f"[{island}] WARNING: Spearman-optimal c={c_sp:.4f} is not a "
              f"well-defined interior peak ({reason}); treat as unreliable.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(c_vals, rho_vals, color='#3a6ea5', linewidth=1.8)
    ax.axvline(c,    color='#e06b3a', linewidth=1.4, linestyle='--',
               label=f'OLS c = {c:.4f}')
    c_sp_label = f'Spearman c = {c_sp:.4f}' + (' (unreliable)' if c_sp_unreliable else '')
    ax.axvline(c_sp, color='#2ca02c', linewidth=1.4, linestyle=':',
               label=c_sp_label)
    if c_sp_unreliable:
        ax.text(0.05, 0.95,
                "No well-defined interior peak --\nc is not a reliable optimum",
                transform=ax.transAxes, fontsize=8.5, verticalalignment='top',
                color='#b22222',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#b22222'))
    ax.set_xlabel('Exponent $c$')
    ax.set_ylabel("Spearman's $\\rho$")
    ax.set_title(island.capitalize())
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{island}_rho_vs_c.png'), dpi=200)
    plt.close(fig)

    # --- Plot 3b: E/S vs Q using Spearman-optimal c and k ---
    x_fit_sp = np.logspace(np.log10(Q_pos.min()), np.log10(Q_pos.max()), 200)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Q_pos, ES_pos, s=3, alpha=0.115, edgecolors='none', color='#888888')
    _kde_plot(ax, Q_pos, ES_pos, log_scale=True)
    ax.plot(x_fit_sp, k_sp * (x_fit_sp ** c_sp), color='#3a6ea5', linewidth=1.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*_log_lim(Q_pos))
    ax.set_ylim(*_log_lim(ES_pos))
    ax.minorticks_on()
    ax.set_xlabel(r'Discharge Proxy ($Q$)')
    ax.set_ylabel(r'$d \,/\, S$ [m]')
    ax.set_title(island.capitalize())
    fit_stats_text = (f"Spearman's ρ = {rho_sp:.4f}\n"
                       f"c = {c_sp:.4f},  k = {k_sp:.4g}")
    if c_sp_unreliable:
        fit_stats_text += "\n(c not a well-defined optimum -- see rho_vs_c plot)"
    ax.text(0.05, 0.05, fit_stats_text,
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99,
                      edgecolor='#b22222' if c_sp_unreliable else 'gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{island}_E_vs_QcS_spearman.png'), dpi=200)
    plt.close(fig)

    # --- Plot 4: log-log residuals vs average geological age ---
    age_flat = nearest_age(xs_pos, ys_pos, crs)

    log_resid = log_E - (m * log_QS + b)             # residuals in log10 space

    age_valid = np.isfinite(age_flat)
    if age_valid.sum() > 0:
        age_ma    = age_flat[age_valid] / 1e6        # convert years → Ma
        resid_sub = log_resid[age_valid]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(age_ma, resid_sub, s=3, alpha=0.115, edgecolors='none', color='#888888')
        ax.axhline(0, color='#e06b3a', linewidth=1.2, linestyle='--')
        ax.set_xlim(age_ma.min(), age_ma.max())
        ax.set_ylim(resid_sub.min(), resid_sub.max())
        ax.set_xlabel('Average Geological Age (Ma)')
        ax.set_ylabel(r'Log-Log Residual  $\log_{10}(d_\mathrm{obs}/d_\mathrm{pred})$')
        ax.set_title(island.capitalize())
        rho_age, _ = stats.spearmanr(age_ma, resid_sub)
        ax.text(0.05, 0.95, f"Spearman ρ = {rho_age:.4f}  (n = {age_valid.sum():,})",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{island}_resid_vs_age.png'), dpi=200)
        plt.close(fig)
    
    return {'island': island,
            'slope': m, 'intercept': b, 'r2_QS': r2_1, 'rho': rho,
            'c': c, 'r2_QcS': r2_2, 'rho2': rho2,
            'c_sp': c_sp, 'k_sp': k_sp, 'rho_sp': rho_sp,
            'c_sp_unreliable': c_sp_unreliable,
            'n': Q_pos.size}

def process_island_erosion_rate(island):
    """
    Same data/masking pipeline and plot set as process_island, but the
    y-axis quantity is erosion RATE -- erosional depth divided by each
    pixel's representative volcano age (usefulfunctions.representative_age)
    -- instead of raw erosional depth, still fit with an ordinary log-log
    OLS power law (same method as process_island, not isotonic).

    The rho-vs-c exponent sweep and its Spearman-optimal power-law fit
    (plots 3 & 3b in process_island) are specific to raw E vs QS and are
    skipped here, same as before.
    """
    slopes_path      = island_files[island]['slope']
    erosion_path     = island_files[island]['erosion']

    pts = points_for_island(island)
    if pts is None:
        print(f"[{island}] no maxflux_stream_points found, skipping")
        return None
    xs, ys = pts.geometry.x.values, pts.geometry.y.values
    crs = pts.crs
    flow_accum = pts['maxflux'].values

    slope_rast = sample_raster_at_points(slopes_path, xs, ys, crs)
    erosion    = sample_raster_at_points(erosion_path, xs, ys, crs)

    valid_mask = np.isfinite(slope_rast) & np.isfinite(flow_accum) & np.isfinite(erosion)

    Q = flow_accum[valid_mask]
    S = slope_rast[valid_mask]
    E = erosion[valid_mask]
    xs, ys = xs[valid_mask], ys[valid_mask]

    pos_mask = (Q > 0) & (S > 0) & (E >= 1)
    Q_pos = Q[pos_mask]
    S_pos = S[pos_mask]
    E_pos = E[pos_mask]
    xs_pos, ys_pos = xs[pos_mask], ys[pos_mask]

    # --- Representative age per pixel (per-volcano average, not per-polygon) ---
    age_flat, tag_flat = nearest_volcano_age(xs_pos, ys_pos, crs, return_tags=True)

    age_ok = np.isfinite(age_flat) & (age_flat > 0)
    Q_pos, S_pos, E_pos = Q_pos[age_ok], S_pos[age_ok], E_pos[age_ok]
    xs_pos, ys_pos = xs_pos[age_ok], ys_pos[age_ok]
    age_pos, tag_pos = age_flat[age_ok], tag_flat[age_ok]

    age_usage = {}
    for tag in np.unique(tag_pos):
        tag_sel = (tag_pos == tag)
        age_usage[tag] = (float(age_pos[tag_sel][0]), int(tag_sel.sum()))

    EA_pos = E_pos / age_pos   # erosion RATE (m/yr)

    QS_pos = Q_pos * S_pos
    log_QS = np.log10(QS_pos)
    log_EA = np.log10(EA_pos)

    rho, _ = stats.spearmanr(QS_pos, EA_pos)

    # --- Fit 1: log10(d/age) = m1*log10(QS) + b1 (OLS) ---
    m1, b1, r1, _, _ = stats.linregress(log_QS, log_EA)
    r2_1 = r1 ** 2

    x_fit = np.logspace(np.log10(QS_pos.min()), np.log10(QS_pos.max()), 200)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(QS_pos, EA_pos, s=3, alpha=0.115, edgecolors='none', color='#888888')
    _kde_plot(ax, QS_pos, EA_pos, log_scale=True)
    ax.plot(x_fit, (10**b1) * (x_fit ** m1), color='#3a6ea5', linewidth=1.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*_log_lim(QS_pos))
    ax.set_ylim(*_log_lim(EA_pos))
    ax.minorticks_on()
    ax.set_xlabel(r'Stream Power Proxy ($Q \times S$)')
    ax.set_ylabel('Erosion Rate (d / Representative Age) [m/yr]')
    ax.set_title(f'{island.capitalize()} (Age-Normalized)')
    textstr = (f"Spearman's rank correlation = {rho:.4f}\n"
               f"Log–Log R² = {r2_1:.4f}")
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{island}_ErateAge_vs_QS.png'), dpi=200)
    plt.close(fig)

    # --- Fit 2: log10((d/age)/S) = c*log10(Q) + b2 (OLS) ---
    EAS_pos = EA_pos / S_pos
    log_Q   = np.log10(Q_pos)
    log_EAS = np.log10(EAS_pos)

    c, b2, r2_raw, _, _ = stats.linregress(log_Q, log_EAS)
    r2_2 = r2_raw ** 2

    rho2, _ = stats.spearmanr(Q_pos, EAS_pos)

    x_fit2 = np.logspace(np.log10(Q_pos.min()), np.log10(Q_pos.max()), 200)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Q_pos, EAS_pos, s=3, alpha=0.115, edgecolors='none', color='#888888')
    _kde_plot(ax, Q_pos, EAS_pos, log_scale=True)
    ax.plot(x_fit2, (10**b2) * (x_fit2 ** c), color='#3a6ea5', linewidth=1.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*_log_lim(Q_pos))
    ax.set_ylim(*_log_lim(EAS_pos))
    ax.minorticks_on()
    ax.set_xlabel(r'Discharge Proxy ($Q$)')
    ax.set_ylabel(r'$(d / \mathrm{Age}) \,/\, S$ [m/yr]')
    ax.set_title(f'{island.capitalize()} (Age-Normalized)')
    textstr2 = (f"Spearman's rank correlation = {rho2:.4f}\n"
                f"Log–Log R² = {r2_2:.4f}\n"
                f"c = {c:.4f},  k = 10^{b2:.4f} = {10**b2:.4g}")
    ax.text(0.05, 0.05, textstr2, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{island}_ErateAge_vs_QcS.png'), dpi=200)
    plt.close(fig)

    result = {'island': island, 'slope': m1, 'intercept': b1, 'r2_QS': r2_1, 'rho': rho,
              'c': c, 'r2_QcS': r2_2, 'rho2': rho2,
              'n': Q_pos.size, 'age_usage': age_usage}

    return result

# --- Run for all islands ---
results = []
for island in islands:
    res = process_island(island)
    if res:
        results.append(res)

# --- Summary table: d vs QS fit (slope, intercept, R², rho) by island ---
print("\n--- d (Erosional Depth) vs QS -- OLS fit + Spearman rho, by island ---")
header = f"{'Island':<10} {'slope (m)':>10} {'intercept (b)':>14} {'R²':>8} {'rho':>8}"
print(header)
print('-' * len(header))
for res in results:
    print(f"{res['island']:<10} {res['slope']:>10.4f} {res['intercept']:>14.4f} "
          f"{res['r2_QS']:>8.4f} {res['rho']:>8.4f}")

# --- Run age-normalized (erosion-rate) variant for all islands ---
results_erosion_rate = []
for island in islands:
    res_er = process_island_erosion_rate(island)
    if res_er:
        results_erosion_rate.append(res_er)

print("\n--- Representative ages used, by island ---")
for res_er in results_erosion_rate:
    print(f"\n[{res_er['island']}]  (n = {res_er['n']:,} points total)")
    for tag, (age, cnt) in sorted(res_er['age_usage'].items(), key=lambda kv: kv[1][0]):
        print(f"    volcano '{tag}':  age = {age:,.0f} yr  ({cnt:,} points)")