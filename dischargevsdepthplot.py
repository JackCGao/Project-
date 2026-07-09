import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rasterio')

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# --- Config: adjust these to match your actual folder structure ---
# Assumes each island has its own subfolder under a shared base directory,
# e.g. .../lanai/new/lanai_d8maxflux_nans.tif
#      .../molokai/new/molokai_d8maxflux_nans.tif, etc.
base_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao'
islands = ['lanai', 'molokai', 'maui', 'kauai']
output_dir = '/Users/jackgao/Desktop'

def load_clean(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr

def process_island(island):
    island_dir = os.path.join(base_dir, island, 'new')
    flow_accum_path = os.path.join(island_dir, f'{island}_d8maxflux_nans.tif')
    slopes_path = os.path.join(island_dir, f'{island}_slope_nans.tif')
    erosion_path = os.path.join(island_dir, f'{island}_erosion_nans.tif')

    for p in [flow_accum_path, slopes_path, erosion_path]:
        if not os.path.exists(p):
            print(f"⚠️  Skipping {island}: missing file {p}")
            return None

    flow_accum = load_clean(flow_accum_path)
    slope = load_clean(slopes_path)
    erosion = load_clean(erosion_path)

    erosion_mask = np.isfinite(erosion)
    predictor = flow_accum * slope
    final_valid = erosion_mask & np.isfinite(predictor)

    x_vals = predictor[final_valid]
    y_vals = erosion[final_valid]

    pos_mask = (x_vals > 0) & (y_vals > 0)
    x_pos = x_vals[pos_mask]
    y_pos = y_vals[pos_mask]
    n_dropped = x_vals.size - x_pos.size

    print(f"\n=== {island.capitalize()} ===")
    print(f"Total valid points: {x_vals.size}")
    print(f"Dropped (x<=0 or y<=0): {n_dropped}")
    print(f"Plotted points: {x_pos.size}")

    if x_pos.size < 2:
        print(f"⚠️  Not enough points to fit for {island}")
        return None

    # Log-log fit: ln(y) = k*ln(x) + ln(c)  =>  y = c * x^k
    log_x = np.log(x_pos)
    log_y = np.log(y_pos)
    k, log_c = np.polyfit(log_x, log_y, 1)
    c = np.exp(log_c)

    log_y_pred = k * log_x + log_c
    ss_res = np.sum((log_y - log_y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    log_log_r2 = 1 - ss_res / ss_tot

    y_pred_linear_scale = c * x_pos ** k
    ss_res_lin = np.sum((y_pos - y_pred_linear_scale) ** 2)
    ss_tot_lin = np.sum((y_pos - np.mean(y_pos)) ** 2)
    linear_r2 = 1 - ss_res_lin / ss_tot_lin

    rho, pval = stats.spearmanr(x_pos, y_pos)

    print(f"Power-law fit: erosion = {c:.4g} * predictor^{k:.4f}")
    print(f"Log-Log R^2 = {log_log_r2:.4f} | Linear R^2 = {linear_r2:.4f} | Spearman's rho = {rho:.4f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_pos, y_pos, s=3, alpha=0.15, edgecolors='none', color='#5b8db8')

    x_fit = np.logspace(np.log10(x_pos.min()), np.log10(x_pos.max()), 200)
    y_fit = c * x_fit ** k
    ax.plot(x_fit, y_fit, color='#3a6ea5', linewidth=1.8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e4, 1e11)
    ax.set_ylim(1e0, 1e3)
    ax.set_xlabel(r'Stream Power (Proxy) ($Q \times S$) [$m^3$/s]')
    ax.set_ylabel('Erosional Depth (d) [m]')
    ax.set_title(island.capitalize())

    textstr = (f"Spearman's rank correlation = {rho:.4f}\n"
               f"Linear R\u00b2 = {linear_r2:.4f}\n"
               f"Log\u2013Log R\u00b2 = {log_log_r2:.4f}")
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{island}_erosion_vs_streampower_loglog.png')
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

    return {'island': island, 'k': k, 'c': c, 'log_log_r2': log_log_r2,
            'linear_r2': linear_r2, 'rho': rho, 'n': x_pos.size}

# --- Run for all islands ---
results = []
for island in islands:
    res = process_island(island)
    if res:
        results.append(res)

# --- Summary table ---
print("\n\n=== Summary ===")
print(f"{'Island':<10} {'n':>8} {'k (exponent)':>13} {'Log-Log R2':>11} {'Linear R2':>10} {'Spearman rho':>13}")
for r in results:
    print(f"{r['island'].capitalize():<10} {r['n']:>8} {r['k']:>13.4f} {r['log_log_r2']:>11.4f} "
          f"{r['linear_r2']:>10.4f} {r['rho']:>13.4f}")