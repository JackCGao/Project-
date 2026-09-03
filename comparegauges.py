import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from maxflux_stream_points import OUTPUT_DIR as MAXFLUX_OUTPUT_DIR

MAXFLUX_POINTS_PATH = MAXFLUX_OUTPUT_DIR + 'all_islands_maxflux_stream_points.shp'


def match_gages_to_raster(gpkg_path, output_path=None, max_dist_m=1000):
    """
    Match gage stations to the nearest point in maxflux_stream_points.py's
    combined output (watershed_outputs/all_islands_maxflux_stream_points.shp
    -- D8 max-flux sampled at stream-pixel points, already restricted to
    each island's own stream mask and within 50 px of the statewide
    hydrography layer) using geopandas sjoin_nearest. Both inputs must be in
    the same projected CRS.
    """

    # 1. Load gage stations
    gdf = gpd.read_file(gpkg_path)
    print(f"Loaded {len(gdf)} gage stations")

    # 2. Load maxflux stream points
    points_gdf = gpd.read_file(MAXFLUX_POINTS_PATH)
    print(f"Loaded {len(points_gdf)} maxflux stream points")

    # 3. CRS checks
    if gdf.crs != points_gdf.crs:
        raise ValueError(
            f"CRS mismatch:\n"
            f"  Gages:  {gdf.crs}\n"
            f"  Points: {points_gdf.crs}\n"
            f"Reproject one to match the other before running."
        )
    if points_gdf.crs.is_geographic:
        raise ValueError(
            f"CRS is geographic ({points_gdf.crs}). Both inputs must be in a projected CRS."
        )

    # 4. Spatial nearest join
    matched = gpd.sjoin_nearest(
        gdf,
        points_gdf[['maxflux', 'island', 'geometry']],
        how='left',
        max_distance=max_dist_m,
        distance_col='dist_to_px'
    )
    matched = matched.rename(columns={'maxflux': 'raster_Q'})

    # 5. Clean up duplicates
    matched = matched.drop(columns=['index_right', 'fid'], errors='ignore')
    matched = matched.drop_duplicates(subset='site_no', keep='first')
    matched = matched.reset_index(drop=True)
    matched['island'] = matched['island'].str.capitalize()

    valid_count = matched['raster_Q'].notna().sum()
    print(f"Matched {valid_count}/{len(matched)} gages within {max_dist_m}m")

    # 6. Save
    if output_path:
        if output_path.endswith('.csv'):
            matched.drop(columns='geometry').to_csv(output_path, index=False)
        elif output_path.endswith('.gpkg'):
            matched.to_file(output_path, driver='GPKG')
        elif output_path.endswith('.shp'):
            matched.to_file(output_path)
        print(f"Saved to: {output_path}")

    return matched

def plot_comparison(gdf, output_path, raster_col='raster_Q', measured_col='mean_Q_m3y'):
    """
    X-Y comparison plot of measured vs raster discharge on log-log axes,
    against a 1:1 reference line -- raster_Q is a flow-accumulation-based
    proxy, so this shows how far off absolute agreement it is rather than
    fitting a trend to it.
    """
    valid = gdf.dropna(subset=[raster_col, measured_col, 'island'])
    x = valid[measured_col].values
    y = valid[raster_col].values

    # Log-scale
    pos = (x > 0) & (y > 0)
    x, y = x[pos], y[pos]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=20, alpha=0.6, edgecolors='k', linewidth=0.3, color='#1f77b4')

    rho, _ = stats.spearmanr(x, y)

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color='black', linewidth=1.8,
            linestyle='--', label='1:1 line')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r'$Q_{\text{gage}}$ (m$^3$/yr)')
    ax.set_ylabel(r'$Q_{\text{model}}$ (m$^3$/yr)')
    ax.legend(fontsize=8, markerscale=1.2, loc='upper left')

    stats_text = f"Spearman's rho = {rho:.4f}"
    ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    print(f"Plotted {pos.sum()} points")


def plot_comparison_linear(gdf, output_path, raster_col='raster_Q', measured_col='mean_Q_m3y'):
    """
    Same measured-vs-raster comparison as plot_comparison(), but on linear
    (not log-log) axes, so the linear OLS fit actually renders as a
    straight line.
    """
    valid = gdf.dropna(subset=[raster_col, measured_col, 'island'])
    x = valid[measured_col].values
    y = valid[raster_col].values

    # drop anomalous negative-measured-discharge gages (data artifacts, not
    # physically meaningful -- e.g. tidally-influenced or diversion sites)
    pos = x > 0
    x, y = x[pos], y[pos]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=20, alpha=0.6, edgecolors='k', linewidth=0.3, color='#1f77b4')

    m_lin, b_lin, r_lin, _, _ = stats.linregress(x, y)
    r2_lin = r_lin ** 2
    rho, _ = stats.spearmanr(x, y)

    x_fit = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_fit, m_lin * x_fit + b_lin, color='#3a6ea5', linewidth=1.8,
            linestyle='--', label='linear OLS fit')

    ax.set_xlabel(r'$Q_{\text{gage}}$ (m$^3$/yr)')
    ax.set_ylabel(r'$Q_{\text{model}}$ (m$^3$/yr)')
    ax.legend(fontsize=8, markerscale=1.2, loc='upper left')

    stats_text = (f"Spearman's rho = {rho:.4f}\n"
                  f"Linear R2 = {r2_lin:.4f}\n"
                  f"slope = {m_lin:.4g}, intercept = {b_lin:.4g}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    print(f"Plotted {len(x)} points (linear scale)")


if __name__ == '__main__':
    output_dir = '/Users/jackgao/Summer Work 2026/Temp Output Placements'
    output_path = f'{output_dir}/gage_raster_comparison.png'

    print("Matching gages to nearest maxflux stream point...")
    gdf = match_gages_to_raster(
        gpkg_path='HI_gages_discharge_daily_albers.shp',
        output_path=f'{output_dir}/HI_gages_matched.gpkg',
        max_dist_m=500
    )

    print("Plotting comparison...")
    plot_comparison(gdf, output_path)
    plot_comparison_linear(gdf, f'{output_dir}/gage_raster_comparison_linear.png')