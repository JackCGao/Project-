import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
import os
from scipy import stats

ISLAND_COLORS = {
    'Hawaii':    '#d62728',
    'Kahoolawe': '#9467bd',
    'Oahu':      '#2ca02c',
    'Kauai':     '#1f77b4',
    'Lanai':     '#8c564b',
    'Molokai':   '#e377c2',
    'Maui':      '#ff7f0e',
}

def merge_rasters_vrt(raster_paths, output_path):
    """
    Uses GDAL to create a virtual dataset (VRT) that references all the input rasters.
    This acts like one giant file but uses very little RAM, and avoids 'Too many
    open files' errors.

    output_path should end in .vrt -- this stays a small XML file that
    references the source rasters (no data is copied/materialized), so
    reading it just reads windows from the underlying files on demand.
    Previously this used driver.CreateCopy() to bake the VRT into a real
    GeoTIFF, but since the 7 islands' combined bounding box spans the whole
    Hawaiian chain (~528km x 373km at ~9.8m resolution, ~2 billion pixels,
    mostly nodata ocean between islands), that materialized an ~8GB+ file
    for almost no benefit -- writing it out is what was timing out.

    Parameters:
    raster_paths (list of str): List of file paths to the input rasters to be merged.
    output_path (str): File path for the output .vrt file.

    Returns:
    None
    """
    vrt_ds = gdal.BuildVRT(output_path, raster_paths)
    vrt_ds = None

def extract_stream_pixels(flowaccum_path, streams_path):
    """
    Load one island's flow-accumulation raster and its companion binary
    streams mask (1 = stream, nodata elsewhere), and return the x/y/value
    of only the pixels that fall on the actual stream network.

    The two rasters must share the same grid (shape + transform) -- true
    for all 7 islands here (same source DEM). Restricting to the streams
    mask is what keeps the pixel count small (thousands, not hundreds of
    millions): without it, ~every land pixel in a flow-accumulation raster
    has *some* nonzero value, so a plain isnan/nodata check treats nearly
    the whole landmass as "stream."
    """
    with rasterio.open(flowaccum_path) as fa_src, rasterio.open(streams_path) as st_src:
        if fa_src.shape != st_src.shape or fa_src.transform != st_src.transform:
            raise ValueError(
                f"Grid mismatch between {flowaccum_path} and {streams_path}: "
                f"{fa_src.shape}/{fa_src.transform} vs {st_src.shape}/{st_src.transform}"
            )
        flow = fa_src.read(1)
        streams = st_src.read(1)
        transform = fa_src.transform
        crs = fa_src.crs

        on_stream = (streams == 1)
        valid_mask = on_stream & np.isfinite(flow) & (flow > 0)
        fa_nodata = fa_src.nodata
        if fa_nodata is not None and not np.isnan(fa_nodata):
            valid_mask &= (flow != fa_nodata)

        rows, cols = np.where(valid_mask)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        values = flow[rows, cols]  # already in m³/yr -- no unit conversion needed

    return xs, ys, values, crs


def match_gages_to_raster(gpkg_path, raster_stream_pairs, output_path=None, max_dist_m=1000):
    """
    Match gage stations to nearest stream pixel using geopandas sjoin_nearest.
    Both inputs must be in the same projected CRS.

    raster_stream_pairs: list of (island_name, flowaccum_path, streams_path)
    tuples, one per island. Each island is read and masked independently
    (rather than merging all islands into one giant raster first) since the
    combined bounding box across the whole Hawaiian chain is mostly nodata
    ocean. island_name is carried through to the matched output so points
    can be colored/grouped by island downstream.
    """

    # 1. Load gage stations
    gdf = gpd.read_file(gpkg_path)
    print(f"Loaded {len(gdf)} gage stations")

    # 2. Extract stream-masked pixels from every island, one at a time
    all_xs, all_ys, all_values, all_islands = [], [], [], []
    crs_raster = None
    for island_name, flowaccum_path, streams_path in raster_stream_pairs:
        xs, ys, values, crs = extract_stream_pixels(flowaccum_path, streams_path)
        if crs_raster is None:
            crs_raster = crs
        elif crs_raster != crs:
            raise ValueError(
                f"CRS mismatch between island rasters:\n"
                f"  {raster_stream_pairs[0][1]}: {crs_raster}\n"
                f"  {flowaccum_path}: {crs}"
            )
        all_xs.extend(xs)
        all_ys.extend(ys)
        all_values.extend(values)
        all_islands.extend([island_name] * len(xs))
        print(f"  {os.path.basename(flowaccum_path)}: {len(xs)} stream pixels")

    xs, ys, values, islands = all_xs, all_ys, all_values, all_islands
    print(f"Found {len(xs)} valid stream pixels total")

    # 3. CRS checks
    if gdf.crs != crs_raster:
        raise ValueError(
            f"CRS mismatch:\n"
            f"  Gages:  {gdf.crs}\n"
            f"  Raster: {crs_raster}\n"
            f"Reproject one to match the other before running."
        )
    if crs_raster.is_geographic:
        raise ValueError(
            f"CRS is geographic ({crs_raster}). Both inputs must be in a projected CRS."
        )

    # 4. Build GeoDataFrame of valid pixels
    pixel_gdf = gpd.GeoDataFrame(
        {'raster_Q': values, 'island': islands},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs_raster
    )

    # 5. Spatial nearest join
    matched = gpd.sjoin_nearest(
        gdf,
        pixel_gdf[['raster_Q', 'island', 'geometry']],
        how='left',
        max_distance=max_dist_m,
        distance_col='dist_to_px'
    )
    
    # 6. Clean up duplicates
    matched = matched.drop(columns=['index_right', 'fid'], errors='ignore')
    matched = matched.drop_duplicates(subset='site_no', keep='first')
    matched = matched.reset_index(drop=True)

    valid_count = matched['raster_Q'].notna().sum()
    print(f"Matched {valid_count}/{len(matched)} gages within {max_dist_m}m")
    
    # 7. Save
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
    X-Y comparison plot of measured vs raster discharge, colored by island,
    with a log-log OLS regression fit (same style as recreation.py's Plot 1)
    in place of a 1:1 line -- raster_Q is a flow-accumulation-based proxy,
    not a claim of absolute agreement with measured discharge.
    """
    valid = gdf.dropna(subset=[raster_col, measured_col, 'island'])
    x = valid[measured_col].values
    y = valid[raster_col].values
    islands = valid['island'].values

    # Log-scale
    pos = (x > 0) & (y > 0)
    x, y, islands = x[pos], y[pos], islands[pos]

    fig, ax = plt.subplots(figsize=(6, 5))
    for island in np.unique(islands):
        mask = islands == island
        ax.scatter(x[mask], y[mask], s=20, alpha=0.6, edgecolors='k',
                   linewidth=0.3, color=ISLAND_COLORS.get(island, '#888888'),
                   label=island)

    # add stats
    log_x, log_y = np.log10(x), np.log10(y)

    m, b, r, _, _ = stats.linregress(log_x, log_y)
    r2_log = r ** 2
    rho, _ = stats.spearmanr(x, y)

    x_fit = np.logspace(log_x.min(), log_x.max(), 200)
    ax.plot(x_fit, (10 ** b) * (x_fit ** m), color='black', linewidth=1.8,
            linestyle='--', label='log-log OLS fit')

    # regular (linear-space) OLS fit, plotted over the same log-log axes --
    # appears as a curve here since it's a straight line in linear space
    m_lin, b_lin, r_lin, _, _ = stats.linregress(x, y)
    r2_lin = r_lin ** 2

    x_fit_lin = np.linspace(x.min(), x.max(), 200)
    y_fit_lin = m_lin * x_fit_lin + b_lin
    lin_plot_mask = y_fit_lin > 0  # log-scale y-axis can't show <=0 values
    ax.plot(x_fit_lin[lin_plot_mask], y_fit_lin[lin_plot_mask], color='#3a6ea5',
            linewidth=1.8, linestyle=':', label='linear OLS fit')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$Q_{\text{gage}}$ (m$^3$/yr)')
    ax.set_ylabel(r'$Q_{\text{model}}$ (m$^3$/yr)')
    ax.legend(fontsize=8, markerscale=1.2, loc='upper left')

    stats_text = (f"Spearman's rho = {rho:.4f}\n"
                  f"Log-Log R2 = {r2_log:.4f}\n"
                  f"Linear R2 = {r2_lin:.4f}")
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
    islands = valid['island'].values

    # drop anomalous negative-measured-discharge gages (data artifacts, not
    # physically meaningful -- e.g. tidally-influenced or diversion sites)
    pos = x > 0
    x, y, islands = x[pos], y[pos], islands[pos]

    fig, ax = plt.subplots(figsize=(6, 5))
    for island in np.unique(islands):
        mask = islands == island
        ax.scatter(x[mask], y[mask], s=20, alpha=0.6, edgecolors='k',
                   linewidth=0.3, color=ISLAND_COLORS.get(island, '#888888'),
                   label=island)

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
    output_path = 'gage_raster_comparison.png'

    # (island name, flow-accumulation raster, companion binary streams mask)
    # per island -- both rasters share the same grid for a given island.
    raster_stream_pairs = [
        ('Hawaii', '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/new (1)/hawaii_d8maxflux_nans.tif',
         '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/new (1)/hawaii_streams_unweighted_albers.tif'),
        ('Maui', '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/maui/new/maui_d8maxflux_nans.tif',
         '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/maui/new/maui_streams_unweighted_albers.tif'),
        ('Lanai', '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/lanai/new/lanai_d8maxflux_nans.tif',
         '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/lanai/new/lanai_streams_unweighted_albers.tif'),
        ('Kahoolawe', '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/new/kahoolawe_d8maxflux_nans.tif',
         '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/new/kahoolawe_streams_unweighted_albers.tif'),
        ('Kauai', '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/kauai/new/kauai_d8maxflux_nans.tif',
         '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/kauai/new/kauai_streams_unweighted_albers.tif'),
        ('Oahu', '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/oahu/new/oahu_d8maxflux_nans.tif',
         '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/oahu/new/oahu_streams_unweighted_albers.tif'),
        ('Molokai', '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/molokai/new/molokai_d8maxflux_nans.tif',
         '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/molokai/new/molokai_streams_unweighted_albers.tif'),
    ]

    print("Matching gages to raster...")
    gdf = match_gages_to_raster(
        gpkg_path='HI_gages_discharge_daily_albers.shp',
        raster_stream_pairs=raster_stream_pairs,
        output_path='HI_gages_matched.gpkg',
        max_dist_m=500
    )

    print("Plotting comparison...")
    plot_comparison(gdf, output_path)
    plot_comparison_linear(gdf, 'gage_raster_comparison_linear.png')