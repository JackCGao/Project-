import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rasterio')

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import geopandas as gpd
import pandas as pd
from rasterio.features import rasterize as rio_rasterize
from rasterio.warp import reproject, Resampling
import whitebox_workflows as wbw
import seaborn as sns
import os

base_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao'
output_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Temp Output Placements'

wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = output_dir + '/'

AGE_SHP = (f'{base_dir}/Jack, Ze-Wen summer project files/'
           f'Haw_St_shapefiles/Haw_St_geo_20070426_region_with_age.shp')

_age_gdf = None

def _load_age_shp():
    global _age_gdf
    if _age_gdf is None:
        gdf = gpd.read_file(AGE_SHP)
        gdf['avg_age'] = (
            pd.to_numeric(gdf['min_age_yr'], errors='coerce') +
            pd.to_numeric(gdf['max_age_yr'], errors='coerce')
        ) / 2.0
        _age_gdf = gdf[gdf['avg_age'] > 0].copy()
    return _age_gdf

def rasterize_age(ref_path, island_name):
    """Rasterize average (min+max)/2 geological age onto the reference raster grid."""
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs       = src.crs
        width     = src.width
        height    = src.height

    gdf = _load_age_shp()
    island_gdf = gdf[gdf['ISLAND'] == island_name.capitalize()].copy()
    if island_gdf.empty:
        return np.full((height, width), np.nan)

    island_gdf = island_gdf.to_crs(crs)

    shapes = (
        (geom, val)
        for geom, val in zip(island_gdf.geometry, island_gdf['avg_age'])
        if geom is not None and not geom.is_empty
    )

    arr = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype=np.float64,
    )
    return arr

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

PRECIP_FIELD = 'S1_RF'   # rainfall field in the water-budget shapefiles (mm/yr)

_recharge_gdfs = {}

def _load_recharge_shp(island):
    if island not in _recharge_gdfs:
        paths = RECHARGE_SHPS[island]
        if isinstance(paths, str):
            paths = [paths]

        parts = []
        for path in paths:
            gdf = gpd.read_file(path)
            cols = ['geometry', 'S1_Tot_rc']
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

def rasterize_recharge(ref_path, island):
    """Rasterize S1_Tot_rc (baseline groundwater recharge, in/yr) onto the reference raster grid."""
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs       = src.crs
        width     = src.width
        height    = src.height

    if island not in RECHARGE_SHPS:
        return np.full((height, width), np.nan)

    gdf = _load_recharge_shp(island).to_crs(crs)

    shapes = (
        (geom, val)
        for geom, val in zip(gdf.geometry, gdf['S1_Tot_rc'])
        if geom is not None and not geom.is_empty
    )

    arr = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype=np.float64,
    )
    return arr

def rasterize_precipitation(ref_path, island):
    """Rasterize S1_RF (rainfall, in/yr) onto the reference raster grid."""
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

def compute_and_save_net_precip_accum(island):
    """
    Build a flow-accumulation raster weighted by net surface water using
    WhiteboxTools d8_mass_flux:
        loading    = precipitation (m³/yr per cell)
        absorption = groundwater recharge (m³/yr per cell)
        efficiency = 1.0 (all non-absorbed water routes downstream)
    S1_RF is in mm/yr (×0.001 → m/yr); S1_Tot_rc is in in/yr (×0.0254 → m/yr).
    Output saved to: <output_dir>/<island>_precip_net_accum.tif
    """
    out_path = island_files[island]['mass_flux']
    if os.path.exists(out_path):
        print(f"Using existing {out_path}")
        return

    dem_path   = island_files[island]['dem']
    precip_key = island_files[island].get('precip')

    with rasterio.open(dem_path) as src:
        ref_transform = src.transform
        ref_crs       = src.crs
        ref_h, ref_w  = src.height, src.width
        ref_profile   = src.profile.copy()

    if precip_key:
        with rasterio.open(precip_key) as psrc:
            precip_rast = np.empty((ref_h, ref_w), dtype=np.float64)
            reproject(
                source=rasterio.band(psrc, 1),
                destination=precip_rast,
                src_transform=psrc.transform,
                src_crs=psrc.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            nd = psrc.nodata
            if nd is not None:
                precip_rast[np.isclose(precip_rast, nd, rtol=0, atol=1e-6)] = np.nan
    else:
        precip_rast = rasterize_precipitation(dem_path, island)  # mm/yr (from shapefile)

    recharge_rast = rasterize_recharge(dem_path, island)  # in/yr

    cell_m2 = abs(ref_transform.a) * abs(ref_transform.e)

    # Convert to m³/yr per cell (zero out nodata so WBT sees no absorption where data is missing)
    precip_arr   = np.where(np.isfinite(precip_rast),   precip_rast   * 0.001  * cell_m2, 0.0)
    recharge_arr = np.where(np.isfinite(recharge_rast), recharge_rast * 0.0254 * cell_m2, 0.0)

    prof = ref_profile.copy()
    prof.update(dtype='float64', count=1, nodata=0.0)

    tmp_precip   = os.path.join(output_dir, f'_tmp_{island}_precip.tif')
    tmp_recharge = os.path.join(output_dir, f'_tmp_{island}_recharge.tif')
    tmp_eff      = os.path.join(output_dir, f'_tmp_{island}_efficiency.tif')

    for path, arr in [(tmp_precip,   precip_arr),
                      (tmp_recharge, recharge_arr),
                      (tmp_eff,      np.ones((ref_h, ref_w), dtype=np.float64))]:
        with rasterio.open(path, 'w', **prof) as dst:
            dst.write(arr, 1)

    dem_wbe     = wbe.read_raster(dem_path)
    loading_wbe = wbe.read_raster(tmp_precip)
    absorb_wbe  = wbe.read_raster(tmp_recharge)
    eff_wbe     = wbe.read_raster(tmp_eff)

    net_accum_wbe = wbe.hydrology.flow_routing.d8_mass_flux(
        dem=dem_wbe,
        loading=loading_wbe,
        efficiency=eff_wbe,
        absorption=absorb_wbe,
    )
    wbe.write_raster(net_accum_wbe, f'{island}_precip_net_accum.tif')

    for p in [tmp_precip, tmp_recharge, tmp_eff]:
        os.remove(p)

    print(f"Saved {out_path}")

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

_hawaii_dir    = f'{base_dir}/new (1)'
_kahoolawe_dir = f'{base_dir}/new'
_oahu_dir      = f'{base_dir}/oahu/new'

island_files = {
    'hawaii': {
        'flow_accum': f'{_hawaii_dir}/hawaii_d8maxflux_nans.tif',
        'slope':      f'{_hawaii_dir}/hawaii_slope_nans.tif',
        'erosion':    f'{_hawaii_dir}/hawaii_erosion_nans.tif',
        'dem':        f'{_hawaii_dir}/hawaii_dem_enforced_qgis_albers.tif',
        'flowdir':    f'{_hawaii_dir}/hawaii_dem_enforced_flowdir.tif',
        'precip':     f'{_hawaii_dir}/hawaii_precip_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/hawaii_precip_net_accum.tif',
    },
    'kahoolawe': {
        'flow_accum': f'{_kahoolawe_dir}/kahoolawe_d8maxflux_nans.tif',
        'slope':      f'{_kahoolawe_dir}/kahoolawe_slope_nans.tif',
        'erosion':    f'{_kahoolawe_dir}/kahoolawe_erosion_nans.tif',
        'dem':        f'{_kahoolawe_dir}/kahoolawe_dem_enforced_qgis_albers.tif',
        'flowdir':    f'{_kahoolawe_dir}/kahoolawe_dem_enforced_flowdir.tif',
        'precip':     f'{_kahoolawe_dir}/kahoolawe_precip_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/kahoolawe_precip_net_accum.tif',
    },
    'oahu': {
        'flow_accum': f'{_oahu_dir}/oahu_d8maxflux_nans.tif',
        'slope':      f'{_oahu_dir}/oahu_slope_nans.tif',
        'erosion':    f'{_oahu_dir}/oahu_erosion_nans.tif',
        'dem':        f'{_oahu_dir}/oahu_dem_enforced_qgis_albers.tif',
        'flowdir':    f'{_oahu_dir}/oahu_dem_enforced_flowdir.tif',
        'precip':     f'{_oahu_dir}/oahu_precip_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/oahu_precip_net_accum.tif',
    },
    'kauai': {
        'flow_accum': f'{base_dir}/kauai/new/kauai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/kauai/new/kauai_slope.tif',
        'erosion':    f'{base_dir}/kauai/new/kauai_erosion_nans.tif',
        'dem':        f'{base_dir}/kauai/new/kauai_dem_enforced_qgis_albers.tif',
        'flowdir':    f'{base_dir}/kauai/new/kauai_dem_enforced_flowdir.tif',
        'precip':     f'{base_dir}/kauai/new/kauai_precip_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/kauai_precip_net_accum.tif',
    },
    'lanai': {
        'flow_accum': f'{base_dir}/lanai/new/lanai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/lanai/new/lanai_slope.tif',
        'erosion':    f'{base_dir}/lanai/new/lanai_erosion_nans.tif',
        'dem':        f'{base_dir}/lanai/new/lanai_dem_enforced_qgis_albers.tif',
        'flowdir':    f'{base_dir}/lanai/new/lanai_dem_enforced_flowdir.tif',
        'precip':     f'{base_dir}/lanai/new/lanai_precip_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/lanai_precip_net_accum.tif',
    },
    'molokai': {
        'flow_accum': f'{base_dir}/molokai/new/molokai_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/molokai/new/molokai_slope_nans.tif',
        'erosion':    f'{base_dir}/molokai/new/molokai_erosion_nans.tif',
        'dem':        f'{base_dir}/molokai/new/molokai_dem_enforced_qgis_albers.tif',
        'flowdir':    f'{base_dir}/molokai/new/molokai_dem_enforced_flowdir.tif',
        'precip':     f'{base_dir}/molokai/new/molokai_precip_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/molokai_precip_net_accum.tif',
    },
    'maui': {
        'flow_accum': f'{base_dir}/maui/new/maui_d8maxflux_nans.tif',
        'slope':      f'{base_dir}/maui/new/maui_slope_nans.tif',
        'erosion':    f'{base_dir}/maui/new/maui_erosion_nans.tif',
        'dem':        f'{base_dir}/maui/new/maui_dem_enforced_qgis_albers.tif',
        'flowdir':    f'{base_dir}/maui/new/maui_dem_enforced_flowdir.tif',
        'precip':     f'{base_dir}/maui/new/maui_precip_qgis_albers.tif',
        'mass_flux':  f'{output_dir}/maui_precip_net_accum.tif',
    },
}
islands = list(island_files.keys())

def load_clean(path):
    with rasterio.open(path) as src:
        arr = src.read([1])[0].astype(np.float64, copy=False)
        nd = src.nodata
        if nd is not None:
            if np.isnan(nd):
                arr[~np.isfinite(arr)] = np.nan
            else:
                arr[np.isclose(arr, nd, rtol=0, atol=1e-6)] = np.nan
        return arr, src.crs, src.transform, src.shape

def process_island(island):
    flow_accum_path  = island_files[island]['flow_accum']
    slopes_path      = island_files[island]['slope']
    erosion_path     = island_files[island]['erosion']
    dem_path         = island_files[island]['dem']
    mass_flux_path   = island_files[island]['mass_flux']

    flow_accum, fa_crs, fa_transform, fa_shape = load_clean(flow_accum_path)
    slope_rast, sl_crs, _,            _        = load_clean(slopes_path)
    erosion,    er_crs, _,            _        = load_clean(erosion_path)
    dem,        _,      _,            _        = load_clean(dem_path)
    mass_flux,  _,      _,            _        = load_clean(mass_flux_path)

    # --- Mask on intersection of all four valid-data footprints, elevation > 1 m ---
    valid_mask = (np.isfinite(flow_accum) & np.isfinite(slope_rast) &
                  np.isfinite(erosion) & (dem > 1))

    final_valid = valid_mask & np.isfinite(flow_accum * slope_rast)

    Q = flow_accum[final_valid]
    S = slope_rast[final_valid]
    E = erosion[final_valid]

    pos_mask = (Q > 0) & (S > 0) & (E >= 1)
    Q_pos = Q[pos_mask]
    S_pos = S[pos_mask]
    E_pos = E[pos_mask]

    QS_pos = Q_pos * S_pos

    # --- OLS in log space: log10(E) = slope*log10(QS) + b  =>  E = 10^b * QS^slope ---
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
    ax.set_ylabel('Erosional Depth (E) [m]')
    ax.set_title(island.capitalize())
    textstr = (f"Spearman's rank correlation = {rho:.4f}\n"
               f"Log–Log R² = {r2_1:.4f}")
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{island}_E_vs_QS.png'), dpi=200)
    plt.close(fig)

    # --- Fit 2: E = Q^c * S  =>  log10(E/S) = c*log10(Q) + b2  =>  E/S = 10^b2 * Q^c ---
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
    ax.set_ylabel(r'$E \,/\, S$ [m]')
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
    ax.set_ylabel(r'$E \,/\, S$ [m]')
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
    age_rast = rasterize_age(flow_accum_path, island)
    age_flat = age_rast[final_valid][pos_mask]       # same masking as Q_pos/E_pos

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
        ax.set_ylabel(r'Log-Log Residual  $\log_{10}(E_\mathrm{obs}/E_\mathrm{pred})$')
        ax.set_title(island.capitalize())
        rho_age, _ = stats.spearmanr(age_ma, resid_sub)
        ax.text(0.05, 0.95, f"Spearman ρ = {rho_age:.4f}  (n = {age_valid.sum():,})",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{island}_resid_vs_age.png'), dpi=200)
        plt.close(fig)
    
    # --- mass_flux: D8MassFlux output (precip − recharge routed downstream) ---
    mass_flux_flat = mass_flux[final_valid][pos_mask]

    # --- Plot 5: log-log residuals vs local groundwater recharge (S1_Tot_rc, m/yr) ---
    recharge_rast = rasterize_recharge(flow_accum_path, island)   # in/yr
    recharge_m    = recharge_rast * 0.0254                        # in/yr → m/yr
    recharge_flat = recharge_m[final_valid][pos_mask]

    rc_all = recharge_m[final_valid]
    rc_all = rc_all[np.isfinite(rc_all)]
    if rc_all.size > 0:
        print(f"[{island}] Recharge (m/yr):  mean={rc_all.mean():.4f}  "
              f"min={rc_all.min():.4f}  max={rc_all.max():.4f}")

    rc_valid = np.isfinite(recharge_flat) & (recharge_flat > 0)
    if rc_valid.sum() > 0:
        rc_sub    = recharge_flat[rc_valid]
        resid_rc  = log_resid[rc_valid]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(rc_sub, resid_rc, s=3, alpha=0.115, edgecolors='none', color='#888888')
        ax.axhline(0, color='#e06b3a', linewidth=1.2, linestyle='--')
        ax.set_xscale('log')
        ax.set_xlim(*_log_lim(rc_sub))
        ax.set_ylim(resid_rc.min(), resid_rc.max())
        ax.set_xlabel('Groundwater Recharge — S1_Tot_rc (m/yr)')
        ax.set_ylabel(r'Log-Log Residual  $\log_{10}(E_\mathrm{obs}/E_\mathrm{pred})$')
        ax.set_title(island.capitalize())
        rho_rc, _ = stats.spearmanr(rc_sub, resid_rc)
        ax.text(0.05, 0.95, f"Spearman ρ = {rho_rc:.4f}  (n = {rc_valid.sum():,})",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{island}_resid_vs_recharge.png'), dpi=200)
        plt.close(fig)

    # --- Plots 6 & 7: power law using D8MassFlux directly as discharge proxy ---
    mf_mask = np.isfinite(mass_flux_flat) & (mass_flux_flat > 0)
    if mf_mask.sum() > 10:
        Q_net = mass_flux_flat[mf_mask]
        E_net = E_pos[mf_mask]
        S_net = S_pos[mf_mask]

        QS_net = Q_net * S_net

        # --- Fit 6: log10(E) = m6*log10(Q_net*S) + b6 ---
        log_QN = np.log10(QS_net)
        log_E6 = np.log10(E_net)
        m6, b6, r6, _, _ = stats.linregress(log_QN, log_E6)
        r2_6 = r6 ** 2
        rho6, _ = stats.spearmanr(QS_net, E_net)

        x_fit6 = np.logspace(np.log10(QS_net.min()), np.log10(QS_net.max()), 200)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(QS_net, E_net, s=3, alpha=0.115, edgecolors='none', color='#888888')
        _kde_plot(ax, QS_net, E_net, log_scale=True)
        ax.plot(x_fit6, (10**b6) * (x_fit6 ** m6), color='#3a6ea5', linewidth=1.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(*_log_lim(QS_net))
        ax.set_ylim(*_log_lim(E_net))
        ax.minorticks_on()
        ax.set_xlabel(r'Stream Power Proxy ($Q_\mathrm{net} \times S$, Recharge Accounted)')
        ax.set_ylabel('Erosional Depth (E) [m]')
        ax.set_title(island.capitalize())
        textstr6 = (f"Spearman's rank correlation = {rho6:.4f}\n"
                    f"Log–Log R² = {r2_6:.4f}")
        ax.text(0.05, 0.05, textstr6, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{island}_E_vs_QR.png'), dpi=200)
        plt.close(fig)

        # --- Fit 7: log10(E/S) = c7*log10(Q_net) + b7 ---
        EN_net = E_net / S_net
        log_Q7 = np.log10(Q_net)
        log_EN = np.log10(EN_net)
        c7, b7, r7, _, _ = stats.linregress(log_Q7, log_EN)
        r2_7 = r7 ** 2
        rho7, _ = stats.spearmanr(Q_net, EN_net)

        x_fit7 = np.logspace(np.log10(Q_net.min()), np.log10(Q_net.max()), 200)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(Q_net, EN_net, s=3, alpha=0.115, edgecolors='none', color='#888888')
        _kde_plot(ax, Q_net, EN_net, log_scale=True)
        ax.plot(x_fit7, (10**b7) * (x_fit7 ** c7), color='#3a6ea5', linewidth=1.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(*_log_lim(Q_net))
        ax.set_ylim(*_log_lim(EN_net))
        ax.minorticks_on()
        ax.set_xlabel(r'Net Discharge Proxy ($Q_\mathrm{net}$, Recharge Accounted)')
        ax.set_ylabel(r'$E \,/\, S$  [m]')
        ax.set_title(island.capitalize())
        textstr7 = (f"Spearman's rank correlation = {rho7:.4f}\n"
                    f"Log–Log R² = {r2_7:.4f}\n"
                    f"c = {c7:.4f},  k = 10^{b7:.4f} = {10**b7:.4g}")
        ax.text(0.05, 0.05, textstr7, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.99, edgecolor='gray'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{island}_E_vs_QcR.png'), dpi=200)
        plt.close(fig)

    return {'island': island,
            'slope': m, 'intercept': b, 'r2_QS': r2_1, 'rho': rho,
            'c': c, 'r2_QcS': r2_2, 'rho2': rho2,
            'c_sp': c_sp, 'k_sp': k_sp, 'rho_sp': rho_sp,
            'c_sp_unreliable': c_sp_unreliable,
            'n': Q_pos.size}

# --- Run for all islands ---
results = []
for island in islands:
    compute_and_save_net_precip_accum(island)   # generate mass_flux file if not already present
    res = process_island(island)
    if res:
        results.append(res)