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

def compute_and_save_net_precip_accum(island):
    """
    Build a flow-accumulation raster weighted by net surface water using
    WhiteboxTools d8_mass_flux:
        loading    = precipitation (m³/yr per cell)
        absorption = groundwater recharge (m³/yr per cell)
        efficiency = 1.0 (all non-absorbed water routes downstream)
    S1_Rain is in in/yr (×0.0254 → m/yr); S1_Tot_rc is in in/yr (×0.0254 → m/yr).
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
        precip_rast = rasterize_precipitation(dem_path, island)  # in/yr (from shapefile)

    recharge_rast = rasterize_recharge(dem_path, island)  # in/yr

    cell_m2 = abs(ref_transform.a) * abs(ref_transform.e)

    # Convert to m³/yr per cell (zero out nodata so WBT sees no absorption where data is missing)
    precip_arr   = np.where(np.isfinite(precip_rast),   precip_rast   * 0.0254 * cell_m2, 0.0)
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