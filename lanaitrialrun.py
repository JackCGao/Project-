import os
import whitebox_workflows as wbw
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize as rio_rasterize, shapes
from shapely.geometry import Point, shape

wbe = wbw.WbEnvironment()
wbe.verbose = True
wbe.working_directory = '/Users/jackgao/Summer Work 2026/Project-/'

# --- Inputs ---
project_dir = wbe.working_directory
dem_path    = project_dir + 'Lanai_DEM_masked.tif'
outlet_path = project_dir + 'outlet_point.shp'
streams_threshold = 500

# --- 1. Read DEM ---
dem = wbe.read_raster(dem_path)

# --- 2 & 3. Fill missing data + smooth ---
if os.path.exists(project_dir + 'dem_smooth.tif'):
    print("Using existing dem_smooth.tif")
    dem = wbe.read_raster(project_dir + 'dem_smooth.tif')
else:
    dem = wbe.terrain.general.fill_missing_data(raster=dem, filter_size=11, exclude_edge_nodata=True)
    dem = wbe.terrain.general.feature_preserving_smoothing(raster=dem, filter_size=11, num_iter=2)
    wbe.write_raster(dem, 'dem_smooth.tif')

# --- 4. Condition DEM ---
if os.path.exists(project_dir + 'dem_conditioned.tif'):
    print("Using existing dem_conditioned.tif")
    dem_cond = wbe.read_raster(project_dir + 'dem_conditioned.tif')
else:
    dem_cond = wbe.hydrology.depressions_storage.breach_depressions_least_cost(
        dem=dem, dist=50, fill_deps=True
    )
    wbe.write_raster(dem_cond, 'dem_conditioned.tif')

# --- 5. Flow direction ---
if os.path.exists(project_dir + 'd8_pointer.tif'):
    print("Using existing d8_pointer.tif")
    d8_pntr = wbe.read_raster(project_dir + 'd8_pointer.tif')
else:
    d8_pntr   = wbe.hydrology.flow_routing.d8_pointer(dem=dem_cond)
    dinf_pntr = wbe.hydrology.flow_routing.dinf_pointer(dem=dem_cond)
    wbe.write_raster(d8_pntr, 'd8_pointer.tif')

# --- 6. Flow accumulation ---
if os.path.exists(project_dir + 'flow_accum.tif'):
    print("Using existing flow_accum.tif")
    flow_accum = wbe.read_raster(project_dir + 'flow_accum.tif')
else:
    flow_accum = wbe.hydrology.flow_routing.d8_flow_accum(
        input=d8_pntr, out_type='cells', input_is_pointer=True
    )
    wbe.write_raster(flow_accum, 'flow_accum.tif')

# --- 6b. Precipitation-net flow accumulation via d8_mass_flux ---
# loading=precipitation, absorption=recharge, efficiency=1 (all non-absorbed water routes downstream)
_net_accum_path = project_dir + 'lanai_precip_net_accum.tif'
if os.path.exists(_net_accum_path):
    print(f"Using existing {_net_accum_path}")
else:
    _recharge_base = ('/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/'
                      'Jack, Ze-Wen summer project files/Groundwater Recharge Files')
    lanai_wb_shp = (f'{_recharge_base}/Lanai Water Budget Components 2020/'
                    'Lanai_water_budget_components_subarea_inches.shp')

    with rasterio.open(project_dir + 'dem_conditioned.tif') as _src:
        _ref_transform = _src.transform
        _ref_crs       = _src.crs
        _ref_h, _ref_w = _src.height, _src.width
        _ref_profile   = _src.profile.copy()

    _wb_gdf = gpd.read_file(lanai_wb_shp).to_crs(_ref_crs)

    def _rasterize_field(gdf, field, height, width, transform):
        _shapes = (
            (geom, val)
            for geom, val in zip(gdf.geometry, gdf[field])
            if geom is not None and not geom.is_empty
        )
        return rio_rasterize(_shapes, out_shape=(height, width),
                             transform=transform, fill=0.0, dtype=np.float64)

    # S1_RF in mm/yr (×0.001 → m/yr); S1_Tot_rc in in/yr (×0.0254 → m/yr); both ×cell_area → m³/yr
    _cell_m2      = abs(_ref_transform.a) * abs(_ref_transform.e)
    _precip_arr   = _rasterize_field(_wb_gdf, 'S1_RF',     _ref_h, _ref_w, _ref_transform) * 0.001  * _cell_m2
    _recharge_arr = _rasterize_field(_wb_gdf, 'S1_Tot_rc', _ref_h, _ref_w, _ref_transform) * 0.0254 * _cell_m2

    # Write temp rasters so WBW can read them
    _prof = _ref_profile.copy()
    _prof.update(dtype='float64', count=1, nodata=0.0)
    _tmp_precip   = project_dir + '_tmp_precip.tif'
    _tmp_recharge = project_dir + '_tmp_recharge.tif'
    _tmp_eff      = project_dir + '_tmp_efficiency.tif'
    for _path, _arr in [(_tmp_precip, _precip_arr),
                         (_tmp_recharge, _recharge_arr),
                         (_tmp_eff, np.ones((_ref_h, _ref_w), dtype=np.float64))]:
        with rasterio.open(_path, 'w', **_prof) as _dst:
            _dst.write(_arr, 1)

    _loading_wbe    = wbe.read_raster(_tmp_precip)
    _absorption_wbe = wbe.read_raster(_tmp_recharge)
    _efficiency_wbe = wbe.read_raster(_tmp_eff)

    _net_accum_wbe = wbe.hydrology.flow_routing.d8_mass_flux(
        dem=dem_cond,
        loading=_loading_wbe,
        efficiency=_efficiency_wbe,
        absorption=_absorption_wbe,
    )
    wbe.write_raster(_net_accum_wbe, 'lanai_precip_net_accum.tif')
    print(f"Saved {_net_accum_path}")

    for _p in [_tmp_precip, _tmp_recharge, _tmp_eff]:
        os.remove(_p)

# --- 7. Streams ---
if os.path.exists(project_dir + 'streams.tif') and os.path.exists(project_dir + 'streams.gpkg'):
    print("Using existing streams.tif / streams.gpkg")
    streams = wbe.read_raster(project_dir + 'streams.tif')
else:
    streams = wbe.streams.network_extraction.extract_streams(
        flow_accumulation=flow_accum, threshold=streams_threshold
    )
    wbe.write_raster(streams, 'streams.tif')
    stream_vec = wbe.streams.network_extraction.raster_streams_to_vector(
        d8_pntr=d8_pntr, streams_raster=streams
    )
    wbe.write_vector(stream_vec, 'streams.gpkg')

# --- 8. Stream order ---
if os.path.exists(project_dir + 'strahler_order.tif'):
    print("Using existing strahler_order.tif")
else:
    strahler = wbe.streams.ordering_metrics.strahler_stream_order(d8_pntr=d8_pntr, streams=streams)
    wbe.write_raster(strahler, 'strahler_order.tif')

# --- 9. Watershed at every stream endpoint ---
if os.path.exists(project_dir + 'pour_points_snapped.gpkg'):
    print("Using existing pour_points_snapped.gpkg")
    snapped_pour_pts = wbe.read_vector(project_dir + 'pour_points_snapped.gpkg')
else:
    streams_gdf = gpd.read_file(project_dir + 'streams.gpkg')

    endpoints = []
    for geom in streams_gdf.geometry:
        if geom.geom_type == 'LineString':
            endpoints.append(Point(geom.coords[-1]))
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                endpoints.append(Point(line.coords[-1]))

    pour_pts_gdf = gpd.GeoDataFrame({'id': range(len(endpoints))}, geometry=endpoints, crs=streams_gdf.crs)
    raw_pour_pts_path = project_dir + 'pour_points_raw.shp'
    pour_pts_gdf.to_file(raw_pour_pts_path)

    raw_pour_pts = wbe.read_vector(raw_pour_pts_path)
    snapped_pour_pts = wbe.hydrology.watersheds_basins.jenson_snap_pour_points(
        pour_pts=raw_pour_pts,
        streams=streams,
        snap_dist=20
    )
    wbe.write_vector(snapped_pour_pts, 'pour_points_snapped.gpkg')

# --- 10. Watershed raster ---
if os.path.exists(project_dir + 'watershed.tif'):
    print("Using existing watershed.tif")
else:
    watershed = wbe.hydrology.watersheds_basins.watershed(d8_pntr=d8_pntr, pour_pts=snapped_pour_pts)
    wbe.write_raster(watershed, 'watershed.tif')

# --- 11. Height above stream ---
if os.path.exists(project_dir + 'height_above_stream.tif'):
    print("Using existing height_above_stream.tif")
else:
    height_above = wbe.hydrology.hydrologic_indices.elevation_above_stream(dem=dem_cond, streams=streams)
    wbe.write_raster(height_above, 'height_above_stream.tif')

# --- 12. Convert watershed raster to vector polygons ---
if os.path.exists(project_dir + 'watersheds.gpkg'):
    print("Using existing watersheds.gpkg")
else:
    with rasterio.open(project_dir + 'watershed.tif') as src:
        image = src.read(1)
        mask  = image != src.nodata if src.nodata is not None else None
        transform = src.transform
        crs = src.crs

        polygons = list(
            {'properties': {'watershed_id': int(v)}, 'geometry': shape(geom)}
            for geom, v in shapes(image, mask=mask, transform=transform)
        )

    watershed_gdf = gpd.GeoDataFrame.from_features(polygons, crs=crs)
    watershed_gdf.to_file(project_dir + 'watersheds.gpkg', driver='GPKG')

print("Watershed analysis pipeline complete.")
