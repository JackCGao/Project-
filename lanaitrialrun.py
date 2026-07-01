import whitebox_workflows as wbw

wbe = wbw.WbEnvironment()
wbe.verbose = True

# --- Inputs ---
project_dir = '/Users/jackgao/Summer Work 2026/Project-/'
dem_path    = project_dir + 'Lanai_DEM_masked.tif'
outlet_path = project_dir + 'outlet_point.shp'
streams_threshold = 500 #ask about optimal 
    
# --- 1. Read DEM ---
dem = wbe.read_raster(dem_path)

# --- 2. Fill missing data ---
dem = wbe.terrain.general.fill_missing_data(raster=dem, filter_size=11, exclude_edge_nodata = True)

# --- 3. Smooth ---
dem = wbe.terrain.general.feature_preserving_smoothing(raster=dem, filter_size=11, num_iter=2)
wbe.write_raster(dem, 'dem_smooth.tif')

# --- 4. Condition DEM ---
dem_cond = wbe.hydrology.depressions_storage.breach_depressions_least_cost(
    dem=dem, dist=50, fill_deps=True
)
wbe.write_raster(dem_cond, 'dem_conditioned.tif')

# --- 5. Flow direction ---
d8_pntr   = wbe.hydrology.flow_routing.d8_pointer(dem=dem_cond)
dinf_pntr = wbe.hydrology.flow_routing.dinf_pointer(dem=dem_cond) # if you want to use D-infinity flow routing for TWI
wbe.write_raster(d8_pntr, 'd8_pointer.tif')

# --- 6. Flow accumulation ---
# 'input' takes DEM or pointer; since we're passing the D8 pointer, flag it
flow_accum = wbe.hydrology.flow_routing.d8_flow_accum(
    input=d8_pntr, out_type='cells', input_is_pointer=True
)
wbe.write_raster(flow_accum, 'flow_accum.tif')

# --- 7. Streams ---
streams = wbe.streams.network_extraction.extract_streams(
    flow_accumulation=flow_accum, threshold=streams_threshold
)
wbe.write_raster(streams, 'streams.tif') # doing the height thing for the streams making .tif file
stream_vec = wbe.streams.network_extraction.raster_streams_to_vector(
    d8_pntr=d8_pntr, streams_raster=streams
)
wbe.write_vector(stream_vec, 'streams.gpkg')

# --- 8. Stream order ---
strahler = wbe.streams.ordering_metrics.strahler_stream_order(d8_pntr=d8_pntr, streams=streams)
wbe.write_raster(strahler, 'strahler_order.tif')

# --- 9. Watershed at every stream endpoint ---
import geopandas as gpd
from shapely.geometry import Point

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

watershed = wbe.hydrology.watersheds_basins.watershed(d8_pntr=d8_pntr, pour_pts=snapped_pour_pts)
wbe.write_raster(watershed, 'watershed.tif')

# --- 11. Height above stream ---
height_above = wbe.hydrology.hydrologic_indices.elevation_above_stream(dem=dem_cond, streams=streams)
wbe.write_raster(height_above, 'height_above_stream.tif')

# --- 12. Convert watershed raster to vector polygons ---

import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

with rasterio.open(project_dir + 'watershed.tif') as src:
    image = src.read(1)
    mask = image != src.nodata if src.nodata is not None else None
    transform = src.transform
    crs = src.crs

    results = (
        {'properties': {'watershed_id': int(v)}, 'geometry': shape(geom)}
        for geom, v in shapes(image, mask=mask, transform=transform)
    )

    polygons = list(results)

watershed_gdf = gpd.GeoDataFrame.from_features(polygons, crs=crs)
watershed_gdf.to_file(project_dir + 'watersheds.gpkg', driver='GPKG')

print("Watershed analysis pipeline complete.")