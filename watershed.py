#!/usr/bin/env python3
"""
Delineate watersheds for every Hawaiian island using WhiteboxTools, with
flow accumulation weighted by each island's precipitation raster.

For each island in clay_vs_precip.island_files (dem + precip paths already
established there):
  1. Fill missing data and smooth the DEM, then condition it (breach
     depressions) so flow routing has no unresolved sinks.
  2. Compute the D8 flow-direction pointer and D8 flow accumulation
     (cell counts, used later purely for stream extraction).
  3. Compute precipitation-weighted flow accumulation via d8_mass_flux,
     using the island's precipitation raster (resampled onto the DEM grid)
     as the loading term with efficiency=1 and absorption=0, i.e. no losses
     -- every upstream cell's precipitation is routed downstream untouched.
  4. Extract streams from the flow accumulation raster, vectorize them, and
     merge every connected network of touching segments -- main stem plus
     every joining tributary, straight through confluences -- into a single
     feature before exporting them as a lines shapefile.
  5. Delineate watersheds directly from the D8 pointer via WhiteboxTools'
     `basins` tool -- every complete drainage basin that drains to a
     valid-data edge (i.e. the coastline), not sub-catchments above internal
     stream junctions. (An earlier version placed a pour point at every
     stream-segment endpoint and ran `watershed()`, which produced one tiny
     basin per tributary junction instead of a handful of true watersheds.)
  6. Vectorize the watershed raster to polygons, then export both the
     polygons and their boundary lines as shapefiles for easy display.

whitebox_workflows' read/write round-trip does not preserve each island's
real CRS (every file it writes gets mislabeled EPSG:4269, even though the
pixel grid/coordinates are untouched and still in the source DEM's true
projection). _ensure_raster_crs / _ensure_vector_crs patch that label back to
the DEM's real CRS wherever a wbw output is written or read from a prior run,
and step 5 resamples precipitation using that real CRS rather than trusting
the (mislabeled) conditioned-DEM file.

After every island finishes, the per-island watershed polygons and streams
are each reprojected to the common ESRI:102007 CRS and concatenated into two
project-wide shapefiles: all_islands_watersheds.shp and all_islands_streams.shp.

Outputs are cached per-island under watershed_outputs/{island}/ so re-running
the script skips steps that already completed.
"""

import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import whitebox_workflows as wbw
from rasterio.features import shapes
from shapely.geometry import shape, MultiLineString
from shapely.ops import linemerge

from clay_vs_precip import island_files, resample_to_ref

PROJECT_DIR = '/Users/jackgao/Summer Work 2026/Project-/'
OUTPUT_DIR = PROJECT_DIR + 'watershed_outputs/'
STREAMS_THRESHOLD = 5000  # contributing cells; raised from 500 so only major channels count as streams
COMBINED_CRS = 'ESRI:102007'

wbe = wbw.WbEnvironment()
wbe.verbose = True
wbe.working_directory = PROJECT_DIR


def _ensure_raster_crs(path, crs):
    """wbw's write_raster mislabels the output CRS (pixel grid is untouched,
    only the tag is wrong) -- overwrite it with the DEM's real CRS."""
    with rasterio.open(path, 'r+') as dst:
        if dst.crs != crs:
            dst.crs = crs


def _merge_stream_segments(gdf):
    """raster_streams_to_vector breaks each drainage network into one
    segment per reach between confluences. Group segments into connected
    networks (sharing an endpoint, transitively -- including through
    confluences where a tributary joins) via union-find on exact endpoint
    coordinates, then merge each whole network into a single feature. A
    network with no internal branching collapses to one LineString; one
    with tributaries becomes a single MultiLineString feature covering the
    entire connected system, so selecting any part of a river highlights
    the whole thing -- main stem and every joining tributary -- at once."""
    lines = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'LineString':
            lines.append(geom)
        elif geom.geom_type == 'MultiLineString':
            lines.extend(geom.geoms)

    n = len(lines)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    endpoint_to_lines = {}
    for i, line in enumerate(lines):
        coords = line.coords
        for pt in (coords[0], coords[-1]):
            endpoint_to_lines.setdefault(pt, []).append(i)
    for idxs in endpoint_to_lines.values():
        for i in idxs[1:]:
            union(idxs[0], i)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])

    merged_geoms = []
    for group_lines in groups.values():
        merged = linemerge(group_lines)
        merged_geoms.append(MultiLineString([merged]) if merged.geom_type == 'LineString' else merged)

    return gpd.GeoDataFrame({'stream_id': range(len(merged_geoms))}, geometry=merged_geoms, crs=gdf.crs)


def _ensure_vector_crs(path, crs):
    """Same CRS-mislabeling issue as _ensure_raster_crs, but for
    wbe.write_vector() outputs (streams.gpkg, pour_points_snapped.gpkg)."""
    gdf = gpd.read_file(path)
    if gdf.crs != crs:
        gdf = gdf.set_crs(crs, allow_override=True)
        gdf.to_file(path)


def process_island(island, paths):
    out_dir = OUTPUT_DIR + island + '/'
    os.makedirs(out_dir, exist_ok=True)

    dem_path = paths['dem']
    precip_path = paths['precip']

    with rasterio.open(dem_path) as src:
        true_crs = src.crs

    # --- 1. Fill missing data + smooth ---
    smooth_path = out_dir + 'dem_smooth.tif'
    if os.path.exists(smooth_path):
        print(f"[{island}] using existing dem_smooth.tif")
        _ensure_raster_crs(smooth_path, true_crs)
        dem = wbe.read_raster(smooth_path)
    else:
        dem = wbe.read_raster(dem_path)
        dem = wbe.terrain.general.fill_missing_data(raster=dem, filter_size=11, exclude_edge_nodata=True)
        dem = wbe.terrain.general.feature_preserving_smoothing(raster=dem, filter_size=11, num_iter=2)
        wbe.write_raster(dem, smooth_path)
        _ensure_raster_crs(smooth_path, true_crs)

    # --- 2. Condition DEM ---
    cond_path = out_dir + 'dem_conditioned.tif'
    if os.path.exists(cond_path):
        print(f"[{island}] using existing dem_conditioned.tif")
        _ensure_raster_crs(cond_path, true_crs)
        dem_cond = wbe.read_raster(cond_path)
    else:
        dem_cond = wbe.hydrology.depressions_storage.breach_depressions_least_cost(
            dem=dem, dist=50, fill_deps=True
        )
        wbe.write_raster(dem_cond, cond_path)
        _ensure_raster_crs(cond_path, true_crs)

    # --- 3. Flow direction ---
    pntr_path = out_dir + 'd8_pointer.tif'
    if os.path.exists(pntr_path):
        print(f"[{island}] using existing d8_pointer.tif")
        _ensure_raster_crs(pntr_path, true_crs)
        d8_pntr = wbe.read_raster(pntr_path)
    else:
        d8_pntr = wbe.hydrology.flow_routing.d8_pointer(dem=dem_cond)
        wbe.write_raster(d8_pntr, pntr_path)
        _ensure_raster_crs(pntr_path, true_crs)

    # --- 4. Flow accumulation (unweighted, cell counts -- used for stream extraction) ---
    accum_path = out_dir + 'flow_accum.tif'
    if os.path.exists(accum_path):
        print(f"[{island}] using existing flow_accum.tif")
        _ensure_raster_crs(accum_path, true_crs)
        flow_accum = wbe.read_raster(accum_path)
    else:
        flow_accum = wbe.hydrology.flow_routing.d8_flow_accum(
            input=d8_pntr, out_type='cells', input_is_pointer=True
        )
        wbe.write_raster(flow_accum, accum_path)
        _ensure_raster_crs(accum_path, true_crs)

    # --- 5. Precipitation-weighted flow accumulation via d8_mass_flux ---
    weighted_path = out_dir + 'precip_weighted_accum.tif'
    if os.path.exists(weighted_path):
        print(f"[{island}] using existing precip_weighted_accum.tif")
        _ensure_raster_crs(weighted_path, true_crs)
    else:
        with rasterio.open(cond_path) as src:
            ref_transform = src.transform
            ref_h, ref_w = src.height, src.width
            ref_profile = src.profile.copy()

        precip_arr = resample_to_ref(precip_path, ref_transform, true_crs, (ref_h, ref_w))
        precip_arr = np.nan_to_num(precip_arr, nan=0.0)

        prof = ref_profile.copy()
        # nodata must not collide with a legitimate data value -- absorption
        # is intentionally all zeros (no losses), so 0.0 cannot be the sentinel
        prof.update(dtype='float64', count=1, nodata=-9999.0, crs=true_crs)
        tmp_precip = out_dir + '_tmp_precip.tif'
        tmp_eff = out_dir + '_tmp_efficiency.tif'
        tmp_abs = out_dir + '_tmp_absorption.tif'
        for path, arr in [(tmp_precip, precip_arr),
                          (tmp_eff, np.ones((ref_h, ref_w), dtype=np.float64)),
                          (tmp_abs, np.zeros((ref_h, ref_w), dtype=np.float64))]:
            with rasterio.open(path, 'w', **prof) as dst:
                dst.write(arr, 1)

        loading_wbe = wbe.read_raster(tmp_precip)
        efficiency_wbe = wbe.read_raster(tmp_eff)
        absorption_wbe = wbe.read_raster(tmp_abs)

        weighted_accum = wbe.hydrology.flow_routing.d8_mass_flux(
            dem=dem_cond, loading=loading_wbe, efficiency=efficiency_wbe, absorption=absorption_wbe,
        )
        wbe.write_raster(weighted_accum, weighted_path)
        _ensure_raster_crs(weighted_path, true_crs)

        for p in (tmp_precip, tmp_eff, tmp_abs):
            os.remove(p)
        print(f"[{island}] saved precip_weighted_accum.tif")

    # --- 6. Streams ---
    streams_tif_path = out_dir + 'streams.tif'
    streams_gpkg_path = out_dir + 'streams.gpkg'
    if os.path.exists(streams_tif_path) and os.path.exists(streams_gpkg_path):
        print(f"[{island}] using existing streams.tif / streams.gpkg")
        _ensure_raster_crs(streams_tif_path, true_crs)
        _ensure_vector_crs(streams_gpkg_path, true_crs)
        streams = wbe.read_raster(streams_tif_path)
    else:
        streams = wbe.streams.network_extraction.extract_streams(
            flow_accumulation=flow_accum, threshold=STREAMS_THRESHOLD
        )
        wbe.write_raster(streams, streams_tif_path)
        _ensure_raster_crs(streams_tif_path, true_crs)
        stream_vec = wbe.streams.network_extraction.raster_streams_to_vector(
            d8_pntr=d8_pntr, streams_raster=streams
        )
        wbe.write_vector(stream_vec, streams_gpkg_path)
        _ensure_vector_crs(streams_gpkg_path, true_crs)

    # --- 6b. Streams as a shapefile (lines), merged into continuous reaches ---
    streams_shp_path = out_dir + 'streams.shp'
    if os.path.exists(streams_shp_path):
        print(f"[{island}] using existing streams.shp")
    else:
        merged_gdf = _merge_stream_segments(gpd.read_file(streams_gpkg_path))
        merged_gdf.to_file(streams_shp_path)
        print(f"[{island}] saved streams.shp ({len(merged_gdf)} merged lines)")

    # --- 7. Stream order ---
    strahler_path = out_dir + 'strahler_order.tif'
    if os.path.exists(strahler_path):
        print(f"[{island}] using existing strahler_order.tif")
        _ensure_raster_crs(strahler_path, true_crs)
    else:
        strahler = wbe.streams.ordering_metrics.strahler_stream_order(d8_pntr=d8_pntr, streams=streams)
        wbe.write_raster(strahler, strahler_path)
        _ensure_raster_crs(strahler_path, true_crs)

    # --- 8. Watershed raster -- full drainage basins from the D8 pointer ---
    # (basins() finds every basin that drains to a valid-data edge, i.e. the
    # coastline, giving true ridge-to-coast watersheds rather than one tiny
    # sub-catchment per stream-segment junction)
    watershed_path = out_dir + 'watershed.tif'
    if os.path.exists(watershed_path):
        print(f"[{island}] using existing watershed.tif")
        _ensure_raster_crs(watershed_path, true_crs)
    else:
        watershed = wbe.hydrology.watersheds_basins.basins(d8_pntr=d8_pntr)
        wbe.write_raster(watershed, watershed_path)
        _ensure_raster_crs(watershed_path, true_crs)

    # --- 9. Watershed raster -> vector polygons ---
    watershed_gpkg_path = out_dir + 'watersheds.gpkg'
    if os.path.exists(watershed_gpkg_path):
        print(f"[{island}] using existing watersheds.gpkg")
        _ensure_vector_crs(watershed_gpkg_path, true_crs)
    else:
        with rasterio.open(watershed_path) as src:
            image = src.read(1)
            mask = image != src.nodata if src.nodata is not None else None
            transform = src.transform

            polygons = [
                {'properties': {'watershed_id': int(v)}, 'geometry': shape(geom)}
                for geom, v in shapes(image, mask=mask, transform=transform)
            ]

        watershed_gdf = gpd.GeoDataFrame.from_features(polygons, crs=true_crs)
        watershed_gdf.to_file(watershed_gpkg_path, driver='GPKG')

    # --- 10. Watershed boundaries as shapefiles ---
    watershed_shp_path = out_dir + 'watersheds.shp'
    boundary_shp_path = out_dir + 'watershed_boundaries.shp'
    if os.path.exists(watershed_shp_path) and os.path.exists(boundary_shp_path):
        print(f"[{island}] using existing watersheds.shp / watershed_boundaries.shp")
    else:
        watershed_gdf = gpd.read_file(watershed_gpkg_path)
        watershed_gdf.to_file(watershed_shp_path)

        boundary_gdf = watershed_gdf.copy()
        boundary_gdf['geometry'] = boundary_gdf.geometry.boundary
        boundary_gdf.to_file(boundary_shp_path)
        print(f"[{island}] saved watersheds.shp / watershed_boundaries.shp")

    print(f"[{island}] watershed pipeline complete.\n")


def combine_across_islands(file_name, out_name):
    """Read `file_name` (e.g. 'watersheds.gpkg') out of every island's output
    directory, tag each feature with its island, reproject to the shared
    COMBINED_CRS, and write the concatenation out as one shapefile."""
    parts = []
    for island in island_files:
        path = OUTPUT_DIR + island + '/' + file_name
        if not os.path.exists(path):
            print(f"[{island}] {file_name} not found, skipping in {out_name}")
            continue
        gdf = gpd.read_file(path).to_crs(COMBINED_CRS)
        gdf['island'] = island
        parts.append(gdf)

    if not parts:
        print(f"No {file_name} found for any island -- skipping {out_name}")
        return

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=COMBINED_CRS)
    combined.to_file(OUTPUT_DIR + out_name)
    print(f"Saved {out_name}  (n = {len(combined)})")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for island, paths in island_files.items():
        print(f"=== Processing {island} ===")
        try:
            process_island(island, paths)
        except Exception as exc:
            print(f"[{island}] FAILED: {exc}")

    print("=== Combining outputs across all islands ===")
    combine_across_islands('watersheds.gpkg', 'all_islands_watersheds.shp')
    combine_across_islands('streams.shp', 'all_islands_streams.shp')


if __name__ == '__main__':
    main()
