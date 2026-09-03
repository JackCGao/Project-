#!/usr/bin/env python3
"""
Unweighted counterpart of maxflux_stream_points.py: same masking logic
(sample at each island's *_streams_unweighted_albers.shp point locations,
then filter to points within PROXIMITY_PX pixels of the statewide
hydrography layer at Hawaii Streams/Streams_reprojected.shp), but the
raster sampled at those points is each island's raw UNWEIGHTED D8 flow
accumulation raster (Island Data/{island}/[new/]{island}_unweighted_albers.tif
-- plain cell counts) instead of the maxflux (precipitation-weighted)
raster in Island Data/weighted_flow/.

The streams shapefile acts purely as a mask of *where* to sample: for every
point, the underlying unweighted GeoTIFF is read at that point's pixel and
the raw cell-count value is kept as a new 'flow_accum' field. Points that
fall outside the raster's extent, or land on a nodata pixel, are dropped.

For each island, writes watershed_outputs/{island}/unweighted_stream_points.shp.
Also builds a combined all_islands_unweighted_stream_points.shp across every
island with data, reprojected to the shared ESRI:102007 CRS and tagged with
an 'island' column -- same combine convention as maxflux_stream_points.py /
watershed.py / resegment_streams.py.

Points dropped by either masking step (no flow_accum data at that pixel, or
too far from the statewide hydrography layer) are kept rather than
discarded: written per-island to unweighted_stream_points_removed.shp and
combined into all_islands_unweighted_stream_points_removed.shp, each point
tagged with a 'reason' field ('no_flow_data' or 'not_near_hydro') so it's
possible to see what got masked out and why.

hawaii has no plain hawaii_streams_unweighted_albers.shp (only threshold
variants, _5000.shp / _10000.shp) -- it's skipped unless ISLAND_FILES is
pointed at one of those explicitly. Same caveat as maxflux_stream_points.py.
"""

import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd

from usefulfunctions import sample_raster_at_points

# Derived from this file's own location rather than hardcoded, since the
# project folder has been renamed before (it used to be "Summer Work 2026",
# with spaces) and a stale hardcoded path here silently no-ops every path
# built from it -- same caveat as maxflux_stream_points.py's PROJECT_DIR.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
OUTPUT_DIR = PROJECT_DIR + 'watershed_outputs/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_CRS = 'ESRI:102007'

HYDRO_STREAMS_SHP = f'{BASE_DIR}/Hawaii Streams/Streams_reprojected.shp'
PROXIMITY_PX = 50  # keep only points within this many pixels of a Streams_reprojected.shp line

# All per-island source data now lives under Island Data/ (the project's
# earlier flat-per-island-folder layout, e.g. "oahu/new/...", was
# reorganized under this common parent -- a stale path here silently
# no-ops every path built from it, same caveat as PROJECT_DIR above).
ISLAND_DATA_DIR = f'{BASE_DIR}/Island Data'

ISLAND_FILES = {
    'oahu': {
        'unweighted': f'{ISLAND_DATA_DIR}/oahu/new/oahu_unweighted_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/oahu/new/oahu_streams_unweighted_albers.shp',
    },
    'kauai': {
        'unweighted': f'{ISLAND_DATA_DIR}/kauai/new/kauai_unweighted_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/kauai/new/kauai_streams_unweighted_albers.shp',
    },
    'lanai': {
        'unweighted': f'{ISLAND_DATA_DIR}/lanai/new/lanai_unweighted_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/lanai/new/lanai_streams_unweighted_albers.shp',
    },
    'maui': {
        'unweighted': f'{ISLAND_DATA_DIR}/maui/new/maui_unweighted_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/maui/new/maui_streams_unweighted_albers.shp',
    },
    'molokai': {
        'unweighted': f'{ISLAND_DATA_DIR}/molokai/new/molokai_unweighted_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/molokai/new/molokai_streams_unweighted_albers.shp',
    },
    # No "new/" subfolder for kahoolawe -- its files sit directly under
    # Island Data/kahoolawe/.
    'kahoolawe': {
        'unweighted': f'{ISLAND_DATA_DIR}/kahoolawe/kahoolawe_unweighted_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/kahoolawe/kahoolawe_streams_unweighted_albers.shp',
    },
    # No plain hawaii_streams_unweighted_albers.shp exists -- use the
    # _10000 (contributing-cells threshold) variant instead. Also no
    # "new/" subfolder -- sits directly under Island Data/hawaii/.
    'hawaii': {
        'unweighted': f'{ISLAND_DATA_DIR}/hawaii/hawaii_unweighted_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/hawaii/hawaii_streams_unweighted_albers_10000.shp',
    },
}

_hydro_gdf = None


def _load_hydro_gdf():
    global _hydro_gdf
    if _hydro_gdf is None:
        _hydro_gdf = gpd.read_file(HYDRO_STREAMS_SHP)[['geometry']]
    return _hydro_gdf


def filter_by_hydro_proximity(gdf, crs, pixel_size):
    """Keep only points within PROXIMITY_PX * pixel_size of the nearest line
    in Streams_reprojected.shp, using exact (non-vertex-approximated)
    nearest-geometry distance via sjoin_nearest. Returns (kept, removed) --
    'removed' is whatever sjoin_nearest couldn't match within max_distance,
    i.e. points with no line of Streams_reprojected.shp close enough."""
    hydro = _load_hydro_gdf()
    if hydro.crs != crs:
        hydro = hydro.to_crs(crs)

    max_dist = PROXIMITY_PX * pixel_size
    joined = gpd.sjoin_nearest(gdf, hydro, max_distance=max_dist, distance_col='hydro_dist')
    joined = joined[~joined.index.duplicated(keep='first')]
    joined = joined.drop(columns=['index_right'], errors='ignore')
    joined = joined.reindex(gdf.index)

    kept = joined.dropna(subset=['hydro_dist'])
    removed = gdf.loc[joined['hydro_dist'].isna()]
    return kept, removed


def sample_island(island, unweighted_path, streams_shp_path):
    """Returns (kept, removed) -- 'removed' is every point dropped by either
    masking step, tagged with a 'reason' column so it's clear which."""
    gdf = gpd.read_file(streams_shp_path)

    with rasterio.open(unweighted_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        pixel_size = src.res[0]
        crs = src.crs

    values = sample_raster_at_points(unweighted_path, gdf.geometry.x.values, gdf.geometry.y.values, crs)

    out_gdf = gdf.drop(columns=['VALUE'], errors='ignore').copy()
    out_gdf['flow_accum'] = values

    finite_mask = np.isfinite(out_gdf['flow_accum'])
    no_data_removed = out_gdf[~finite_mask].copy()
    no_data_removed['reason'] = 'no_flow_data'

    out_gdf = out_gdf[finite_mask].reset_index(drop=True)

    kept_gdf, hydro_removed = filter_by_hydro_proximity(out_gdf, crs, pixel_size)
    kept_gdf = kept_gdf.reset_index(drop=True)
    hydro_removed = hydro_removed.copy()
    hydro_removed['reason'] = 'not_near_hydro'

    removed_gdf = gpd.GeoDataFrame(
        pd.concat([no_data_removed, hydro_removed], ignore_index=True),
        crs=crs,
    )
    return kept_gdf, removed_gdf


def process_island(island, paths):
    unweighted_path = paths['unweighted']
    streams_shp_path = paths['streams_shp']

    if streams_shp_path is None or not os.path.exists(streams_shp_path):
        print(f"[{island}] streams shapefile not found, skipping")
        return None, None
    if not os.path.exists(unweighted_path):
        print(f"[{island}] unweighted flow-accumulation raster not found, skipping")
        return None, None

    out_dir = OUTPUT_DIR + island + '/'
    out_path = out_dir + 'unweighted_stream_points.shp'
    removed_path = out_dir + 'unweighted_stream_points_removed.shp'
    if os.path.exists(out_path):
        print(f"[{island}] using existing unweighted_stream_points.shp")
        removed = gpd.read_file(removed_path) if os.path.exists(removed_path) else None
        return gpd.read_file(out_path), removed

    os.makedirs(out_dir, exist_ok=True)
    result, removed = sample_island(island, unweighted_path, streams_shp_path)
    result.to_file(out_path)
    print(f"[{island}] saved unweighted_stream_points.shp "
          f"(n={len(result)}, flow_accum mean={result['flow_accum'].mean():.3f})")

    if len(removed) > 0:
        removed.to_file(removed_path)
        by_reason = removed['reason'].value_counts().to_dict()
        print(f"[{island}] saved unweighted_stream_points_removed.shp "
              f"(n={len(removed)}, {by_reason})")
    else:
        print(f"[{island}] no points removed by masking")

    return result, removed


def combine_across_islands(per_island, out_name):
    parts = []
    for island, gdf in per_island.items():
        if gdf is None or len(gdf) == 0:
            continue
        tagged = gdf.to_crs(COMBINED_CRS).copy()
        tagged['island'] = island
        parts.append(tagged)

    if not parts:
        print(f"No data found for any island -- skipping {out_name}")
        return

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=COMBINED_CRS)
    out_path = OUTPUT_DIR + out_name
    combined.to_file(out_path)
    print(f"Saved {out_name} (n = {len(combined)})")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    per_island = {}
    per_island_removed = {}
    for island, paths in ISLAND_FILES.items():
        kept, removed = process_island(island, paths)
        per_island[island] = kept
        per_island_removed[island] = removed
    combine_across_islands(per_island, 'all_islands_unweighted_stream_points.shp')
    combine_across_islands(per_island_removed, 'all_islands_unweighted_stream_points_removed.shp')


if __name__ == '__main__':
    main()
