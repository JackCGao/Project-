#!/usr/bin/env python3
"""
Sample each island's D8 max-flux raster at the stream-pixel point locations
in its *_streams_unweighted_albers.shp (a points layer -- one point per
stream pixel, already the point-per-pixel conversion of the corresponding
streams raster, VALUE == 1 everywhere -- see oahu_streams_unweighted_albers.shp).

The streams shapefile acts purely as a mask of *where* to sample: for every
point, the underlying maxflux GeoTIFF is read at that point's pixel and the
raw max-flux value is kept as a new 'maxflux' field. Points that fall outside
the maxflux raster's extent, or land on a nodata pixel, are dropped.

Points are further filtered to only those within PROXIMITY_PX pixels (island's
own raster resolution) of the statewide hydrography layer at
Hawaii Streams/Streams_reprojected.shp, via exact nearest-line distance
(geopandas sjoin_nearest, not vertex-approximated).

For each island, writes watershed_outputs/{island}/maxflux_stream_points.shp.
Also builds a combined all_islands_maxflux_stream_points.shp across every
island with data, reprojected to the shared ESRI:102007 CRS and tagged with
an 'island' column, following the same combine convention as watershed.py /
resegment_streams.py.

Points dropped by either masking step (no maxflux data at that pixel, or
too far from the statewide hydrography layer) are kept rather than
discarded: written per-island to maxflux_stream_points_removed.shp and
combined into all_islands_maxflux_stream_points_removed.shp, each point
tagged with a 'reason' field ('no_maxflux_data' or 'not_near_hydro') so
it's possible to see what got masked out and why.

hawaii has no plain hawaii_streams_unweighted_albers.shp (only threshold
variants, _5000.shp / _10000.shp) -- it's skipped unless ISLAND_FILES is
pointed at one of those explicitly.
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
# built from it.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
OUTPUT_DIR = PROJECT_DIR + 'watershed_outputs/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_CRS = 'ESRI:102007'

HYDRO_STREAMS_SHP = f'{BASE_DIR}/Hawaii Streams/Streams_reprojected.shp'
PROXIMITY_PX = 50  # keep only points within this many pixels of a Streams_reprojected.shp line
MAXFLUX_SCALE = 1e-3  # applied to raw raster values before writing the 'maxflux' field

# All per-island source data now lives under Island Data/ (the project's
# earlier flat-per-island-folder layout, e.g. "oahu/new/...", was
# reorganized under this common parent -- a stale path here silently
# no-ops every path built from it, same caveat as PROJECT_DIR above).
ISLAND_DATA_DIR = f'{BASE_DIR}/Island Data'

# Weighted flow accumulation (D8 max-flux, masked to streams) rasters live
# here, one per island, rather than under each island's own "new/" folder.
WEIGHTED_FLOW_DIR = f'{ISLAND_DATA_DIR}/weighted_flow'

ISLAND_FILES = {
    'oahu': {
        'maxflux': f'{WEIGHTED_FLOW_DIR}/oahu_d8maxflux_masked_streams_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/oahu/new/oahu_streams_unweighted_albers.shp',
    },
    'kauai': {
        'maxflux': f'{WEIGHTED_FLOW_DIR}/kauai_d8maxflux_masked_streams_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/kauai/new/kauai_streams_unweighted_albers.shp',
    },
    'lanai': {
        'maxflux': f'{WEIGHTED_FLOW_DIR}/lanai_d8maxflux_masked_streams_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/lanai/new/lanai_streams_unweighted_albers.shp',
    },
    'maui': {
        'maxflux': f'{WEIGHTED_FLOW_DIR}/maui_d8maxflux_masked_streams_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/maui/new/maui_streams_unweighted_albers.shp',
    },
    'molokai': {
        'maxflux': f'{WEIGHTED_FLOW_DIR}/molokai_d8maxflux_masked_streams_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/molokai/new/molokai_streams_unweighted_albers.shp',
    },
    # No "new/" subfolder for kahoolawe -- its shapefile sits directly
    # under Island Data/kahoolawe/.
    'kahoolawe': {
        'maxflux': f'{WEIGHTED_FLOW_DIR}/kahoolawe_d8maxflux_masked_streams_albers.tif',
        'streams_shp': f'{ISLAND_DATA_DIR}/kahoolawe/kahoolawe_streams_unweighted_albers.shp',
    },
    # No plain hawaii_streams_unweighted_albers.shp exists -- use the
    # _10000 (contributing-cells threshold) variant instead. Also no
    # "new (1)/" subfolder -- sits directly under Island Data/hawaii/.
    'hawaii': {
        'maxflux': f'{WEIGHTED_FLOW_DIR}/hawaii_d8maxflux_masked_streams_albers.tif',
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


def sample_island(island, maxflux_path, streams_shp_path):
    """Returns (kept, removed) -- 'removed' is every point dropped by either
    masking step, tagged with a 'reason' column so it's clear which."""
    gdf = gpd.read_file(streams_shp_path)

    with rasterio.open(maxflux_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        pixel_size = src.res[0]
        crs = src.crs

    values = sample_raster_at_points(maxflux_path, gdf.geometry.x.values, gdf.geometry.y.values, crs)

    out_gdf = gdf.drop(columns=['VALUE'], errors='ignore').copy()
    out_gdf['maxflux'] = values * MAXFLUX_SCALE

    finite_mask = np.isfinite(out_gdf['maxflux'])
    no_data_removed = out_gdf[~finite_mask].copy()
    no_data_removed['reason'] = 'no_maxflux_data'

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
    maxflux_path = paths['maxflux']
    streams_shp_path = paths['streams_shp']

    if streams_shp_path is None or not os.path.exists(streams_shp_path):
        print(f"[{island}] streams shapefile not found, skipping")
        return None, None
    if not os.path.exists(maxflux_path):
        print(f"[{island}] maxflux raster not found, skipping")
        return None, None

    out_dir = OUTPUT_DIR + island + '/'
    out_path = out_dir + 'maxflux_stream_points.shp'
    removed_path = out_dir + 'maxflux_stream_points_removed.shp'
    if os.path.exists(out_path):
        print(f"[{island}] using existing maxflux_stream_points.shp")
        removed = gpd.read_file(removed_path) if os.path.exists(removed_path) else None
        return gpd.read_file(out_path), removed

    os.makedirs(out_dir, exist_ok=True)
    result, removed = sample_island(island, maxflux_path, streams_shp_path)
    result.to_file(out_path)
    print(f"[{island}] saved maxflux_stream_points.shp "
          f"(n={len(result)}, maxflux mean={result['maxflux'].mean():.3f})")

    if len(removed) > 0:
        removed.to_file(removed_path)
        by_reason = removed['reason'].value_counts().to_dict()
        print(f"[{island}] saved maxflux_stream_points_removed.shp "
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
    combine_across_islands(per_island, 'all_islands_maxflux_stream_points.shp')
    combine_across_islands(per_island_removed, 'all_islands_maxflux_stream_points_removed.shp')


if __name__ == '__main__':
    main()
