#!/usr/bin/env python3
"""
Converts each island's maxflux_stream_points_removed.shp (points dropped by
maxflux_stream_points.py's masking, tagged with a 'reason' field -- see
that script's docstring) into line geometry:
maxflux_stream_lines_removed.shp -- one short LineString per pair of
raster-adjacent removed points (8-connected, i.e. within one pixel-
diagonal of each other), each segment's 'maxflux' field the average of its
two endpoints' values, and 'reason' carried over when both endpoints
share the same reason or set to 'mixed' when they don't.

Same purely-geometric adjacency reconstruction as maxflux_stream_lines.py
(which does this for the KEPT points, maxflux_stream_points.shp) -- this
does NOT snap onto watershed.py's streams_segmented.shp vector network,
so connectivity here reflects exactly whichever points got dropped by the
same-island masking, nothing else.

Both versions are kept: this script only reads the already-built
maxflux_stream_points_removed.shp / all_islands_maxflux_stream_points_removed.shp
(run maxflux_stream_points.py first if they don't exist yet) and writes
new maxflux_stream_lines_removed.shp files alongside them -- it never
touches or replaces the point outputs.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from maxflux_stream_points import OUTPUT_DIR, ISLAND_FILES, COMBINED_CRS

# 8-connected raster-adjacency threshold: a bit above sqrt(2) * pixel_size
# so diagonal neighbors are linked but points two-or-more pixels apart
# (e.g. across a gap left by masking) are not -- same value as
# maxflux_stream_lines.py.
ADJACENCY_FACTOR = 1.5


def points_to_lines(gdf, pixel_size):
    """One 2-point LineString per pair of points within
    ADJACENCY_FACTOR * pixel_size of each other; 'maxflux' = mean of the
    pair's values (NaN-safe -- no_maxflux_data points have no valid
    maxflux, so a segment touching one keeps NaN, same as leaving it out
    of a numeric average); 'reason' = the shared reason, or 'mixed' if the
    pair's reasons differ (e.g. one endpoint dropped for no_maxflux_data,
    the other for not_near_hydro)."""
    coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    tree = cKDTree(coords)
    pairs = sorted(tree.query_pairs(r=ADJACENCY_FACTOR * pixel_size))

    if not pairs:
        return gpd.GeoDataFrame({'maxflux': [], 'reason': []}, geometry=[], crs=gdf.crs)

    maxflux = gdf['maxflux'].values
    reason = gdf['reason'].values

    lines = [LineString([coords[i], coords[j]]) for i, j in pairs]
    values = [np.nanmean([maxflux[i], maxflux[j]]) for i, j in pairs]
    reasons = [reason[i] if reason[i] == reason[j] else 'mixed' for i, j in pairs]

    return gpd.GeoDataFrame({'maxflux': values, 'reason': reasons}, geometry=lines, crs=gdf.crs)


def process_island(island):
    points_path = f'{OUTPUT_DIR}{island}/maxflux_stream_points_removed.shp'
    if not os.path.exists(points_path):
        print(f"[{island}] maxflux_stream_points_removed.shp not found, skipping "
              f"(run maxflux_stream_points.py first)")
        return None

    out_path = f'{OUTPUT_DIR}{island}/maxflux_stream_lines_removed.shp'
    if os.path.exists(out_path):
        print(f"[{island}] using existing maxflux_stream_lines_removed.shp")
        return gpd.read_file(out_path)

    gdf = gpd.read_file(points_path)
    if len(gdf) == 0:
        print(f"[{island}] no removed points, skipping")
        return None

    with rasterio.open(ISLAND_FILES[island]['maxflux']) as src:
        pixel_size = src.res[0]

    lines_gdf = points_to_lines(gdf, pixel_size)
    lines_gdf.to_file(out_path)
    print(f"[{island}] saved maxflux_stream_lines_removed.shp "
          f"({len(lines_gdf)} segments from {len(gdf)} points)")
    return lines_gdf


def combine_across_islands(per_island):
    parts = []
    for island, gdf in per_island.items():
        if gdf is None or len(gdf) == 0:
            continue
        tagged = gdf.to_crs(COMBINED_CRS).copy()
        tagged['island'] = island
        parts.append(tagged)

    if not parts:
        print("No maxflux_stream_lines_removed found for any island -- skipping combined output")
        return

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=COMBINED_CRS)
    out_path = OUTPUT_DIR + 'all_islands_maxflux_stream_lines_removed.shp'
    combined.to_file(out_path)
    print(f"Saved all_islands_maxflux_stream_lines_removed.shp (n = {len(combined)})")


def main():
    per_island = {}
    for island in ISLAND_FILES:
        per_island[island] = process_island(island)
    combine_across_islands(per_island)


if __name__ == '__main__':
    main()
