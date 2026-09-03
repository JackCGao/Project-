#!/usr/bin/env python3
"""
Converts each island's maxflux_stream_points.shp (points, one per stream
pixel, built by maxflux_stream_points.py) into line geometry:
maxflux_stream_lines.shp -- one short LineString per pair of raster-
adjacent points (8-connected, i.e. within one pixel-diagonal of each
other), each segment's 'maxflux' field the average of its two endpoints'
values.

This is a purely geometric reconstruction from the points' own raster
adjacency -- it does NOT snap onto or otherwise depend on watershed.py's
separately-thresholded streams_segmented.shp vector network, so
connectivity here reflects exactly whatever mask maxflux_stream_points.py
applied, nothing else. Since that masking (streams-only + hydro-proximity)
can and does remove whole runs of points, the result is generally many
short, disconnected 2-point segments rather than one continuous line per
stream reach.

Both versions are kept: this script only reads the already-built
maxflux_stream_points.shp / all_islands_maxflux_stream_points.shp (run
maxflux_stream_points.py first if they don't exist yet) and writes new
maxflux_stream_lines.shp files alongside them -- it never touches or
replaces the point outputs.
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
# (e.g. across a gap left by masking) are not.
ADJACENCY_FACTOR = 1.5


def points_to_lines(gdf, pixel_size):
    """One 2-point LineString per pair of points within
    ADJACENCY_FACTOR * pixel_size of each other; 'maxflux' = mean of the
    pair's values."""
    coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    tree = cKDTree(coords)
    pairs = sorted(tree.query_pairs(r=ADJACENCY_FACTOR * pixel_size))

    if not pairs:
        return gpd.GeoDataFrame({'maxflux': []}, geometry=[], crs=gdf.crs)

    maxflux = gdf['maxflux'].values
    lines = [LineString([coords[i], coords[j]]) for i, j in pairs]
    values = [(maxflux[i] + maxflux[j]) / 2.0 for i, j in pairs]

    return gpd.GeoDataFrame({'maxflux': values}, geometry=lines, crs=gdf.crs)


def process_island(island):
    points_path = f'{OUTPUT_DIR}{island}/maxflux_stream_points.shp'
    if not os.path.exists(points_path):
        print(f"[{island}] maxflux_stream_points.shp not found, skipping "
              f"(run maxflux_stream_points.py first)")
        return None

    out_path = f'{OUTPUT_DIR}{island}/maxflux_stream_lines.shp'
    if os.path.exists(out_path):
        print(f"[{island}] using existing maxflux_stream_lines.shp")
        return gpd.read_file(out_path)

    gdf = gpd.read_file(points_path)
    with rasterio.open(ISLAND_FILES[island]['maxflux']) as src:
        pixel_size = src.res[0]

    lines_gdf = points_to_lines(gdf, pixel_size)
    lines_gdf.to_file(out_path)
    print(f"[{island}] saved maxflux_stream_lines.shp "
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
        print("No maxflux_stream_lines found for any island -- skipping combined output")
        return

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=COMBINED_CRS)
    out_path = OUTPUT_DIR + 'all_islands_maxflux_stream_lines.shp'
    combined.to_file(out_path)
    print(f"Saved all_islands_maxflux_stream_lines.shp (n = {len(combined)})")


def main():
    per_island = {}
    for island in ISLAND_FILES:
        per_island[island] = process_island(island)
    combine_across_islands(per_island)


if __name__ == '__main__':
    main()
