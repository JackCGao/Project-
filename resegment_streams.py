#!/usr/bin/env python3
"""
Explode the merged, per-network stream shapefiles (streams.shp, built by
watershed.py's connected-network merge -- one feature per whole connected
drainage system, main stem plus every joining tributary) back into
individual line segments: one feature per maximal non-branching reach,
i.e. the same granularity streams.gpkg had before that merge fused whole
networks together.

For each island, reads watershed_outputs/{island}/streams.shp and writes
watershed_outputs/{island}/streams_segmented.shp, exploding every
LineString/MultiLineString feature into one row per constituent line part.
Also builds a combined all_islands_streams_segmented.shp across every
island, following the same convention as watershed.py's
combine_across_islands.

Outputs are cached per-island so re-running skips islands already done.
"""

import os
import pandas as pd
import geopandas as gpd

from clay_vs_precip import island_files

PROJECT_DIR = '/Users/jackgao/Summer Work 2026/Project-/'
OUTPUT_DIR = PROJECT_DIR + 'watershed_outputs/'
COMBINED_CRS = 'ESRI:102007'


def resegment_island(island):
    out_dir = OUTPUT_DIR + island + '/'
    merged_path = out_dir + 'streams.shp'
    segmented_path = out_dir + 'streams_segmented.shp'

    if not os.path.exists(merged_path):
        print(f"[{island}] streams.shp not found, skipping")
        return

    if os.path.exists(segmented_path):
        print(f"[{island}] using existing streams_segmented.shp")
        return

    merged_gdf = gpd.read_file(merged_path)
    exploded = (
        merged_gdf.rename(columns={'stream_id': 'network_id'})
        .explode(index_parts=False)
        .reset_index(drop=True)
    )
    exploded['seg_id'] = range(len(exploded))
    exploded.to_file(segmented_path)
    print(f"[{island}] saved streams_segmented.shp "
          f"({len(exploded)} segments, from {len(merged_gdf)} networks)")


def combine_across_islands():
    parts = []
    for island in island_files:
        path = OUTPUT_DIR + island + '/streams_segmented.shp'
        if not os.path.exists(path):
            print(f"[{island}] streams_segmented.shp not found, skipping in combined output")
            continue
        gdf = gpd.read_file(path).to_crs(COMBINED_CRS)
        gdf['island'] = island
        parts.append(gdf)

    if not parts:
        print("No streams_segmented.shp found for any island -- skipping combined output")
        return

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=COMBINED_CRS)
    out_path = OUTPUT_DIR + 'all_islands_streams_segmented.shp'
    combined.to_file(out_path)
    print(f"Saved all_islands_streams_segmented.shp  (n = {len(combined)})")


def main():
    for island in island_files:
        resegment_island(island)
    combine_across_islands()


if __name__ == '__main__':
    main()
