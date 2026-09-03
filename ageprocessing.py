#!/usr/bin/env python3
"""
Volcano-tagged average age regions, from the Hawaii statewide geology-with-age
shapefile.

- Read Haw_St_geo_20070426_region_with_age.shp, look at VOLC_STAGE.
- Keep only VOLC_STAGE in {shield, postsh, rejuv} (excludes alluv, water).
- Age per polygon = (min_age_yr + max_age_yr) / 2, same convention as
  volcanic_province_age.py / recreation.py's rasterize_age().
- Group polygons by their volcano tag (VOLCANO field, case-normalized so
  e.g. "Hale"/"hale" merge into one tag) and take the average age per group.
- Dissolve (merge) each group's polygons into one multipolygon feature and
  write out a new shapefile with an AVG_AGE field.
"""

import os
import geopandas as gpd
import pandas as pd

# Derived from this file's own location rather than hardcoded, since the
# project folder has been renamed before (it used to be "Summer Work 2026",
# with spaces) and a stale hardcoded path here silently no-ops every path
# built from it -- same caveat as recreation.py's PROJECT_DIR.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AGE_SHP = (f'{base_dir}/Jack, Ze-Wen summer project files/Haw_St_shapefiles/'
           'Haw_St_geo_20070426_region_with_age.shp')

output_dir = f'{base_dir}/Temp Output Placements'
OUT_SHP = f'{output_dir}/volcano_avg_age_regions.shp'
OUT_CSV = f'{base_dir}/Project-/volcano_avg_age.csv'

VOLC_STAGES = ['shield', 'postsh', 'rejuv']


def main():
    gdf = gpd.read_file(AGE_SHP)

    volcanic = gdf[gdf['VOLC_STAGE'].isin(VOLC_STAGES)].copy()
    print(f"{len(volcanic)}/{len(gdf)} polygons have VOLC_STAGE in {VOLC_STAGES}")

    volcanic['avg_age'] = (
        pd.to_numeric(volcanic['min_age_yr'], errors='coerce') +
        pd.to_numeric(volcanic['max_age_yr'], errors='coerce')
    ) / 2.0
    volcanic = volcanic[volcanic['avg_age'] > 0]
    print(f"{len(volcanic)} polygons have a valid positive age")

    volcanic['volcano_tag'] = volcanic['VOLCANO'].str.lower()

    avg_age = (
        volcanic.groupby(['ISLAND', 'volcano_tag'])['avg_age']
        .mean()
        .rename('AVG_AGE')
    )

    dissolved = (
        volcanic.dissolve(by=['ISLAND', 'volcano_tag'])
        .join(avg_age)
        .reset_index()[['ISLAND', 'volcano_tag', 'AVG_AGE', 'geometry']]
        .rename(columns={'volcano_tag': 'VOLCANO'})
    )

    dissolved.to_file(OUT_SHP)
    print(f"\nSaved {OUT_SHP} ({len(dissolved)} volcano regions)")

    avg_age.sort_values().to_csv(OUT_CSV)
    print(f"Saved {OUT_CSV}")

    print("\nAverage age (years) by volcano region:")
    print(avg_age.sort_values().to_string())


if __name__ == '__main__':
    main()
