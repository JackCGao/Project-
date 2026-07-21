#!/usr/bin/env python3
"""
Median geologic age per island, restricted to volcanic-province polygons.

"Volcanic province" = VOLC_STAGE in {shield, postsh, rejuv} (the three
Hawaiian volcanic eruptive stages), excluding non-volcanic units like
alluvium (VOLC_STAGE = 'alluv') and open water (VOLC_STAGE = 'water').

Age per polygon = (min_age_yr + max_age_yr) / 2, same convention as
recreation.py's rasterize_age().
"""

import geopandas as gpd
import pandas as pd

AGE_SHP = ('/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/'
           'Jack, Ze-Wen summer project files/Haw_St_shapefiles/'
           'Haw_St_geo_20070426_region_with_age.shp')

VOLCANIC_STAGES = ['shield', 'postsh', 'rejuv']


def main():
    gdf = gpd.read_file(AGE_SHP)

    volcanic = gdf[gdf['VOLC_STAGE'].isin(VOLCANIC_STAGES)].copy()
    print(f"{len(volcanic)}/{len(gdf)} polygons are volcanic-province "
          f"(VOLC_STAGE in {VOLCANIC_STAGES})")

    volcanic['avg_age'] = (
        pd.to_numeric(volcanic['min_age_yr'], errors='coerce') +
        pd.to_numeric(volcanic['max_age_yr'], errors='coerce')
    ) / 2.0
    volcanic = volcanic[volcanic['avg_age'] > 0]

    median_age = (
        volcanic.groupby('ISLAND')['avg_age']
        .median()
        .sort_values()
        .rename('median_age_yr')
    )

    print("\nMedian volcanic-province age by island (years):")
    print(median_age.to_string())

    out_csv = '/Users/jackgao/Summer Work 2026/Project-/volcanic_province_median_age.csv'
    median_age.to_csv(out_csv)
    print(f"\nSaved {out_csv}")


if __name__ == '__main__':
    main()
