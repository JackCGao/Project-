#!/usr/bin/env python3
"""
Set (define/stamp) a shapefile's CRS to IAU2015 Mars WITHOUT reprojecting —
i.e. this only relabels the coordinate system, it does not move or transform
any coordinates. Use this when a file has no CRS defined, or has the wrong
one recorded, but the coordinate values themselves are already correct.

If you actually need to transform coordinates from one CRS into another,
use the reproject_to_iau_crs.py script instead.

Requires:
    pip install geopandas
"""

import os

import geopandas as gpd

# ---------------------------------------------------------------------------
# EDIT THIS: paste the shapefile path(s) whose CRS you want to set.
# ---------------------------------------------------------------------------
FILE_PATHS = [
    "/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Mars_Drainage_Gao.shp",
]

# Target CRS to stamp onto the file(s).
TARGET_CRS = "IAU_2015:49900"  # Mars (2015) - Sphere, geographic

# If True, overwrite the file in place. If False, save a copy with a suffix.
OVERWRITE = True
OUTPUT_SUFFIX = "crs_set"  # only used if OVERWRITE is False
# ---------------------------------------------------------------------------


def set_crs(shp_path, target_crs, overwrite, suffix):
    gdf = gpd.read_file(shp_path)

    if gdf.crs is not None:
        print(f"  Current CRS: {gdf.crs}")
    else:
        print("  Current CRS: none defined")

    gdf = gdf.set_crs(target_crs, allow_override=True)

    if overwrite:
        out_path = shp_path
    else:
        base, ext = os.path.splitext(shp_path)
        out_path = f"{base}_{suffix}{ext}"

    gdf.to_file(out_path)
    return out_path


def main():
    for shp in FILE_PATHS:
        print(f"[shapefile] {shp}")
        try:
            out = set_crs(shp, TARGET_CRS, OVERWRITE, OUTPUT_SUFFIX)
            print(f"  -> CRS set to {TARGET_CRS}, saved {out}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")


if __name__ == "__main__":
    main()