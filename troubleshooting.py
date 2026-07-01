#!/usr/bin/env python3
"""
mask_dem.py
Masks the Lanai DEM to the island coastline polygon, setting ocean to NoData.
Run this once before the main watershed pipeline.
"""

import subprocess
import rasterio

# --- Paths ---
project_dir   = '/Users/jackgao/Summer Work 2026/Project-/'
dem_path      = project_dir + 'Lanai_DEM_10m_UTM.tif'
coastline     = project_dir + 'lanai_coastline.gpkg'   # update if you used a different name/format
coastline_reproj = project_dir + 'lanai_coastline_reproj.gpkg'
dem_masked    = project_dir + 'Lanai_DEM_masked.tif'

NODATA_VALUE = -9999

# --- Step 1: check DEM CRS ---
with rasterio.open(dem_path) as src:
    dem_crs = src.crs
    dem_nodata = src.nodata
    print(f"DEM CRS: {dem_crs}")
    print(f"DEM existing NoData: {dem_nodata}")

dem_epsg = dem_crs.to_epsg()
if dem_epsg is None:
    raise RuntimeError("Could not determine EPSG code for DEM CRS — check manually with gdalinfo.")

# --- Step 2: assign CRS to coastline polygon (source had no CRS in metadata, but matches DEM) ---
print(f"Assigning EPSG:{dem_epsg} to coastline (no transform needed — already matches) ...")
subprocess.run([
    'ogr2ogr',
    '-a_srs', f'EPSG:{dem_epsg}',
    '-overwrite',
    coastline_reproj,
    coastline
], check=True)

# --- Step 3: clip DEM to coastline polygon, set ocean to NoData ---
print("Clipping DEM to coastline (masking ocean) ...")
subprocess.run([
    'gdalwarp',
    '-cutline', coastline_reproj,
    '-crop_to_cutline',
    '-dstnodata', str(NODATA_VALUE),
    '-overwrite',
    dem_path,
    dem_masked
], check=True)

# --- Step 4: verify result ---
with rasterio.open(dem_masked) as src:
    print(f"Masked DEM NoData: {src.nodata}")
    data = src.read(1, masked=True)
    print(f"Masked DEM elevation range: {data.min()} to {data.max()}")

print(f"\nDone. Masked DEM saved to: {dem_masked}")
print("Update your pipeline script's dem_path to point to this file.")