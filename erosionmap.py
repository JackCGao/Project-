import rasterio
from rasterio.merge import merge
from rasterio.features import rasterize
from rasterio.crs import CRS
import geopandas as gpd
import numpy as np

path = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Jack, Ze-Wen summer project files/Haw_St_shapefiles/Haw_St_geo_20070426_region_with_age.shp'
files = [
    '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/lanai/new/lanai_erosion_nans.tif',
    '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/molokai/new/molokai_erosion_nans.tif',
    '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/kauai/new/kauai_erosion_nans.tif',
    '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/maui/new/maui_erosion_nans.tif'
]

polygons = gpd.read_file(path)
srcs = [rasterio.open(f) for f in files]
merged, transform = merge(srcs)
raster_crs = srcs[0].crs
height, width = merged.shape[1], merged.shape[2]

merged = merged[0]  # single band

depth_valid_mask = np.isfinite(merged)

# Reproject shapefile to match raster CRS if needed
if polygons.crs != raster_crs:
    polygons = polygons.to_crs(raster_crs)

# Compute average age per polygon
polygons['avg_age'] = (polygons['min_age_yr'] + polygons['max_age_yr']) / 2.0

# Filter to only polygons with confirmed ages (age_notes == 'none')
polygons_certain = polygons[polygons['age_notes'].str.lower().str.strip() == 'none']

def make_age_raster(poly_subset):
    shapes = (
        (geom, val)
        for geom, val in zip(poly_subset.geometry, poly_subset['avg_age'])
        if geom is not None and np.isfinite(val)
    )
    return rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype='float32',
    )

def compute_and_write(age_raster, output_path, label):
    age_valid_mask = np.isfinite(age_raster) & (age_raster > 0)
    valid_mask = depth_valid_mask & age_valid_mask
    erosion_rate = np.full((height, width), np.nan, dtype='float32')
    erosion_rate[valid_mask] = (merged[valid_mask] / age_raster[valid_mask]) * 1_000_000  # m/yr -> m/Mya
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': raster_crs,
        'transform': transform,
        'nodata': np.nan,
    }
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(erosion_rate, 1)
    print(f"{label} written to {output_path}")
    print(f"  Valid pixels: {valid_mask.sum()} / {depth_valid_mask.sum()} depth pixels covered")

base = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao'

# All polygons (includes uncertain and null age_notes)
compute_and_write(
    make_age_raster(polygons),
    f'{base}/erosion_rate_all.tif',
    'All age notes'
)

# Only polygons where age_notes == 'none' (no uncertainty)
compute_and_write(
    make_age_raster(polygons_certain),
    f'{base}/erosion_rate_certain.tif',
    'Certain age notes only'
)