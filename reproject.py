#!/usr/bin/env python3
"""
Batch-reproject shapefiles (.shp) and GeoTIFFs (.tif/.tiff) to a Mars
planetary CRS (IAU2000 or IAU2015 family).

Just paste folder paths and/or individual file paths into CRS_PATHS below,
set your target CRS, and run:
    python3 reproject_to_iau_crs.py

Requires:
    pip install geopandas rasterio

IAU_2015 codes require a reasonably recent PROJ/GDAL under the hood
(PROJ >= 6.3, ideally 9+). If pyproj/rasterio can't find "IAU_2015:49900",
update PROJ (e.g. `brew upgrade proj` on Mac, or `conda install -c conda-forge proj`).
"""

import glob
import os

import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ---------------------------------------------------------------------------
# EDIT THIS: paste folder paths and/or full file paths (.shp, .tif, .tiff).
# Folders are searched recursively for both file types.
# ---------------------------------------------------------------------------
CRS_PATHS = [
    # NOTE: Haw_St_shapefiles is deliberately NOT here -- it's real Hawaii
    # (Earth, NAD83 UTM Zone 4N) geology data used as-is by recreation.py's
    # rasterize_age(). Reprojecting it to a Mars IAU CRS is a cross-celestial-
    # body transform; PROJ correctly refuses it, and every past run of this
    # script has been silently no-op'ing on that folder as a result.
    '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Jack, Ze-Wen summer project files/Luo et al. Data',
    '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Jack, Ze-Wen summer project files/Goudge et al. Data',
    '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Jack, Ze-Wen summer project files/Hynek et al. Data',
    '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Jack, Ze-Wen summer project files/Zaki et al. Data',
    '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Mars_Drainage_Gao.shp',
    '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Jack, Ze-Wen summer project files/mars_depositional_features.shp',
]

# ---------------------------------------------------------------------------
# EDIT THIS: same idea, but for DEMs / other continuous-value rasters
# (elevation, slope, etc). These get bilinear resampling (DEM_RESAMPLING
# below) instead of nearest-neighbor, since nearest-neighbor introduces
# blocky artifacts in continuous data. Only .tif/.tiff are looked for here;
# any .shp found in these paths is ignored.
# ---------------------------------------------------------------------------
DEM_PATHS = [
    '/Volumes/Crucial X10/mars_data/Mars_MO_THEMIS-IR-Day_mosaic_global_100m_v12_PROJ.tif',
    '/Volumes/Crucial X10/mars_data/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif',
]

# Resampling method for DEM_PATHS rasters: "bilinear" or "cubic".
# ("cubic" is smoother but slower; "bilinear" is the common default for DEMs.)
DEM_RESAMPLING = "bilinear"

# Target CRS family: "iau2000" or "iau2015"
TARGET = "iau2015"

# Override TARGET with a specific code if you need a non-default variant
# (e.g. the ellipsoid instead of the sphere, or a projected CRS).
CRS_CODE = None  # e.g. "IAU_2015:49911"

# If some of your files have NO CRS defined at all, set this to the CRS
# they're actually already in so the script can reproject correctly.
# Leave as None if all files already have a CRS embedded.
SOURCE_CRS = None  # e.g. "ESRI:104905"

# Where to save output. None = save next to each original file.
OUT_DIR = None

# Whether to append a suffix to output filenames (keeps originals safe).
ADD_SUFFIX = True
OUTPUT_SUFFIX = "reprojected"

# ---------------------------------------------------------------------------

DEFAULT_TARGET_CRS = {    # Mars 2000 (Sphere), geographic
    "iau2015": "IAU_2015:49900",  # Mars (2015) - Sphere, geographic
}


def find_files(paths, include_shp=True):
    shp_files, tif_files = [], []
    exts = ("*.shp", "*.tif", "*.tiff") if include_shp else ("*.tif", "*.tiff")
    for p in paths:
        if os.path.isdir(p):
            for ext in exts:
                found = glob.glob(os.path.join(p, "**", ext), recursive=True)
                for f in found:
                    (shp_files if f.lower().endswith(".shp") else tif_files).append(f)
        elif os.path.isfile(p):
            lower = p.lower()
            if include_shp and lower.endswith(".shp"):
                shp_files.append(p)
            elif lower.endswith((".tif", ".tiff")):
                tif_files.append(p)
            else:
                print(f"Skipping (not a recognized file type): {p}")
        else:
            print(f"Skipping (path not found): {p}")
    return sorted(set(shp_files)), sorted(set(tif_files))


def output_path(src_path, label, out_dir, add_suffix):
    base, ext = os.path.splitext(os.path.basename(src_path))
    out_folder = out_dir or os.path.dirname(src_path) or "."
    os.makedirs(out_folder, exist_ok=True)
    name = f"{base}_{label}{ext}" if add_suffix else f"{base}{ext}"
    return os.path.join(out_folder, name)


def reproject_shapefile(shp_path, target_crs, out_dir, add_suffix, source_crs):
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        if source_crs is None:
            print("  WARNING: no CRS defined and SOURCE_CRS is not set - skipping "
                  "(set SOURCE_CRS to the file's true current CRS to fix this)")
            return None
        gdf = gdf.set_crs(source_crs, allow_override=True)

    reprojected = gdf.to_crs(target_crs)
    out_path = output_path(shp_path, OUTPUT_SUFFIX, out_dir, add_suffix)
    reprojected.to_file(out_path)
    return out_path


RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}


def reproject_raster(tif_path, target_crs, out_dir, add_suffix, source_crs, resampling="nearest"):
    resampling_method = RESAMPLING_METHODS[resampling]
    with rasterio.open(tif_path) as src:
        src_crs = src.crs or source_crs
        if src_crs is None:
            print("  WARNING: no CRS defined and SOURCE_CRS is not set - skipping "
                  "(set SOURCE_CRS to the file's true current CRS to fix this)")
            return None

        transform, width, height = calculate_default_transform(
            src_crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(crs=target_crs, transform=transform, width=width, height=height)

        out_path = output_path(tif_path, OUTPUT_SUFFIX, out_dir, add_suffix)
        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling_method,
                )
    return out_path


def main():
    target_crs = CRS_CODE or DEFAULT_TARGET_CRS[TARGET]

    shp_files, tif_files = find_files(CRS_PATHS)
    _, dem_files = find_files(DEM_PATHS, include_shp=False)
    # Avoid double-processing if a path shows up in both lists
    dem_files = [f for f in dem_files if f not in set(tif_files)]

    total = len(shp_files) + len(tif_files) + len(dem_files)
    if total == 0:
        print("No .shp or .tif/.tiff files found in CRS_PATHS / DEM_PATHS.")
        return

    print(f"Found {len(shp_files)} shapefile(s), {len(tif_files)} raster(s), "
          f"and {len(dem_files)} DEM(s).")
    print(f"Reprojecting to {target_crs}...\n")

    for shp in shp_files:
        print(f"[shapefile] {shp}")
        try:
            out = reproject_shapefile(shp, target_crs, OUT_DIR, ADD_SUFFIX, SOURCE_CRS)
            if out:
                print(f"  -> saved {out}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")

    for tif in tif_files:
        print(f"[raster] {tif}")
        try:
            out = reproject_raster(tif, target_crs, OUT_DIR, ADD_SUFFIX, SOURCE_CRS,
                                    resampling="nearest")
            if out:
                print(f"  -> saved {out}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")

    for dem in dem_files:
        print(f"[dem] {dem}")
        try:
            out = reproject_raster(dem, target_crs, OUT_DIR, ADD_SUFFIX, SOURCE_CRS,
                                    resampling=DEM_RESAMPLING)
            if out:
                print(f"  -> saved {out}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")


if __name__ == "__main__":
    main()