import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
import geopandas as gpd
import pandas as pd
from rasterio.features import rasterize as rio_rasterize
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import rasterio
import whitebox_workflows as wbw
import seaborn as sns
import os

# Derived from this file's own location rather than hardcoded, since the
# project folder has been renamed before (it used to be "Summer Work 2026",
# with spaces) and a stale hardcoded path here silently no-ops every path
# built from it.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AGE_SHP = (f'{_BASE_DIR}/Jack, Ze-Wen summer project files/'
           'Haw_St_shapefiles/Haw_St_geo_20070426_region_with_age.shp')

STREAMS_SHP = f'{_BASE_DIR}/Hawaii Streams/Streams_reprojected.shp'

_streams_gdf = None

def _load_streams_gdf():
    global _streams_gdf
    if _streams_gdf is None:
        _streams_gdf = gpd.read_file(STREAMS_SHP)
    return _streams_gdf

def stream_mask(ref_path, island=None):
    """Boolean raster mask, True wherever a stream line (from the statewide
    Streams_reprojected.shp hydrography layer, pre-reprojected to
    ESRI:102007 -- the common CRS every other island raster/vector in this
    project uses) passes through the pixel, on the grid of `ref_path`. This
    dataset has no per-island field; `island` is accepted for backwards
    compatibility but unused -- rasterizing directly onto `ref_path`'s own
    grid/extent already restricts the mask to that island's streams."""
    with rasterio.open(ref_path) as src:
        transform, crs, width, height = src.transform, src.crs, src.width, src.height

    gdf = _load_streams_gdf()
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    burned = rio_rasterize(
        ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='uint8',
    )
    return burned.astype(bool)

VOLC_STAGES = ['shield', 'postsh']

def load_clean(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nd = src.nodata
        if nd is not None:
            if np.isnan(nd):
                arr[~np.isfinite(arr)] = np.nan
            else:
                arr[np.isclose(arr, nd, rtol=0, atol=1e-6)] = np.nan
    return arr

def masking(path, mask_path):
    mask = load_clean(mask_path)
    data = load_clean(path)
    
    valid_mask = np.isfinite(mask) & np.isfinite(data)
    masked_data = np.where(valid_mask, data, np.nan)
    return masked_data

def reproject_raster(src_path, dst_path, dst_crs, resampling_method=Resampling.nearest):
    with rasterio.open(src_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method
                )


    """
    Assign weighted-average surface clay (%) from SSURGO to points.

    Parameters
    ----------
    gdb_path : str
        Path to the SSURGO geodatabase.
    xs, ys : array-like
        Point coordinates in target_crs.
    target_crs : str
        CRS of the input coordinates.

    Returns
    -------
    np.ndarray
        Clay percentage for each point (NaN if outside all polygons).
    """

    # Component table
    comp = pyogrio.read_dataframe(
        gdb_path,
        layer="component",
        columns=["mukey", "cokey", "comppct_r"],
        read_geometry=False,
    )

    # Horizon table
    hor = pyogrio.read_dataframe(
        gdb_path,
        layer="chorizon",
        columns=["cokey", "hzdept_r", "claytotal_r"],
        read_geometry=False,
    )

    # Surface horizon for each component
    surf = hor.loc[hor.groupby("cokey")["hzdept_r"].idxmin()]
    surf = surf.dropna(subset=["claytotal_r"])

    # Join with component table
    merged = surf.merge(comp, on="cokey")

    # Component-percentage weighted clay
    merged["weighted"] = merged["claytotal_r"] * merged["comppct_r"]

    clay = (
        merged.groupby("mukey")[["weighted", "comppct_r"]]
        .sum()
    )

    clay["clay_pct"] = clay["weighted"] / clay["comppct_r"]

    # Load polygons
    mupoly = pyogrio.read_dataframe(
        gdb_path,
        layer="MUPOLYGON",
        columns=["MUKEY"],
    )

    # Join clay values
    mupoly = (
        mupoly.merge(
            clay[["clay_pct"]],
            left_on="MUKEY",
            right_index=True,
            how="inner",
        )
        .to_crs(target_crs)
    )

    # Build point GeoDataFrame
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xs, ys),
        crs=target_crs,
    )

    # Point-in-polygon join
    joined = gpd.sjoin(
        points,
        mupoly[["clay_pct", "geometry"]],
        how="left",
        predicate="within",
    )

    return joined["clay_pct"].to_numpy()

def clayraster(
    gdb_path,
    output_path,
    resolution=30,
    target_crs="ESRI:102007"
):
    """
    Create SSURGO weighted surface clay percentage raster.

    Parameters
    ----------
    gdb_path : str
        Path to gSSURGO geodatabase.
    output_path : str
        Output GeoTIFF path.
    resolution : float
        Pixel size in CRS units (meters for ESRI:102007).
    target_crs : str
        Projection for output raster.

    Returns
    -------
    None
    """

    # -----------------------------
    # Build MUKEY -> clay %
    # -----------------------------

    comp = pyogrio.read_dataframe(
        gdb_path,
        layer="component",
        columns=["mukey", "cokey", "comppct_r"],
        read_geometry=False,
    )

    hor = pyogrio.read_dataframe(
        gdb_path,
        layer="chorizon",
        columns=["cokey", "hzdept_r", "claytotal_r"],
        read_geometry=False,
    )


    # Surface horizon
    surf = hor.loc[
        hor.groupby("cokey")["hzdept_r"].idxmin()
    ]

    surf = surf.dropna(
        subset=["claytotal_r"]
    )


    merged = surf.merge(
        comp,
        on="cokey",
        how="inner"
    )


    merged["weighted_clay"] = (
        merged["claytotal_r"] *
        merged["comppct_r"]
    )


    clay = (
        merged
        .groupby("mukey")
        [["weighted_clay", "comppct_r"]]
        .sum()
    )


    clay["clay_pct"] = (
        clay["weighted_clay"] /
        clay["comppct_r"]
    )


    # -----------------------------
    # Load SSURGO polygons
    # -----------------------------

    mupoly = pyogrio.read_dataframe(
        gdb_path,
        layer="MUPOLYGON",
        columns=["MUKEY"],
    )


    mupoly = mupoly.merge(
        clay[["clay_pct"]],
        left_on="MUKEY",
        right_index=True,
        how="inner"
    )


    mupoly = mupoly.to_crs(
        target_crs
    )


    # -----------------------------
    # Create raster grid
    # -----------------------------

    bounds = mupoly.total_bounds

    xmin, ymin, xmax, ymax = bounds


    width = int(
        np.ceil((xmax - xmin) / resolution)
    )

    height = int(
        np.ceil((ymax - ymin) / resolution)
    )


    transform = from_bounds(
        xmin,
        ymin,
        xmax,
        ymax,
        width,
        height
    )


    # -----------------------------
    # Rasterize clay values
    # -----------------------------

    shapes = (
        (geom, value)
        for geom, value
        in zip(
            mupoly.geometry,
            mupoly.clay_pct
        )
    )


    clay_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype="float32"
    )


    # -----------------------------
    # Save GeoTIFF
    # -----------------------------

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": target_crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate"
    }


    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            clay_raster,
            1
        )


    print(
        f"Saved clay raster: {output_path}"
    )

_age_gdf = None

def _load_age_gdf():
    """Volcanic-province polygons (VOLC_STAGE in {shield, postsh}) with
    per-polygon age = (min_age_yr + max_age_yr) / 2, same convention as
    recreation.py's rasterize_age() / ageprocessing.py."""
    global _age_gdf
    if _age_gdf is None:
        gdf = gpd.read_file(AGE_SHP)
        gdf = gdf[gdf['VOLC_STAGE'].isin(VOLC_STAGES)].copy()
        gdf['avg_age'] = (
            pd.to_numeric(gdf['min_age_yr'], errors='coerce') +
            pd.to_numeric(gdf['max_age_yr'], errors='coerce')
        ) / 2.0
        _age_gdf = gdf[gdf['avg_age'] > 0]
    return _age_gdf

def nearest_age(xs, ys, crs):
    """Point-level core of nearestage(): for arbitrary point coordinates (in
    `crs`), finds each point's nearest raw geology polygon (by centroid
    distance) in the volcanic-province age shapefile and returns its
    avg_age (years)."""
    gdf = _load_age_gdf().to_crs(crs)
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])
    tree = cKDTree(coords)

    points = np.column_stack([xs, ys])
    _, idx = tree.query(points, k=1)
    return gdf['avg_age'].values[idx]


def nearestage(raster):
    """
    Compare an already-opened raster to the volcanic-province age shapefile
    and return the nearest age (years) for every valid pixel.

    For each non-nodata pixel in `raster`, finds the nearest volcanic
    province polygon (by centroid distance, same nearest-neighbor approach
    as clay_vs_erosion.py) and looks up its age. See nearest_age() for the
    same lookup against arbitrary point coordinates rather than a whole
    raster.

    Parameters
    ----------
    raster : rasterio.DatasetReader
        A loaded raster, e.g. `rasterio.open(path)`.

    Returns
    -------
    np.ndarray
        Array the same shape as `raster`, with the nearest volcanic-province
        age (years) at every valid pixel and NaN elsewhere.
    """
    data = raster.read(1).astype(np.float64)
    nd = raster.nodata
    if nd is not None:
        if np.isnan(nd):
            data[~np.isfinite(data)] = np.nan
        else:
            data[np.isclose(data, nd, atol=1e-6)] = np.nan

    valid_mask = np.isfinite(data)
    rows, cols = np.where(valid_mask)
    xs, ys = rasterio.transform.xy(raster.transform, rows, cols)

    age_arr = np.full(data.shape, np.nan)
    age_arr[rows, cols] = nearest_age(xs, ys, raster.crs)
    return age_arr

VOLCANO_AGE_SHP = f'{_BASE_DIR}/Temp Output Placements/volcano_avg_age_regions.shp'

_volcano_age_gdf = None

def _load_volcano_age_gdf():
    """Per-volcano dissolved regions with a representative AVG_AGE field
    (mean of shield+postsh polygon ages per named volcano), built by
    ageprocessing.py."""
    global _volcano_age_gdf
    if _volcano_age_gdf is None:
        _volcano_age_gdf = gpd.read_file(VOLCANO_AGE_SHP)
    return _volcano_age_gdf

def nearest_volcano_age(xs, ys, crs, return_tags=False):
    """Point-level core of representative_age(): for arbitrary point
    coordinates (in `crs`), finds each point's nearest volcano region (by
    centroid distance) in volcano_avg_age_regions.shp and returns that
    volcano's AVG_AGE (years), and optionally its VOLCANO tag.

    Parameters
    ----------
    xs, ys : array-like
        Point coordinates in `crs`.
    crs : CRS
        CRS of `xs`, `ys`.
    return_tags : bool
        If True, also return the matched VOLCANO tag per point.

    Returns
    -------
    np.ndarray, or (np.ndarray, np.ndarray) if return_tags
        Matched AVG_AGE (years) per point, and optionally matched VOLCANO tags.
    """
    gdf = _load_volcano_age_gdf().to_crs(crs)
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])
    tree = cKDTree(coords)

    points = np.column_stack([xs, ys])
    _, idx = tree.query(points, k=1)
    matched_age = gdf['AVG_AGE'].values[idx]

    if not return_tags:
        return matched_age
    return matched_age, gdf['VOLCANO'].values[idx]


def representative_age(raster, return_tags=False):
    """
    Like nearestage(), but returns each pixel's REPRESENTATIVE age -- the
    average age of the whole volcano it belongs to (dissolved shield+postsh
    polygons per named volcano, see ageprocessing.py) -- rather than the age
    of the single nearest raw geology polygon.

    For each valid pixel, finds the nearest volcano region (by centroid
    distance) in volcano_avg_age_regions.shp and returns that volcano's
    AVG_AGE (years). See nearest_volcano_age() for the same lookup against
    arbitrary point coordinates rather than a whole raster.

    Parameters
    ----------
    raster : rasterio.DatasetReader
        A loaded raster, e.g. `rasterio.open(path)`.
    return_tags : bool
        If True, also return an array of the matched VOLCANO tag at every
        valid pixel (for reporting which volcano/age was used where).

    Returns
    -------
    np.ndarray, or (np.ndarray, np.ndarray) if return_tags
        Array the same shape as `raster`, with the nearest volcano's
        representative age (years) at every valid pixel and NaN elsewhere.
        If return_tags, also an object array of matched VOLCANO tags
        (empty string where invalid).
    """
    data = raster.read(1).astype(np.float64)
    nd = raster.nodata
    if nd is not None:
        if np.isnan(nd):
            data[~np.isfinite(data)] = np.nan
        else:
            data[np.isclose(data, nd, atol=1e-6)] = np.nan

    valid_mask = np.isfinite(data)
    rows, cols = np.where(valid_mask)
    xs, ys = rasterio.transform.xy(raster.transform, rows, cols)

    age_arr = np.full(data.shape, np.nan)
    if not return_tags:
        age_arr[rows, cols] = nearest_volcano_age(xs, ys, raster.crs)
        return age_arr

    matched_age, matched_tag = nearest_volcano_age(xs, ys, raster.crs, return_tags=True)
    age_arr[rows, cols] = matched_age
    tag_arr = np.full(data.shape, '', dtype=object)
    tag_arr[rows, cols] = matched_tag
    return age_arr, tag_arr


def sample_raster_at_points(path, xs, ys, crs=None):
    """Sample a single-band raster at arbitrary point coordinates, returning
    NaN for points outside the raster's extent or on a nodata pixel.

    Parameters
    ----------
    path : str
        Path to the raster.
    xs, ys : array-like
        Point coordinates, in `crs` if given, else assumed already in the
        raster's own CRS.
    crs : CRS, optional
        CRS of `xs`, `ys`. If it differs from the raster's CRS, points are
        reprojected before sampling.

    Returns
    -------
    np.ndarray
        Sampled value per point (NaN where invalid).
    """
    with rasterio.open(path) as src:
        if crs is not None and crs != src.crs:
            reproj = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=crs).to_crs(src.crs)
            xs_use, ys_use = reproj.x.values, reproj.y.values
        else:
            xs_use, ys_use = np.asarray(xs), np.asarray(ys)

        rows, cols = rasterio.transform.rowcol(src.transform, xs_use, ys_use)
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        in_bounds = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)

        data = src.read(1)
        values = np.full(len(xs_use), np.nan)
        values[in_bounds] = data[rows[in_bounds], cols[in_bounds]]

        nodata = src.nodata
        if nodata is not None:
            if np.isnan(nodata):
                values[~np.isfinite(values)] = np.nan
            else:
                values[np.isclose(values, nodata, rtol=0, atol=1e-6)] = np.nan
    return values

def roughnessmeasure(rough_path):
    # still temp
    return 0
