import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# =====================================================
# File paths
# =====================================================

csv_file = r"/Users/jackgao/Downloads/Martian Depositional Features.csv"          # <-- Change this
output_shp = r"/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Jack, Ze-Wen summer project files/mars_depositional_features.shp"

# =====================================================
# Read CSV
# =====================================================

df = pd.read_csv(csv_file)

# Check required columns
required_columns = ["Latitude", "Longitude"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Remove rows with missing coordinates
df = df.dropna(subset=["Latitude", "Longitude"])

# =====================================================
# Create point geometry
# =====================================================

geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]

# =====================================================
# Mars Geographic CRS (Mars Sphere)
# Radius = 3,396,190 meters
# =====================================================

mars_geographic = (
    "+proj=longlat "
    "+a=3396190 "
    "+b=3396190 "
    "+no_defs"
)

gdf = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs=mars_geographic
)

# =====================================================
# OPTIONAL: Reproject to Mars Equidistant Cylindrical
# Set REPROJECT = True if you want projected coordinates.
# =====================================================

REPROJECT = False

if REPROJECT:

    mars_eqc = (
        "+proj=eqc "
        "+lat_ts=0 "
        "+lat_0=0 "
        "+lon_0=0 "
        "+a=3396190 "
        "+b=3396190 "
        "+units=m "
        "+no_defs"
    )

    gdf = gdf.to_crs(mars_eqc)

# =====================================================
# Save shapefile
# =====================================================

gdf.to_file(output_shp)

print("--------------------------------")
print("Finished!")
print(f"Number of points: {len(gdf)}")
print(f"CRS: {gdf.crs}")
print(f"Saved to: {output_shp}")
print("--------------------------------")