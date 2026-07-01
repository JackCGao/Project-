import xarray as xr
import rioxarray

url = "https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/usgs_dem_10m_lanai"

dem = xr.open_dataset(url)["elev"]

dem = dem.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
dem = dem.rio.write_crs("EPSG:4326")

# Reproject to UTM Zone 4N
dem_utm = dem.rio.reproject("EPSG:32604")

dem_utm.rio.to_raster("Lanai_DEM_10m_UTM.tif")