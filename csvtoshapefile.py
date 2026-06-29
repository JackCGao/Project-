import geopandas as gp
import pandas as pd
from geodatasets import get_path

#File Names
path = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Jack, Ze-Wen summer project files/Haw_St_shapefiles/Haw_St_geo_20070426_region.shp'
path_csv = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Jack, Ze-Wen summer project files/hawaii_geo_table_processed.csv'




#Accessing the Files
file = gp.read_file(path)
csv_adding_coloumns = pd.read_csv(path_csv, usecols=['min_age_yr','max_age_yr','age_notes'])
csv_adding_coloumns = csv_adding_coloumns.round({'min_age_yr': 2, 'max_age_yr': 2})
file = file.merge(csv_adding_coloumns, left_index=True, right_index=True)

file.to_file('/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Jack, Ze-Wen summer project files/Haw_St_shapefiles/Haw_St_geo_20070426_region_with_age.shp')





