import dataretrieval.nwis as nwis
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# 1. Get all HI streamgage sites with discharge
sites, _ = nwis.get_info(
    stateCd='HI',
    parameterCd='00060',
    siteType='ST',
    hasDataTypeCd='dv'
)
print(f"Found {len(sites)} sites")

# 2. Pull daily discharge values and compute mean + year count per site
site_list = sites['site_no'].tolist()
results = []

for i, site in enumerate(site_list):
    print(f"[{i+1}/{len(site_list)}] Querying {site}...", end=' ')
    try:
        dv, _ = nwis.get_dv(
            sites=site,
            parameterCd='00060',
            start='1900-01-01',
            end='2024-12-31'
        )
        if dv.empty:
            print("no data")
            continue
        
        # Find the discharge column (usually named like '00060_Mean')
        q_cols = [c for c in dv.columns if '00060' in c]
        if not q_cols:
            print("no discharge column")
            continue
        
        q = pd.to_numeric(dv[q_cols[0]], errors='coerce').dropna()
        
        if q.empty:
            print("all NaN")
            continue
        
        # Count unique years with data
        years = q.index.year.nunique()
        mean_cfs = q.mean()
        start_year = q.index.year.min()
        end_year = q.index.year.max()
        
        results.append({
            'site_no': site,
            'mean_Q_cfs': mean_cfs,
            'n_years': years,
            'start_yr': start_year,
            'end_yr': end_year
        })
        print(f"OK — {years} years, mean={mean_cfs:.2f} cfs")
        
    except Exception as e:
        print(f"error: {e}")
        continue

print(f"\nSuccessfully retrieved data for {len(results)} sites")

# 3. Build results dataframe
results_df = pd.DataFrame(results)

# 4. Merge with site info
merged = sites.merge(results_df, on='site_no', how='inner')  # inner = only valid

# 5. Clean coordinates
merged = merged.dropna(subset=['dec_lat_va', 'dec_long_va'])
merged['dec_lat_va'] = merged['dec_lat_va'].astype(float)
merged['dec_long_va'] = merged['dec_long_va'].astype(float)

if 'geometry' in merged.columns:
    merged = merged.drop(columns=['geometry'])

# 6. Convert cfs to m³/yr
# 1 cfs = 0.028317 m³/s × 86400 s/day × 365.25 days/yr
CFS_TO_M3YR = 0.028316846592 * 86400 * 365.25

clean = pd.DataFrame({
    'site_no':    merged['site_no'].astype(str),
    'name':       merged['station_nm'].astype(str).str[:80],
    'lat':        merged['dec_lat_va'],
    'lon':        merged['dec_long_va'],
    'alt_ft':     pd.to_numeric(merged['alt_va'], errors='coerce'),
    'drain_sqmi': pd.to_numeric(merged['drain_area_va'], errors='coerce'),
    'mean_Q_cfs': merged['mean_Q_cfs'],
    'mean_Q_m3y': merged['mean_Q_cfs'] * CFS_TO_M3YR,
    'n_years':    merged['n_years'].astype(int),
    'start_yr':   merged['start_yr'].astype(int),
    'end_yr':     merged['end_yr'].astype(int)
})

geometry = [Point(xy) for xy in zip(clean['lon'], clean['lat'])]
gdf = gpd.GeoDataFrame(clean, geometry=geometry, crs='EPSG:4326')

# 7. Export
gdf.to_file('HI_gages_discharge_daily.gpkg', driver='GPKG')
gdf.to_file('HI_gages_discharge_daily.shp')
print(f"\nSaved {len(gdf)} gages")
print(gdf[['site_no', 'name', 'mean_Q_cfs', 'mean_Q_m3y', 'n_years']].head(10).to_string())