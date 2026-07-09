import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import os

base_dir   = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao'
output_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao/Temp Output Placements'

_recharge_base = f'{base_dir}/Jack, Ze-Wen summer project files/Groundwater Recharge Files'
RECHARGE_SHPS = {
    'kauai':   f'{_recharge_base}/Kauai Water Budget Components 2020/Kauai_water_budget_components_subarea_inches.shp',
    'lanai':   f'{_recharge_base}/Lanai Water Budget Components 2020/Lanai_water_budget_components_subarea_inches.shp',
    'molokai': f'{_recharge_base}/Molokai Water Budget Components 2020/Molokai_water_budget_components_subarea_inches.shp',
    'maui':    f'{_recharge_base}/Maui Water Budget Components 2020/Maui_water_budget_components_subarea_inches.shp',
}

FIELD = 'S1_Tot_rc'   # in/yr
IN_TO_M = 0.0254      # in/yr → m/yr

for island, shp_path in RECHARGE_SHPS.items():
    gdf = gpd.read_file(shp_path)

    if FIELD not in gdf.columns:
        print(f"[{island}] field '{FIELD}' not found — skipping")
        continue

    gdf[FIELD] = pd.to_numeric(gdf[FIELD], errors='coerce')
    gdf['recharge_m'] = gdf[FIELD] * IN_TO_M

    valid = gdf['recharge_m'].dropna()
    vmin, vmax = valid.min(), valid.max()
    mean_val   = valid.mean()

    fig, ax = plt.subplots(figsize=(8, 6))

    gdf.plot(
        column='recharge_m',
        ax=ax,
        cmap='YlGnBu',
        vmin=vmin,
        vmax=vmax,
        edgecolor='none',
        legend=False,
    )

    sm = cm.ScalarMappable(cmap='YlGnBu', norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Groundwater Recharge (m/yr)', fontsize=10)

    ax.set_title(f'{island.capitalize()} — S1_Tot_rc', fontsize=12)
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    ax.tick_params(labelsize=8)

    ax.text(0.02, 0.02,
            f"mean = {mean_val:.4f} m/yr\n"
            f"min  = {vmin:.4f} m/yr\n"
            f"max  = {vmax:.4f} m/yr",
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{island}_recharge_map.png')
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
