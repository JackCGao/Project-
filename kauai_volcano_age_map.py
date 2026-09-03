#!/usr/bin/env python3
"""
Kauai-only, higher-resolution version of volcano_age_map.py: zooms into
Kauai's extent and colors its volcanic province (usefulfunctions.
representative_age's source data, built by ageprocessing.py into
volcano_avg_age_regions.shp) by representative volcano age.

The color scale is kept identical to the statewide map (same log-scale
vmin/vmax across all islands) so this figure stays visually comparable to
volcano_age_map.py -- Kauai just fills more of the frame. Kauai has a
single named volcanic province (kaua), so the colored region is one flat
color; areas with no assigned volcanic-province age (VOLC_STAGE outside
{shield, postsh, rejuv} -- alluvium, water, etc.) are drawn first in
grey from the full statewide geology shapefile so the island's true
outline still shows through.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

AGE_SHP = ('/Users/jackgao/SummerWork2026/Jack, Ze-Wen summer project files/'
           'Haw_St_shapefiles/Haw_St_geo_20070426_region_with_age.shp')

VOLCANO_AGE_SHP = ('/Users/jackgao/SummerWork2026/Temp Output Placements/'
                    'volcano_avg_age_regions.shp')

output_dir = '/Users/jackgao/SummerWork2026/Temp Output Placements'

ISLAND = 'Kauai'


def main():
    full_gdf = gpd.read_file(AGE_SHP)
    volcano_gdf = gpd.read_file(VOLCANO_AGE_SHP).to_crs(full_gdf.crs)

    volcano_gdf['AVG_AGE_KYR'] = volcano_gdf['AVG_AGE'] / 1000.0

    # Keep the color scale statewide (all islands) so this zoomed-in figure
    # stays comparable to volcano_age_map.py, then subset to Kauai to draw.
    norm = LogNorm(vmin=volcano_gdf['AVG_AGE_KYR'].min(),
                    vmax=volcano_gdf['AVG_AGE_KYR'].max())
    cmap = plt.cm.viridis

    full_kauai = full_gdf[full_gdf['ISLAND'] == ISLAND]
    volcano_kauai = volcano_gdf[volcano_gdf['ISLAND'] == ISLAND]

    fig, ax = plt.subplots(figsize=(9, 9))

    full_kauai.plot(ax=ax, color='#bfbfbf', edgecolor='none', zorder=1)
    volcano_kauai.plot(ax=ax, column='AVG_AGE_KYR', cmap=cmap, norm=norm,
                        edgecolor='black', linewidth=0.5, zorder=2)

    age_yr = volcano_kauai['AVG_AGE'].iloc[0]
    ax.set_title(f'Kauai Volcanic Province by Representative Age\n'
                 f'({age_yr / 1e6:.2f} Myr, volcano "kaua")')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    ax.set_aspect('equal')

    fig.subplots_adjust(left=0.03, right=0.85, top=0.90, bottom=0.05)

    fig.canvas.draw()
    box = ax.get_position()
    cax = fig.add_axes([0.88, box.y0, 0.03, box.height])

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Representative Volcano Age (kyr, statewide scale)')

    out_path = f'{output_dir}/kauai_volcano_age_map.png'
    plt.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
