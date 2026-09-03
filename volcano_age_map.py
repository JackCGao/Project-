#!/usr/bin/env python3
"""
Statewide map of Hawaii's volcanic provinces, colored by representative
volcano age (usefulfunctions.representative_age's source data, built by
ageprocessing.py into volcano_avg_age_regions.shp) -- each named volcano is
its own color, keyed to a continuous (log-scale) age colorbar.

Areas with no assigned volcanic-province age -- VOLC_STAGE outside
{shield, postsh, rejuv} (alluvium, water, etc.) -- are drawn first in
grey from the full statewide geology shapefile, so the true outline of
each island still shows through underneath the colored regions.
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

# Derived from this file's own location rather than hardcoded, since the
# project folder has been renamed/moved before (used to live under a
# Dropbox path) and a stale hardcoded path here silently no-ops every path
# built from it -- same caveat as ageprocessing.py's base_dir.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AGE_SHP = (f'{base_dir}/Jack, Ze-Wen summer project files/Haw_St_shapefiles/'
           'Haw_St_geo_20070426_region_with_age.shp')

VOLCANO_AGE_SHP = f'{base_dir}/Temp Output Placements/volcano_avg_age_regions.shp'

output_dir = f'{base_dir}/Temp Output Placements'


def main():
    full_gdf = gpd.read_file(AGE_SHP)
    volcano_gdf = gpd.read_file(VOLCANO_AGE_SHP).to_crs(full_gdf.crs)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Base layer: full statewide geology footprint in grey -- shows the true
    # island shape, including areas with no assigned volcanic-province age.
    full_gdf.plot(ax=ax, color='#bfbfbf', edgecolor='none', zorder=1)

    # Overlay: each volcanic province, colored by its representative age.
    volcano_gdf['AVG_AGE_KYR'] = volcano_gdf['AVG_AGE'] / 1000.0

    cmap = plt.cm.viridis
    norm = LogNorm(vmin=volcano_gdf['AVG_AGE_KYR'].min(), vmax=volcano_gdf['AVG_AGE_KYR'].max())
    volcano_gdf.plot(ax=ax, column='AVG_AGE_KYR', cmap=cmap, norm=norm,
                      edgecolor='black', linewidth=0.3, zorder=2)

    ax.set_title('Hawaii Volcanic Provinces by Representative Age')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    ax.set_aspect('equal')

    # Reserve margin for the colorbar up front -- tight_layout would otherwise
    # stretch this wide, short map to the figure edge and push it off-canvas.
    fig.subplots_adjust(left=0.03, right=0.85, top=0.93, bottom=0.05)

    # Colorbar axes sized to exactly match the map box's height (aspect='equal'
    # shrinks that box at draw time, so its final position must be read after).
    fig.canvas.draw()
    box = ax.get_position()
    cax = fig.add_axes([0.88, box.y0, 0.03, box.height])

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Representative Volcano Age (kyr)')

    out_path = f'{output_dir}/volcano_age_map.png'
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
