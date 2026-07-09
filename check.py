import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rasterio')

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = '/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao'
islands = ['lanai', 'molokai', 'maui', 'kauai']
output_dir = '/Users/jackgao/Desktop'

def load_clean(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        extent = rasterio.plot.plotting_extent(src)
    return arr, extent

import rasterio.plot

for island in islands:
    island_dir = os.path.join(base_dir, island, 'new')
    slopes_path = os.path.join(island_dir, f'{island}_slope_nans.tif')

    if not os.path.exists(slopes_path):
        print(f"⚠️  Skipping {island}: missing {slopes_path}")
        continue

    slope, extent = load_clean(slopes_path)
    slope_mask = np.isfinite(slope)

    n_valid = slope_mask.sum()
    n_total = slope_mask.size
    print(f"{island.capitalize()}: {n_valid} / {n_total} valid slope pixels "
          f"({100*n_valid/n_total:.1f}%)")

    valid_rows, valid_cols = np.where(slope_mask)
    if valid_rows.size > 0:
        print(f"  row range: {valid_rows.min()}–{valid_rows.max()} (of 0–{slope.shape[0]-1})")
        print(f"  col range: {valid_cols.min()}–{valid_cols.max()} (of 0–{slope.shape[1]-1})")

    # --- Plot 1: binary mask, valid vs nodata ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].imshow(slope_mask, cmap='gray', extent=extent, origin='upper')
    axes[0].set_title(f'{island.capitalize()}: Slope Valid-Data Mask\n'
                       f'({n_valid} / {n_total} pixels, {100*n_valid/n_total:.1f}%)')
    axes[0].set_xlabel('Easting (m)')
    axes[0].set_ylabel('Northing (m)')

    # --- Plot 2: actual slope values (only where valid), for context ---
    im = axes[1].imshow(slope, cmap='viridis', extent=extent, origin='upper')
    axes[1].set_title(f'{island.capitalize()}: Slope Values (where valid)')
    axes[1].set_xlabel('Easting (m)')
    axes[1].set_ylabel('Northing (m)')
    plt.colorbar(im, ax=axes[1], label='Slope (degrees)', shrink=0.8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{island}_slope_mask_check.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}\n")