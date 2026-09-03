#!/bin/bash
#
# Global map of Luo et al.'s mars_watershed_with_h_reprojected.shp (2562
# watersheds) over a MOLA hillshade backdrop, built with GMT. Drawn as
# plain basin outlines in one uniform color -- no attribute classification
# (the shapefile's 'n_exponent' field isn't used here; see
# mars_luo_watershed_map.sh's git history / prior version for the 6-class
# Jenks-natural-breaks choropleth built from it, matched to this project's
# original reference figure).
#
# Background: reuses the same cached MOLA grid mars_drainage_map.sh builds
# from 'Ze-Wen Project.mola.nc' (CACHE_DIR/zewen_mola_downsampled.grd) --
# built fresh here too if that cache doesn't exist yet.
#
# Projection note: -JQ (GMT's true Equidistant Cylindrical projection)
# reliably crashes Ghostscript ("/undefinedresult in --charpath--" on the
# "100" degree-annotation glyph) once a grdimage/plot layer is added on top
# of a global -R-180/180/-90/90 frame, in this GMT 6.6.0 / Ghostscript
# 10.07.1 setup. -JX (Cartesian, treating lon/lat degrees as a linear grid)
# renders identically for an equirectangular figure like this and does not
# trigger it, so that's used here instead (same fix as mars_drainage_map.sh).
#
# Requires GMT 6 (conda install -c conda-forge gmt).
#
# Usage: ./mars_luo_watershed_map.sh
# Output: <OUT_DIR>/mars_luo_watershed_map.png

set -euo pipefail
export PATH="/Users/jackgao/miniconda3/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$BASE_DIR/Temp Output Placements"
mkdir -p "$OUT_DIR"

SHP="$BASE_DIR/Jack, Ze-Wen summer project files/Luo et al. Data/mars_watershed_with_h_reprojected.shp"
MOLA_SRC="/Users/jackgao/Downloads/Ze-Wen Project.mola.nc"
MARS_RADIUS_M=3396190
FILL="20/210/90"
OUTLINE="5/50/20"
FILL_TRANSPARENCY=15
OUTLINE_WIDTH=0.4p

CACHE_DIR="$SCRIPT_DIR/mars_map_work"
mkdir -p "$CACHE_DIR"
DOWNSAMPLED="$CACHE_DIR/zewen_mola_downsampled.grd"
HILLSHADE="$CACHE_DIR/zewen_hillshade.grd"
GRAY_CPT="$CACHE_DIR/mola_gray_elev.cpt"

if [[ ! -f "$SHP" ]]; then
    echo "Luo watershed shapefile not found: $SHP" >&2
    exit 1
fi

# --- background: downsample MOLA once, cache it, compute hillshade ---
if [[ ! -f "$DOWNSAMPLED" ]]; then
    if [[ ! -f "$MOLA_SRC" ]]; then
        echo "MOLA source not found: $MOLA_SRC" >&2
        echo "and no cached downsample at $DOWNSAMPLED either -- nothing to use as background." >&2
        exit 1
    fi
    echo "Downsampling MOLA netCDF to 0.25 deg/px (cached for next time)..."
    gmt grdfilter "$MOLA_SRC" -Fg25 -D2 -I0.25 -Ni -G"$DOWNSAMPLED"
else
    echo "Using cached downsampled MOLA DEM: $DOWNSAMPLED"
fi

if [[ ! -f "$HILLSHADE" ]]; then
    gmt grdgradient "$DOWNSAMPLED" -Nt1 -A315/45 -G"$HILLSHADE"
fi
gmt makecpt -Cgray -T-8500/21500 -H > "$GRAY_CPT"

read -r RX0 RX1 RY0 RY1 _ <<< "$(gmt grdinfo -Cn "$DOWNSAMPLED")"
REGION="$RX0/$RX1/$RY0/$RY1"

N_WATERSHEDS=$(ogrinfo -al -so "$SHP" | awk -F': ' '/Feature Count/{print $2}')

# --- manual scale bar (2000 km at the equator, in Mars-sphere degrees) ---
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
read -r KM_PER_DEG DEG_1000 DEG_2000 <<< "$(awk -v R="$MARS_RADIUS_M" 'BEGIN{
    kmdeg = 2*3.14159265358979*R/360/1000
    printf "%.6f %.6f %.6f", kmdeg, 1000/kmdeg, 2000/kmdeg
}')"
LON0=-170
LAT0=-80
awk -v lon0="$LON0" -v lat0="$LAT0" -v d1="$DEG_1000" -v d2="$DEG_2000" \
    'BEGIN{printf "%f %f\n%f %f\n%f %f\n", lon0,lat0, lon0+d1,lat0, lon0+d2,lat0}' \
    > "$WORKDIR/scalebar_line.txt"
awk -v lon0="$LON0" -v lat0="$LAT0" -v d1="$DEG_1000" -v d2="$DEG_2000" 'BEGIN{
    printf ">\n%f %f\n%f %f\n", lon0,lat0+1, lon0,lat0-1
    printf ">\n%f %f\n%f %f\n", lon0+d1,lat0+1, lon0+d1,lat0-1
    printf ">\n%f %f\n%f %f\n", lon0+d2,lat0+1, lon0+d2,lat0-1
}' > "$WORKDIR/scalebar_ticks.txt"
awk -v lon0="$LON0" -v lat0="$LAT0" -v d1="$DEG_1000" -v d2="$DEG_2000" 'BEGIN{
    printf "%f %f 0\n%f %f 1000\n%f %f 2000 km\n", lon0,lat0-2.2, lon0+d1,lat0-2.2, lon0+d2,lat0-2.2
}' > "$WORKDIR/scalebar_labels.txt"

# --- render ---
MAP_NAME="mars_luo_watershed_map"

# GMT modern mode's `gmt begin <name>` silently mangles paths containing
# spaces, so cd into the output directory first.
cd "$OUT_DIR"

gmt begin "$MAP_NAME" png
    gmt basemap -R"$REGION" -JX24c/12c -Bxafg30+l"Longitude" -Byafg30+l"Latitude" \
        -BWSen+t"Luo et al. Mars Watersheds"
    gmt grdimage "$DOWNSAMPLED" -C"$GRAY_CPT" -I"$HILLSHADE"
    gmt plot "$SHP" -G"$FILL" -t"$FILL_TRANSPARENCY" -W"$OUTLINE_WIDTH","$OUTLINE"
    gmt plot "$WORKDIR/scalebar_line.txt" -W2p,white
    gmt plot "$WORKDIR/scalebar_ticks.txt" -W1.5p,white
    gmt text "$WORKDIR/scalebar_labels.txt" -F+f8p,Helvetica,white+jCT
gmt end

echo "Saved $OUT_DIR/${MAP_NAME}.png"
