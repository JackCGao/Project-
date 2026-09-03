#chmod +x "/Users/jackgao/SummerWork2026/Project-/mars_drainage_map.sh"
#!/bin/bash
#
# Global map of Mars_Drainage_Gao.shp (30 hand-delineated drainage basins,
# all drawn in one uniform light blue) plus Zaki et al.'s
# Large_drainage_systems_reprojected.shp (16 large drainage systems, in the
# same orange used for mars_zaki_drainage_map.sh) over a MOLA hillshade
# backdrop, built with GMT.
#
# Mars_Drainage_Gao.shp only carries an FID field -- no ratio/metric exists
# yet to classify basins into bins the way a reference figure might (e.g.
# "0.05-0.37", "0.38-0.46", ...), so all 30 basins are drawn identically
# (translucent light-blue fill + thin outline); Zaki's 16 systems likewise
# have no field worth classing by (just X, Y, Area). A legend distinguishes
# the two datasets since otherwise nothing on the map would.
#
# Background: 'Ze-Wen Project.mola.nc' is a global MOLA grid (0.0078 deg/px,
# 46263x22670) in GMT-native netCDF form. GDAL can't open it directly in
# this environment (netCDF4/HDF5-backed, and the gdal_HDF5 plugin isn't
# installed), so unlike the old *_reprojected.tif source this script used
# to read via gdal_translate, downsampling here goes through GMT's own
# grdfilter instead (which reads the file natively) -- averaged down to
# 0.25 deg/px with a 25km Gaussian low-pass window, cached in CACHE_DIR so
# this only has to happen once.
#
# Projection note: -JQ (GMT's true Equidistant Cylindrical projection)
# reliably crashes Ghostscript ("/undefinedresult in --charpath--" on the
# "100" degree-annotation glyph) once a grdimage/plot layer is added on top
# of a global -R-180/180/-90/90 frame, in this GMT 6.6.0 / Ghostscript
# 10.07.1 setup. -JX (Cartesian, treating lon/lat degrees as a linear grid
# -- the same approach hawaii_maxflux_map.sh uses for projected meters)
# renders identically for an equirectangular figure like this one and does
# not trigger it, so that's used here instead. This also means GMT's own
# geographic map-scale (-L) isn't available (Cartesian-only limitation), so
# the scale bar is drawn manually from the Mars sphere radius.
#
# Requires GMT 6 (conda install -c conda-forge gmt).
#
# Usage: ./mars_drainage_map.sh
# Output: <OUT_DIR>/mars_drainage_basins_map.png

set -euo pipefail
export PATH="/Users/jackgao/miniconda3/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$BASE_DIR/Temp Output Placements"
mkdir -p "$OUT_DIR"

SHP="$BASE_DIR/Mars_Drainage_Gao.shp"
ZAKI_ZIP="$BASE_DIR/Jack, Ze-Wen summer project files/Zaki et al. Data/Large_drainage_systems_reprojected.zip"
ZAKI_SHP_NAME="Large_drainage_systems_reprojected.shp"
MOLA_SRC="/Users/jackgao/Downloads/Ze-Wen Project.mola.nc"
MARS_RADIUS_M=3396190
BASIN_FILL="120/175/230"
BASIN_OUTLINE="10/40/90"
BASIN_TRANSPARENCY=20
ZAKI_FILL="255/130/0"
ZAKI_OUTLINE="140/60/0"
ZAKI_TRANSPARENCY=15

CACHE_DIR="$SCRIPT_DIR/mars_map_work"
mkdir -p "$CACHE_DIR"
DOWNSAMPLED="$CACHE_DIR/zewen_mola_downsampled.grd"
HILLSHADE="$CACHE_DIR/zewen_hillshade.grd"
GRAY_CPT="$CACHE_DIR/mola_gray_elev.cpt"
ZAKI_DIR="$CACHE_DIR/zaki_drainage"
mkdir -p "$ZAKI_DIR"

if [[ ! -f "$SHP" ]]; then
    echo "Drainage shapefile not found: $SHP" >&2
    exit 1
fi
if [[ ! -f "$ZAKI_ZIP" ]]; then
    echo "Zaki drainage systems zip not found: $ZAKI_ZIP" >&2
    exit 1
fi

ZAKI_SHP="$ZAKI_DIR/$ZAKI_SHP_NAME"
if [[ ! -f "$ZAKI_SHP" ]]; then
    echo "Extracting Zaki et al. drainage systems shapefile..."
    unzip -oq "$ZAKI_ZIP" -d "$ZAKI_DIR"
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

# actual data region of the downsampled grid (this MOLA product doesn't
# quite reach the poles -- roughly -88.2/88.2 lat, not a clean -90/90 --
# so the frame is built from what's actually there instead of assuming
# full coverage)
read -r RX0 RX1 RY0 RY1 _ <<< "$(gmt grdinfo -Cn "$DOWNSAMPLED")"
REGION="$RX0/$RX1/$RY0/$RY1"

N_BASINS=$(ogrinfo -al -so "$SHP" | awk -F': ' '/Feature Count/{print $2}')
N_ZAKI=$(ogrinfo -al -so "$ZAKI_SHP" | awk -F': ' '/Feature Count/{print $2}')

# --- legend: one swatch per dataset ---
LEGEND_FILE="$CACHE_DIR/gao_zaki_legend.txt"
cat > "$LEGEND_FILE" << EOF
S 0.1c s 0.3c $BASIN_FILL 0.25p,$BASIN_OUTLINE 0.5c Mars_Drainage_Gao basins (n=$N_BASINS)
S 0.1c s 0.3c $ZAKI_FILL 0.25p,$ZAKI_OUTLINE 0.5c Zaki et al. drainage systems (n=$N_ZAKI)
EOF

# --- manual scale bar (2000 km at the equator, in Mars-sphere degrees) ---
# GMT's own -L map scale needs a true geographic projection (-JQ), which is
# exactly the projection that crashes here (see header note), so this is
# built from plain lon/lat points instead of GMT's scale-bar machinery.
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
MAP_NAME="mars_drainage_basins_map"

# GMT modern mode's `gmt begin <name>` silently mangles paths containing
# spaces, so cd into the output directory first (same fix as
# hawaii_maxflux_map.sh) -- "Temp Output Placements" has spaces in it.
cd "$OUT_DIR"

gmt begin "$MAP_NAME" png
    gmt basemap -R"$REGION" -JX24c/12c -Bxafg30+l"Longitude" -Byafg30+l"Latitude" \
        -BWSen+t"Mars Drainage Basins"
    gmt grdimage "$DOWNSAMPLED" -C"$GRAY_CPT" -I"$HILLSHADE"
    gmt plot "$SHP" -G"$BASIN_FILL" -t"$BASIN_TRANSPARENCY" -W0.75p,"$BASIN_OUTLINE"
    gmt plot "$ZAKI_SHP" -G"$ZAKI_FILL" -t"$ZAKI_TRANSPARENCY" -W0.75p,"$ZAKI_OUTLINE"
    gmt plot "$WORKDIR/scalebar_line.txt" -W2p,white
    gmt plot "$WORKDIR/scalebar_ticks.txt" -W1.5p,white
    gmt text "$WORKDIR/scalebar_labels.txt" -F+f8p,Helvetica,white+jCT
    gmt legend "$LEGEND_FILE" -DjTR+w8c+o0.2c/1c -F+p1p+gwhite@10
gmt end

echo "Saved $OUT_DIR/${MAP_NAME}.png"
