#!/bin/bash
#
# Aerial-view map of the actual stream network across the main Hawaiian
# Islands, colored by D8 max-flux (flow accumulation), built with GMT.
#
# Each island's *_d8maxflux_albers.tif is already projected (ESRI:102007 /
# Hawaii Albers Equal-Area Conic, meters) -- since that's already an
# equal-area map projection, this plots the projected easting/northing
# directly as a Cartesian frame (-JX) rather than re-projecting through
# GMT's own map-projection engine, so the whole chain lines up correctly
# with no further geographic transform needed.
#
# For each island:
#   1. GMT's GDAL grid importer respects the GeoTIFF's nodata tag (-32768)
#      automatically, so nodata pixels already read in as NaN.
#   2. A small fraction of pixels in some islands' rasters (kahoolawe,
#      molokai, maui) are negative edge artifacts from the D8 algorithm,
#      not real flux values -- grdclip masks any remaining <=0 pixel to
#      NaN too, same Q>0 filter used throughout the project's Python
#      analysis (e.g. maxflux_stream_points.py).
#   3. The matching *_streams_unweighted_albers.tif (same grid, VALUE==1 on
#      stream pixels, NaN elsewhere) is used as a mask so only actual
#      channel pixels keep their max-flux value -- plotting the raw
#      max-flux surface instead just paints the whole island a speckled
#      near-uniform color, since >99% of pixels are low-order hillslope
#      cells; masking to the stream network is what makes the drainage
#      pattern visible.
#   4. Values are log10-transformed (maxflux spans ~10^2 to ~10^10) for a
#      readable color scale.
#   5. A single stream pixel is only ~10m wide -- far thinner than one
#      output-image pixel at this map's scale (islands span the figure at
#      roughly 150-200m/pixel), so a plain grdimage of the masked grid
#      would lose most of the network to resampling. gmt grdfilter -Fu
#      (max-in-window) regrids each island to STREAM_DISP_RES_M spacing,
#      taking the max log10 value within a STREAM_DILATE_M window --
#      this both thickens and downsamples the network in one pass so it
#      survives rendering.
#
# Islands are drawn over a light-gray fill of Coastline.shp (reprojected
# to ESRI:102007 to match) so the land itself is visible, not just the
# stream pixels; grdimage -Q makes the non-stream (NaN) parts of each
# island's grid transparent so that fill shows through.
#
# All islands share one CPT (built from the combined log10 range) and one
# region/projection, so they're plotted as layers of a single figure.
#
# Requires GMT 6 (conda install -c conda-forge gmt) with GDAL/OGR support.
#
# Usage: ./hawaii_maxflux_map.sh
# Output: <OUT_DIR>/hawaii_maxflux_map.png

set -euo pipefail
export PATH="/Users/jackgao/miniconda3/bin:$PATH"

# Derive the project paths from the script's own location instead of a
# hardcoded absolute path -- the folder this repo lives under has been
# renamed before (it used to be "Summer Work 2026", with spaces) and a
# stale hardcoded BASE_DIR silently no-ops the whole script: every *_tif
# lookup below just misses, ${#STREAM_GRIDS[@]} stays 0, and it exits with
# "No stream grids found".
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$BASE_DIR/Temp Output Placements"
mkdir -p "$OUT_DIR"

COASTLINE_SHP="$SCRIPT_DIR/Coastline.shp"

# regrid spacing / dilation window for making the 1-pixel-wide stream
# network survive downsampling to the final map scale (see header note 5)
STREAM_DISP_RES_M=150
STREAM_DILATE_M=450

ISLANDS=(hawaii kahoolawe oahu kauai lanai molokai maui)
TIFS=(
    "$BASE_DIR/new (1)/hawaii_d8maxflux_albers.tif"
    "$BASE_DIR/new/kahoolawe_d8maxflux_albers.tif"
    "$BASE_DIR/oahu/new/oahu_d8maxflux_albers.tif"
    "$BASE_DIR/kauai/new/kauai_d8maxflux_albers.tif"
    "$BASE_DIR/lanai/new/lanai_d8maxflux_albers.tif"
    "$BASE_DIR/molokai/new/molokai_d8maxflux_albers.tif"
    "$BASE_DIR/maui/new/maui_d8maxflux_albers.tif"
)
STREAM_TIFS=(
    "$BASE_DIR/new (1)/hawaii_streams_unweighted_albers.tif"
    "$BASE_DIR/new/kahoolawe_streams_unweighted_albers.tif"
    "$BASE_DIR/oahu/new/oahu_streams_unweighted_albers.tif"
    "$BASE_DIR/kauai/new/kauai_streams_unweighted_albers.tif"
    "$BASE_DIR/lanai/new/lanai_streams_unweighted_albers.tif"
    "$BASE_DIR/molokai/new/molokai_streams_unweighted_albers.tif"
    "$BASE_DIR/maui/new/maui_streams_unweighted_albers.tif"
)

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

INFO_FILE="$WORKDIR/grid_info.txt"
LABELS_FILE="$WORKDIR/island_labels.txt"
> "$INFO_FILE"
> "$LABELS_FILE"

STREAM_GRIDS=()

echo "Building island outline layer..."
COASTLINE_GMT="$WORKDIR/coastline_albers.gmt"
if [[ -f "$COASTLINE_SHP" ]]; then
    ogr2ogr -f "OGR_GMT" "$COASTLINE_GMT" -t_srs ESRI:102007 "$COASTLINE_SHP"
else
    echo "  Coastline.shp not found at $COASTLINE_SHP -- skipping island fill" >&2
    COASTLINE_GMT=""
fi

echo "Building log10(max-flux) stream-network grids..."
for i in "${!ISLANDS[@]}"; do
    island="${ISLANDS[$i]}"
    tif="${TIFS[$i]}"
    stream_tif="${STREAM_TIFS[$i]}"

    if [[ ! -f "$tif" ]]; then
        echo "  [$island] maxflux tif not found, skipping: $tif" >&2
        continue
    fi
    if [[ ! -f "$stream_tif" ]]; then
        echo "  [$island] streams tif not found, skipping: $stream_tif" >&2
        continue
    fi

    clipped="$WORKDIR/${island}_clipped.grd"
    mask="$WORKDIR/${island}_streammask.grd"
    log10="$WORKDIR/${island}_streams_log10.grd"
    coarse="$WORKDIR/${island}_streams_log10_coarse.grd"

    # mask non-positive pixels (nodata + a few edge-artifact negatives) to NaN
    gmt grdclip "$tif" -Sb0/NaN+e -G"$clipped"
    # binarize the streams raster to {1, NaN} -- nodata is already NaN via
    # the GDAL import, this just guards against any stray non-stream value
    gmt grdclip "$stream_tif" -Sb0.5/NaN+e -G"$mask"
    # keep max-flux only on stream pixels (mask multiply propagates NaN
    # everywhere else), then log10-transform
    gmt grdmath "$clipped" "$mask" MUL LOG10 = "$log10"
    # thicken + downsample so the network survives rendering (header note 5)
    gmt grdfilter "$log10" -Fu"$STREAM_DILATE_M" -D0 -I"$STREAM_DISP_RES_M" -Ni -G"$coarse"
    STREAM_GRIDS+=("$coarse")

    line=$(gmt grdinfo -Cn "$coarse")
    echo "$line" >> "$INFO_FILE"

    read -r x0 x1 y0 y1 _ <<< "$line"
    awk -v x0="$x0" -v x1="$x1" -v y0="$y0" -v y1="$y1" -v name="$island" \
        'BEGIN{printf "%.3f %.3f %s\n", (x0+x1)/2, (y0+y1)/2, name}' >> "$LABELS_FILE"

    echo "  [$island] done"
done

if [[ ${#STREAM_GRIDS[@]} -eq 0 ]]; then
    echo "No stream grids found -- nothing to plot." >&2
    exit 1
fi

# --- combined region and color range across every island ---
XMIN=$(awk '{print $1}' "$INFO_FILE" | sort -g | head -1)
XMAX=$(awk '{print $2}' "$INFO_FILE" | sort -g | tail -1)
YMIN=$(awk '{print $3}' "$INFO_FILE" | sort -g | head -1)
YMAX=$(awk '{print $4}' "$INFO_FILE" | sort -g | tail -1)
ZMIN=$(awk '{print $5}' "$INFO_FILE" | sort -g | head -1)
ZMAX=$(awk '{print $6}' "$INFO_FILE" | sort -g | tail -1)

REGION="$XMIN/$XMAX/$YMIN/$YMAX"

WIDTH_CM=22
HEIGHT_CM=$(awk -v xmin="$XMIN" -v xmax="$XMAX" -v ymin="$YMIN" -v ymax="$YMAX" -v w="$WIDTH_CM" \
    'BEGIN{printf "%.3f", w*(ymax-ymin)/(xmax-xmin)}')

echo "Region: $REGION   Width/height: ${WIDTH_CM}c/${HEIGHT_CM}c   log10(max-flux) range: $ZMIN to $ZMAX"

MAP_NAME="hawaii_maxflux_map"

# GMT modern mode's `gmt begin <name>` silently mangles paths containing
# spaces (e.g. strips them when building the psconvert output path), so
# cd into the output directory first and pass a bare filename rather than
# an absolute path -- "Temp Output Placements" has spaces in it.
cd "$OUT_DIR"

gmt begin "$MAP_NAME" png
    gmt basemap -R"$REGION" -JX"${WIDTH_CM}c/${HEIGHT_CM}c" \
        -Baf -BWSen+t"Hawaiian Islands -- Streams by D8 Max Flux"
    if [[ -n "$COASTLINE_GMT" ]]; then
        gmt plot "$COASTLINE_GMT" -G230 -W0.3p,gray40
    fi
    gmt makecpt -Cviridis -T"$ZMIN"/"$ZMAX"
    for g in "${STREAM_GRIDS[@]}"; do
        gmt grdimage "$g" -Q
    done
    gmt text "$LABELS_FILE" -F+f9p,Helvetica-Bold,white=0.5p,black+jCB -Dj0/0.15c
    gmt colorbar -DJBC+w"${WIDTH_CM}"c/0.4c+h+o0/1c -Bxaf+l"log@-10@-(stream max flux, m@+3@+/yr)"
gmt end

echo "Saved $OUT_DIR/${MAP_NAME}.png"
