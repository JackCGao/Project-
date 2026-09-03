#chmod +x "/Users/jackgao/SummerWork2026/Project-/mars_multi_basin_3d_view.sh"
#!/bin/bash
#
# 3D perspective view of MULTIPLE drainage basins from Mars_Drainage_Gao.shp
# at once (default: FIDs 22 23), draping every selected basin's outline,
# valley/stream network, and paleolakes over a shared MOLA terrain surface,
# built with GMT. Same data sources and rendering approach as
# mars_basin22_3d_view.sh (that script's per-basin sibling) -- this one just
# unions however many basins you pass in instead of handling exactly one.
#
# Elevation: cut directly from 'Ze-Wen Project.mola.nc' (the same global
# MOLA netCDF mars_drainage_map.sh uses) at native 0.0078 deg/px resolution,
# over the padded bounding box of ALL selected basins combined -- still no
# downsampling needed unless you pass in a very large/spread-out basin set.
#
# Streams: Goudge et al. (2021)'s valley network catalog
# (Goudge et al. Data/1_Valley_Catalog_reprojected.tar.gz), same CRS as the
# basin shapefile, no reprojection. Segments tagged
# ValleyType=Removed_Inaccurate_Valley_Interpretations are excluded.
# Clipped to the *union* of the selected basins' exact polygons (not just
# their combined bounding box), so segments in the gaps between
# non-adjacent basins are dropped.
#
# Lakes: Goudge et al. (2021)'s paleolake catalog (Goudge et al. Data/
# 2_Lake_Catalog_reprojected.tar.gz), same CRS, no reprojection. Uses the
# catalog's Breach_Contour_Outline polygons (mapped shoreline extent, not
# the separate Lake_Catalog point layer) so lakes render as filled areas.
# Clipped to the union of selected basins the same way as the valley
# network.
#
# Draping: the basin outlines and clipped valley/lake lines are 2D (lon/lat)
# geometry with no Z of their own, so grdtrack samples the same DEM at
# their vertices to get an elevation, lifted +40m to avoid z-fighting with
# the terrain mesh, before plot3d draws them -- using the same -R/-J/-JZ/-p
# as the grdview call so everything lines up in the same 3D space.
#
# Vertical exaggeration: -JZ2c against -JX18c, same values as
# mars_basin22_3d_view.sh -- if your basin set spans much more relief than
# a single basin typically does, revisit these.
#
# Requires GMT 6 (conda install -c conda-forge gmt).
#
# Usage: ./mars_multi_basin_3d_view.sh [FID ...]   (default: 22 23)
# Output: <OUT_DIR>/mars_basins_<FID>_<FID>..._3d_view.png

set -euo pipefail
export PATH="/Users/jackgao/miniconda3/bin:$PATH"

if [[ $# -eq 0 ]]; then
    FIDS=(12 13 14)   # default basin set
else
    FIDS=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$BASE_DIR/Temp Output Placements"
mkdir -p "$OUT_DIR"

SHP="$BASE_DIR/Mars_Drainage_Gao.shp"
MOLA_SRC="/Users/jackgao/Downloads/Ze-Wen Project.mola.nc"
VALLEY_TAR="$BASE_DIR/Jack, Ze-Wen summer project files/Goudge et al. Data/1_Valley_Catalog_reprojected.tar.gz"
VALLEY_SHP_NAME="Goudge_et_al_2021_Nature_Full_Valley_Catalog_reprojected.shp"
LAKE_TAR="$BASE_DIR/Jack, Ze-Wen summer project files/Goudge et al. Data/2_Lake_Catalog_reprojected.tar.gz"
LAKE_SHP_NAME="Goudge_et_al_2021_Nature_Full_Lake_Catalog_Breach_Contour_Outline_reprojected.shp"
BROWN_LOW="50/25/0"
BROWN_HIGH="235/210/175"

CACHE_DIR="$SCRIPT_DIR/mars_map_work"
VALLEY_DIR="$CACHE_DIR/valley_catalog"
LAKE_DIR="$CACHE_DIR/lake_catalog"
FIDS_JOINED="$(IFS=_; echo "${FIDS[*]}")"
WORKDIR="$CACHE_DIR/basins_${FIDS_JOINED}"
mkdir -p "$VALLEY_DIR" "$LAKE_DIR" "$WORKDIR"

for f in "$SHP" "$MOLA_SRC" "$VALLEY_TAR" "$LAKE_TAR"; do
    if [[ ! -f "$f" ]]; then
        echo "Required input not found: $f" >&2
        exit 1
    fi
done

VALLEY_SHP="$VALLEY_DIR/$VALLEY_SHP_NAME"
if [[ ! -f "$VALLEY_SHP" ]]; then
    echo "Extracting Goudge et al. valley catalog..."
    tar -xzf "$VALLEY_TAR" -C "$VALLEY_DIR"
fi

LAKE_SHP="$LAKE_DIR/$LAKE_SHP_NAME"
if [[ ! -f "$LAKE_SHP" ]]; then
    echo "Extracting Goudge et al. lake catalog..."
    tar -xzf "$LAKE_TAR" -C "$LAKE_DIR"
fi

FIDS_CSV="$(IFS=,; echo "${FIDS[*]}")"

# --- select the basins, union them, clip valley network + lakes to that union ---
echo "Selecting basins FID=[$FIDS_CSV] and clipping valley network + lakes to them..."
python3 - "$SHP" "$VALLEY_SHP" "$LAKE_SHP" "$FIDS_CSV" "$WORKDIR" << 'PYEOF'
import sys
import geopandas as gpd
from shapely.ops import unary_union

shp_path, valley_path, lake_path, fids_csv, workdir = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
)
fids = [int(x) for x in fids_csv.split(',')]

basins = gpd.read_file(shp_path)
match = basins[basins['FID'].isin(fids)]
found = set(match['FID'].tolist())
missing = [fid for fid in fids if fid not in found]
if missing:
    print(f"  WARNING: no basin found for FID(s) {missing} -- skipping those")
if len(match) == 0:
    sys.exit(f"None of FID(s) {fids} found in {shp_path}")

basin_union = unary_union(match.geometry.tolist())
minx, miny, maxx, maxy = match.total_bounds

pad = 0.5
region = (minx - pad, maxx + pad, miny - pad, maxy + pad)

valleys = gpd.read_file(valley_path)
bbox_hits = valleys.cx[region[0]:region[1], region[2]:region[3]]
bbox_hits = bbox_hits[bbox_hits['ValleyType'] != 'Removed_Inaccurate_Valley_Interpretations']
clipped = gpd.clip(bbox_hits, basin_union)
clipped = clipped[~clipped.geometry.is_empty]
print(f"  combined bounds: {minx:.3f}/{maxx:.3f}/{miny:.3f}/{maxy:.3f}  "
      f"({len(match)} basin(s), {len(clipped)} valley segments kept)")

lakes = gpd.read_file(lake_path)
lake_bbox_hits = lakes.cx[region[0]:region[1], region[2]:region[3]]
lakes_clipped = gpd.clip(lake_bbox_hits, basin_union)
lakes_clipped = lakes_clipped[~lakes_clipped.geometry.is_empty]
print(f"  ({len(lakes_clipped)} lake(s) within the selected basins)")

def write_gmt_lines(geoms, path):
    with open(path, 'w') as f:
        for geom in geoms:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'LineString':
                parts = [geom]
            elif geom.geom_type == 'MultiLineString':
                parts = list(geom.geoms)
            elif geom.geom_type == 'Polygon':
                parts = [geom.exterior] + list(geom.interiors)
            elif geom.geom_type == 'MultiPolygon':
                parts = []
                for g in geom.geoms:
                    parts.append(g.exterior)
                    parts.extend(g.interiors)
            else:
                continue
            for part in parts:
                f.write(">\n")
                for x, y in part.coords:
                    f.write(f"{x:.6f} {y:.6f}\n")

write_gmt_lines(match.geometry.tolist(), f'{workdir}/outline.txt')
write_gmt_lines(clipped.geometry.tolist(), f'{workdir}/valleys.txt')
write_gmt_lines(lakes_clipped.geometry.tolist(), f'{workdir}/lakes.txt')
with open(f'{workdir}/region.txt', 'w') as f:
    f.write(f"{region[0]:.6f} {region[1]:.6f} {region[2]:.6f} {region[3]:.6f}\n")
with open(f'{workdir}/found_fids.txt', 'w') as f:
    f.write(','.join(str(fid) for fid in sorted(found)))
PYEOF

read -r RX0 RX1 RY0 RY1 < "$WORKDIR/region.txt"
PADDED_REGION="$RX0/$RX1/$RY0/$RY1"
FOUND_FIDS="$(cat "$WORKDIR/found_fids.txt")"

# --- DEM: cut at native resolution over the combined padded region ---
echo "Cutting MOLA DEM for the combined basin area..."
gmt grdcut "$MOLA_SRC" -R"$PADDED_REGION" -G"$WORKDIR/dem.grd"
gmt grdgradient "$WORKDIR/dem.grd" -Nt1 -A315/45 -G"$WORKDIR/hillshade.grd"

read -r ZMIN ZMAX _ <<< "$(gmt grdinfo -Cn "$WORKDIR/dem.grd" | awk '{print $5, $6}')"
# monochrome brown ramp: dark = low elevation, light = high elevation
gmt makecpt -C"$BROWN_LOW,$BROWN_HIGH" -T"$ZMIN"/"$ZMAX" > "$WORKDIR/elev.cpt"

read -r RX0 RX1 RY0 RY1 _ <<< "$(gmt grdinfo -Cn "$WORKDIR/dem.grd")"
REGION3D="$RX0/$RX1/$RY0/$RY1/$ZMIN/$ZMAX"

# --- drape the outlines + valley lines + lake outlines onto the DEM (sample Z, lift off the mesh a bit) ---
gmt grdtrack "$WORKDIR/outline.txt" -G"$WORKDIR/dem.grd" \
    | awk '{if ($1==">") print; else printf "%s %s %.3f\n", $1, $2, $3+40}' > "$WORKDIR/outline_z.txt"
gmt grdtrack "$WORKDIR/valleys.txt" -G"$WORKDIR/dem.grd" \
    | awk '{if ($1==">") print; else printf "%s %s %.3f\n", $1, $2, $3+40}' > "$WORKDIR/valleys_z.txt"
gmt grdtrack "$WORKDIR/lakes.txt" -G"$WORKDIR/dem.grd" \
    | awk '{if ($1==">") print; else printf "%s %s %.3f\n", $1, $2, $3+40}' > "$WORKDIR/lakes_z.txt"

# --- legend spec (line-color key, since basin outlines vs. valley network vs.
# lakes isn't otherwise labeled anywhere in a 3D perspective scene) ---
cat > "$WORKDIR/legend.txt" << EOF
S 0.1c - 0.3c - 1.5p,white 0.5c Basin outlines ($FOUND_FIDS)
S 0.1c - 0.3c - 1p,cyan 0.5c Valley network (Goudge et al. 2021)
S 0.1c s 0.3c steelblue 0.5p,darkblue 0.5c Paleolake extent (Goudge et al. 2021)
EOF

# --- render ---
MAP_NAME="mars_basins_${FIDS_JOINED}_3d_view"
JX=18c
JZ=2c
PVIEW=200/42

# GMT modern mode's `gmt begin <name>` silently mangles paths containing
# spaces, so cd into the output directory first (same fix as
# hawaii_maxflux_map.sh / mars_drainage_map.sh) -- "Temp Output Placements"
# has spaces in it.
cd "$OUT_DIR"

gmt begin "$MAP_NAME" png
    gmt grdview "$WORKDIR/dem.grd" -R"$REGION3D" -JX"$JX" -JZ"$JZ" -p"$PVIEW" \
        -Qs -C"$WORKDIR/elev.cpt" -I"$WORKDIR/hillshade.grd" \
        -Bxaf+l"Longitude" -Byaf+l"Latitude" -Bzaf+l"Elevation (m)" \
        -BWSneZ+t"Mars Drainage Basins $FOUND_FIDS"
    gmt plot3d "$WORKDIR/outline_z.txt" -R"$REGION3D" -JX"$JX" -JZ"$JZ" -p"$PVIEW" -W1.5p,white
    gmt plot3d "$WORKDIR/valleys_z.txt" -R"$REGION3D" -JX"$JX" -JZ"$JZ" -p"$PVIEW" -W1p,cyan
    if [[ -s "$WORKDIR/lakes_z.txt" ]]; then
        gmt plot3d "$WORKDIR/lakes_z.txt" -R"$REGION3D" -JX"$JX" -JZ"$JZ" -p"$PVIEW" \
            -L -Gsteelblue@40 -W0.5p,darkblue
    fi
    gmt colorbar -DJMR+o1.2c/0+w8c/0.4c -C"$WORKDIR/elev.cpt" -Bxaf+l"Elevation (m)"
    gmt legend "$WORKDIR/legend.txt" -DjTR+w7c+o0.2c/1c -F+p1p+gwhite@20
gmt end

echo "Saved $OUT_DIR/${MAP_NAME}.png"
