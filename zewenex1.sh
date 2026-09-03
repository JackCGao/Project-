#!/bin/bash
gmt set PS_LINE_CAP = round
gmt set PS_LINE_JOIN = miter
# gmt set FONT 16p, Helvetica,white
gmt set FONT_LABEL 16p,Helvetica,black
gmt set FONT_ANNOT_PRIMARY 16p,Helvetica,black
gmt set FORMAT_GEO_OUT = D
gmt set FORMAT_GEO_MAP = D
gmt set PS_MEDIA = a2
gmt set FONT_TAG = auto,Helvetica-Bold
gmt set MAP_FRAME_TYPE plain
gmt set MAP_TICK_LENGTH = 0.5c
gmt set MAP_ANNOT_OBLIQUE = tick_extend
gmt set MAP_FRAME_PEN = black
gmt set MAP_TICK_PEN = .75p,black
gmt set PS_PAGE_ORIENTATION = landscape
gmt set COLOR_NAN=white

Rplot=-Rkohala_dem.nc
Jplot="-JX6i"

gmt grdconvert ./flow_acc_kohala/weighted_flow_valleys.tiff+s100 -G./flow_acc_kohala/weighted_flow_valleys_colours.nc
gmt grdconvert ./flow_acc_kohala/weighted_flow_valleys.tiff -G./flow_acc_kohala/weighted_flow_valleys_sizes.nc 

gmt begin flow_acc_kohala_valleys

    # 1. Plot the DEM as a 2D hillshaded image
    gmt grdimage ./kohala_dem.nc $Rplot $Jplot -Cgray -I+d -t25 -B+n

    # 2. Create the CPT for the stream points
    gmt makecpt -Cmagma -T1e8/1e11/1+l -Z -Ic

    # 3. Prepare the data file (this part stays the same)
    # Add 3rd column: elevation
    gmt grdtrack -G./kohala_dem.nc ./flow_acc_kohala/flow_points_valleys.csv $Rplot -s+a > ./flow_acc_kohala/flow_points_tracked_valleys.txt
    # Add 4th column: flow accumulation for color
    gmt grdtrack -G./flow_acc_kohala/weighted_flow_valleys_colours.nc ./flow_acc_kohala/flow_points_tracked_valleys.txt $Rplot -s+a > ./flow_acc_kohala/weighted_flow_points_tracked_valleys.txt
    
    # 4. Plot the points in 2D, skipping the elevation column (2)
    gmt grdtrack -G./flow_acc_kohala/weighted_flow_valleys_sizes.nc ./flow_acc_kohala/weighted_flow_points_tracked_valleys.txt $Rplot -s+a \
        | gmt plot $Rplot $Jplot -Sc -C -i0,1,3,4+l+d180

    # 5. Add scale bar and color bar
    # gmt colorbar -Dx0.1i/0.5i+w4.5c/0.4c+e+h -F+r -Q -By+lm
    gmt colorbar -Dx3i/6i+w4.5c/0.4c+e+h -F+r -Q -By+lm --MAP_TICK_LENGTH=0.2c
    # gmt basemap $Rplot $Jplot -LjBL+o1c/1c+w20k

    gmt basemap $Rplot $Jplot -LjBL+o1c/1c+w5000+l"Pixels"
    # gmt basemap ${Jplot} ${Rplot} -LjBL+o1c/1c+w1k+l"km"
    # gmt basemap ${Jplot} ${Rplot} -Lg124.5E/13.5S+w1k+l"km" --FONT_LABEL=16p,Helvetica,black --FONT_ANNOT_PRIMARY=16p,Helvetica,black --MAP_TICK_PEN_PRIMARY=2.5p,black


gmt end show