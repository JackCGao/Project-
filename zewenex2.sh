
#!/bin/bash
gmt set PS_LINE_CAP = round
gmt set PS_LINE_JOIN = miter
...
gmt set FONT_TAG = auto,Helvetica-Bold
etc
​
# Input files, can export excel into text (tab-delimited txt is best)
# "dos2unix file_name" can help convert windows data file for unix/mac scripting
# Use GDAL ogr2ogr to convert shapefile to a gmt format
indir="/Users/nasa2/Desktop/gmt_plotting"
infile="test.txt"
​
# Make a colour palette, based on seismic colour scheme, 0-100, in intervals of 10 
gmt makecpt -Cseis -T0/100/10 -Z > seis.cpt
​
# Plotting EXAMPLE 1
# Need to indicate variables with $, and () can make it clearer
# -R for region (x and y axis limits)
# -J for projection type, X for normal x/y with no lat/long projection, this one
# says 5 inch by 5 inch
# -B for orders, capitalise the ones with ticks and numbers, lowercase otherwise
# -Bx specifies x axis, a2f1 means 2 units between labels, 1 unit between ticks
# -S for symbol, c for circle, s for square, etc. 0.2c for 0.2cm 
# -G to fill the symbol, red in this case
gmt plot $(infile) -R0/10/2/9 -JX5i/5i -BSWne -Bxa2f1+l"X axis" -Bya2f1+l"Y axis" -Sc0.2c,black -Gred -pdf test1
​
#let's say you want to use column 1 and 3 of your data file, then you can use
awk '{print $1,$3}' $(infile) | # you don't need to input the file name again, | is pipeline operator
gmt plot -R0/10/2/9 -JX5i/5i -BSWne -Bxa2f1+l"X axis" -Bya2f1+l"Y axis" -Sc0.2c,black -Gred -pdf test1
​
# or if you want conditionals
awk '{if ($1>5 && $1<150) print $1,$3}' $(infile) | 
gmt plot -R0/10/2/9 -JX5i/5i -BSWne -Bxa2f1+l"X axis" -Bya2f1+l"Y axis" -Sc0.2c,black -Gred -pdf test1
​
# EXAMPLE 2
gmt begin fig #2x1 is 2 row, 1 column
    gmt subplot begin 2x1 etc
        gmt set PROJ_ELLIPSOID=Mars
        gmt subplot set 0,0 ....
        gmt histogram etc
​
        gmt subplot set 1,0 ...
        gmt gridimage ./grids/mars_dem.tif etc 
        gmt rose ...
    gmt subplot end
gmt end show
​
# BEFORE RUNNING THIS SCRIPT NEED TO RUN "chmod u+x example.gmt"