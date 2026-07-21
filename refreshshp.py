#!/usr/bin/env python3
"""
Find the most recently modified version of the Drainage_Basins_Gao shapefile
(including Dropbox "conflicted copy" / "view-only conflicts" duplicates) and
save a copy of the full shapefile fileset under the name Mars_Drainage_Gao.

Only uses the Python standard library, so no pip installs are needed.

Usage:
    python3 find_latest_drainage_basin.py [source_dir] [dest_dir]

If no arguments are given, it searches your Dropbox "Jack Gao" folder and
writes the output back into that same folder.
"""

import datetime
import glob
import os
import shutil
import sys

# Common shapefile sidecar extensions to copy alongside the .shp
SHAPEFILE_EXTENSIONS = [
    ".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx",
    ".shp.xml", ".qix", ".fbn", ".fbx", ".ain", ".aih",
]

DEFAULT_SOURCE_DIR = (
    "/Users/jackgao/Library/CloudStorage/Dropbox-Jackgaoc/Jack Gao"
)
BASE_NAME = "Drainage_Basins_Gao"
NEW_BASE_NAME = "Mars_Drainage_Gao"


def find_candidate_shp_files(source_dir, base_name=BASE_NAME):
    """Find every .shp file whose name starts with base_name, including
    Dropbox conflicted-copy variants."""
    pattern = os.path.join(source_dir, f"{base_name}*.shp")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"No files matching '{base_name}*.shp' found in {source_dir}"
        )
    return candidates


def pick_latest(candidates):
    """Pick the file with the most recent modification time."""
    return max(candidates, key=os.path.getmtime)


def copy_shapefile_set(shp_path, dest_dir, new_base_name=NEW_BASE_NAME):
    """Copy the .shp and all matching sidecar files to dest_dir under new_base_name."""
    src_dir = os.path.dirname(shp_path)
    src_base = os.path.basename(shp_path)[:-4]  # strip ".shp"

    os.makedirs(dest_dir, exist_ok=True)
    copied = []
    for ext in SHAPEFILE_EXTENSIONS:
        src_file = os.path.join(src_dir, src_base + ext)
        if os.path.exists(src_file):
            dest_file = os.path.join(dest_dir, new_base_name + ext)
            shutil.copy2(src_file, dest_file)
            copied.append(dest_file)
    return copied


def main():
    source_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE_DIR
    dest_dir = sys.argv[2] if len(sys.argv) > 2 else source_dir

    candidates = find_candidate_shp_files(source_dir)

    print(f"Found {len(candidates)} candidate file(s) in {source_dir}:\n")
    for c in sorted(candidates, key=os.path.getmtime, reverse=True):
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(c)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"  {ts}  {os.path.basename(c)}")

    latest = pick_latest(candidates)
    print(f"\nLatest version (by modification time): {os.path.basename(latest)}")

    copied = copy_shapefile_set(latest, dest_dir)
    print(f"\nSaved as '{NEW_BASE_NAME}' in {dest_dir}:")
    for f in copied:
        print(f"  {os.path.basename(f)}")


if __name__ == "__main__":
    main()