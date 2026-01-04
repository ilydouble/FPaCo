#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to synchronize detection JSON files from a Source Dataset (e.g., fingerA)
to a Target Dataset (e.g., fingerB).

UPDATED LOGIC:
1. Index Source: Scan ALL JSON files in Source directory and map {filename: full_path}.
   This allows matching even if 'train'/'val'/'test' splits are different.
2. Scan Target: Iterate through all images in Target Dataset.
3. Match: Look up the corresponding JSON filename in the Source Index.
4. If found: Copy Source JSON -> Target JSON.
5. If NOT found: Delete existing Target JSON (to allow fresh regeneration).
"""

import argparse
import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to Source dataset (e.g., ../datasets/fingerA)")
    parser.add_argument("--dst", type=str, required=True, help="Path to Target dataset (e.g., ../datasets/fingerB)")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    if not src_root.exists():
        print(f"Error: Source dir {src_root} does not exist.")
        return
    if not dst_root.exists():
        print(f"Error: Target dir {dst_root} does not exist.")
        return

    print(f"Indexing Source JSONs from: {src_root} ...")
    # 1. Build Source Index { 'image_name.json': Path(full_path) }
    # recurse=True equivalent in rglob
    src_json_map = {}
    src_jsons = list(src_root.rglob("*.json"))
    
    for p in src_jsons:
        # Use filename as key. Warning: Duplicates will be overwritten, but assuming unique image IDs.
        src_json_map[p.name] = p
        
    print(f"Found {len(src_json_map)} unique JSONs in Source.")

    print(f"Syncing to Target: {dst_root} ...")

    # 2. Gather all images in Destination
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
    dst_images = []
    for ext in extensions:
        dst_images.extend(dst_root.rglob(ext))
        dst_images.extend(dst_root.rglob(ext.upper()))
    
    dst_images = sorted(list(set(dst_images)))
    print(f"Found {len(dst_images)} images in Target directory.")

    count_copied = 0
    count_deleted = 0
    count_missing_source = 0

    for dst_img_path in tqdm(dst_images, desc="Syncing"):
        
        # Expected JSON name: image.png -> image.json
        expected_json_name = dst_img_path.with_suffix('.json').name
        
        # Target JSON path (where we want to write)
        dst_json_path = dst_img_path.with_suffix('.json')

        # 3. Look up in Source Index
        if expected_json_name in src_json_map:
            src_json_path = src_json_map[expected_json_name]
            
            # Copy found JSON to Target
            try:
                shutil.copy2(src_json_path, dst_json_path)
                count_copied += 1
            except Exception as e:
                print(f"Error copying {src_json_path}: {e}")
        else:
            # Source JSON not found
            count_missing_source += 1
            
            # 4. If Source missing, DELETE Target JSON if it exists
            if dst_json_path.exists():
                try:
                    os.remove(dst_json_path)
                    count_deleted += 1
                except Exception as e:
                    print(f"Error deleting {dst_json_path}: {e}")

    print("\nSync Complete.")
    print(f"  Copied from Source: {count_copied}")
    print(f"  Deleted from Target (Stale): {count_deleted}")
    print(f"  Missing in Source: {count_missing_source}")

if __name__ == "__main__":
    main()
