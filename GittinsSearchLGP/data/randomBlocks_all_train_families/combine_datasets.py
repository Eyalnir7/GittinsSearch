"""
Combines the CSV files from randomBlocks_2blocks_2goals_2blockedgoals and
randomBlocks_3blocks_3goals_2blockedgoals into a single output folder.

CSV files are concatenated; aggregated_configurations.json is copied from
the first source folder (they share the same scene configs).
"""
import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_FOLDERS = [
    "randomBlocks_2blocks_2goals_2blockedgoals",
    "randomBlocks_3blocks_3goals_2blockedgoals",
]

OUTPUT_FOLDER = "randomBlocks_2blocks_3blocks_combined"

CSV_FILES = [
    "aggregated_lgp_by_plan.csv",
    "aggregated_rrt_by_action.csv",
    "aggregated_waypoints.csv",
]

CONFIG_FILE = "aggregated_configurations.json"


def main():
    output_dir = os.path.join(BASE_DIR, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder: {output_dir}")

    # --- Combine CSVs ---
    for csv_file in CSV_FILES:
        dfs = []
        for folder in SOURCE_FOLDERS:
            src = os.path.join(BASE_DIR, folder, csv_file)
            print(f"  Reading {src} ...")
            df = pd.read_csv(src)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(output_dir, csv_file)
        combined.to_csv(out_path, index=False)
        print(f"  Written {len(combined)} rows -> {out_path}")

    # --- Copy configurations from first source folder ---
    config_src = os.path.join(BASE_DIR, SOURCE_FOLDERS[0], CONFIG_FILE)
    config_dst = os.path.join(output_dir, CONFIG_FILE)
    shutil.copy2(config_src, config_dst)
    print(f"Copied {CONFIG_FILE} from '{SOURCE_FOLDERS[0]}' -> {config_dst}")

    print("Done.")


if __name__ == "__main__":
    main()
