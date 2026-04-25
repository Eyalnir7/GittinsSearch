import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import Dict, List, Tuple, Iterable, Optional, List
import os
import ast
import json

class DataFrameParsing:

    def parse_blocks(text: str) -> List[Dict]:
        """Parse blocks of graph data with footer metadata."""
        footer_re = re.compile(r'#(\[.+?\])#(\[.+?\])#(.+)#(\d+)$')
        lines = text.splitlines()
        header_seen = False
        buf: List[str] = []
        rows: List[Dict] = []
        
        for raw in lines:
            line = raw.rstrip("\n")
            
            if not header_seen:
                # Check for both old and new header formats
                if "#" in line and ("feasible" in line or "feas" in line) and "time" in line and "plan" in line.lower():
                    header_seen = True
                    # Determine if this is the new format (has "feas" instead of "feasible")
                    is_new_format = "feas#time#plan#planID" in line
                continue

            # Check if this line is a footer (ends with metadata pattern)
            m = footer_re.search(line)
            if m:
                try:
                    # Parse arrays from string representation
                    feas_array = ast.literal_eval(m.group(1))
                    time_array = ast.literal_eval(m.group(2))
                    plan = m.group(3)
                    plan_id = int(m.group(4))
                    
                    # Graph text: everything accumulated + the current line up to the first '#'
                    head = line.split("#", 1)[0]
                    graph_text = "\n".join(buf + [head]).strip()
                    
                    # Create single row with arrays as values
                    rows.append({
                        "feasible": feas_array,
                        "time": time_array,
                        "plan": plan,
                        "planID": plan_id
                    })
                    
                    buf = []  # reset for next block
                    continue
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Failed to parse footer line: {line[:100]}... Error: {e}")
                    pass
            
            # If we get here, this line is not a footer, add to buffer
            buf.append(line)
        
        return rows

    def read_dataset(file_path: str = "z.data") -> pd.DataFrame:
        """Read and parse a single dataset file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        rows = DataFrameParsing.parse_blocks(text)
        df = pd.DataFrame(rows)
        return df

    def load_all_datasets(start_file, end_file) -> pd.DataFrame:
        """Load and concatenate all datasets from z.data0 to z.data9."""
        all_dfs = []

        for i in range(start_file, end_file + 1):
            file_path = f"../data_blockObj/z.data{i}"
            try:
                df_i = DataFrameParsing.read_dataset(file_path)
                df_i['file_id'] = i  # Add file ID column
                all_dfs.append(df_i)
                print(f"  Loaded {len(df_i)} records from {file_path}")
            except FileNotFoundError:
                print(f"  Warning: {file_path} not found, skipping...")
                # print the trace of the exception

                traceback.print_exc()
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                traceback.print_exc()
        
        if all_dfs:
            # Concatenate all DataFrames
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"\nTotal combined records: {len(combined_df)}")
            print(f"Files loaded: {len(all_dfs)}/{end_file - start_file + 1}")
            return combined_df
        else:
            print("No datasets were loaded successfully!")
            traceback.print_exc()
            return pd.DataFrame()
        

class ConfigurationParsing:

    def get_relevant_key(scene_dict, search_key):
        for key in scene_dict.keys():
            if search_key in key:
                return key
        return None

    def get_parent(object_key):
        match = re.search(r'\((.*?)\)', object_key)
        if match:
            return match.group(1)
        return None

    def get_absolute_position_aux(scene_dict, object_name, total_position):
        relevant_key = ConfigurationParsing.get_relevant_key(scene_dict, object_name)

        # Safely get the current pose, with a default if missing
        current_pose = np.array(scene_dict.get(relevant_key, {}).get('pose', [0.0, 0.0, 0.0]))

        if '(' not in relevant_key:
            return list(total_position + current_pose)

        # Extract the string between parentheses (the parent object)
        inner = ConfigurationParsing.get_parent(relevant_key)

        return ConfigurationParsing.get_absolute_position_aux(scene_dict, inner, total_position + current_pose)

    def get_absolute_position(scene_dict, object_name):
        return ConfigurationParsing.get_absolute_position_aux(scene_dict, object_name, np.array([0.0, 0.0, 0.0]))

    def parse_object_line(line: str):
        """
        Parse a single line of the form:
        { key: value, key2: value2, ... }
        into a Python dictionary.
        """
        # Clean up whitespace and outer braces
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            line = line[1:-1].strip()

        # Replace True/False with lowercase (for JSON compatibility)
        line = line.replace('True', 'true').replace('False', 'false')

        def parse_value(v):
            v = v.strip()
            # Handle lists
            if v.startswith('[') and v.endswith(']'):
                return json.loads(v)
            # Handle nested dicts
            elif v.startswith('{') and v.endswith('}'):
                return ConfigurationParsing.parse_object_line(v)
            # Handle booleans
            elif v.lower() in ['true', 'false']:
                return v.lower() == 'true'
            # Handle numbers
            elif re.match(r'^-?\d+(\.\d+)?$', v):
                return float(v) if '.' in v else int(v)
            # Otherwise, treat as string
            else:
                return v

        # Split into key-value pairs, ignoring commas inside [] or {}
        parts = re.split(r',\s*(?![^{}\[\]]*[\]\}])', line)
        result = {}

        for part in parts:
            if not part.strip():
                continue
            key, value = map(str.strip, part.split(':', 1))
            result[key] = parse_value(value)

        return result

    def parse_conf_file(file_path: str) -> Dict:
        """
        Parse a z.conf file into a nested dictionary structure.
        
        Args:
            file_path: Path to the z.conf file
            
        Returns:
            Dictionary representing the scene configuration
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        scene_dict = {}
        for line in content.splitlines():
            line = line.strip()
            
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                scene_dict[key] = ConfigurationParsing.parse_object_line(value)

        scene_dict.pop('camera_init', None)

        for key in scene_dict.keys():
            abs_pos = ConfigurationParsing.get_absolute_position(scene_dict, key)
            scene_dict[key]['absolute_pose'] = abs_pos

        # if there are parenthesis in the key, change the name of the key to be without parenthesis (and what's inside)
        for key in list(scene_dict.keys()):
            if '(' in key:
                new_key = key.split('(')[0].strip()
                scene_dict[new_key] = scene_dict.pop(key)

        return scene_dict
    
if __name__ == "__main__":
    print("current working directory:", os.getcwd())
    conf = ConfigurationParsing.parse_conf_file("GittinsSearchLGP/FolTest/data_blockObj/z.conf4.g")
    # print(conf['goalD']['logical']['is_place'])
    # print the entries that don't have a shape
    for key, value in conf.items():
        if 'shape' not in value:
            print(f"{key}: {value}")
    # df = DataFrameParsing.load_all_datasets(4, 56)
    # print(df.head())