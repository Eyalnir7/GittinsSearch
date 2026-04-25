#!/usr/bin/env python3
"""
Script to aggregate data files from agent directories into separate CSV files
for each data type (waypoints, rrt, lgp).

The script:
1. Traverses directory/data_raw/agent_XX/{waypoints,rrt,lgp}/ structure
2. Extracts file_id as concatenation of agent_number and file_number
3. Creates separate aggregated files for each data type
4. Outputs to aggregated_waypoints.csv, aggregated_rrt.csv, aggregated_lgp.csv
"""

import os
import re
import csv
import json
import sys
from pathlib import Path

def extract_file_number(filename):
    """Extract file number from filename like 'z.dataWaypoints5' -> 5"""
    # Strip whitespace and match the entire number at the end
    filename = filename.strip()
    match = re.search(r'z\.data\w+?(\d+)$', filename)
    return int(match.group(1)) if match else None

def extract_agent_number(agent_dir_name):
    """Extract agent number from directory name like 'agent_17' -> 17"""
    match = re.search(r'agent_(\d+)$', agent_dir_name)
    return int(match.group(1)) if match else None

def compute_file_id(agent_number, file_number):
    """Compute file_id as concatenation: agent_number * 1000 + file_number
    For example: agent_17, file 0 -> 17000
                 agent_17, file 5 -> 17005
    """
    return agent_number * 1000 + file_number

def process_waypoints_files(directory):
    """Process all waypoints files in the directory/data_raw/agent_XX/waypoints/ structure"""
    directory = Path(directory)
    data_raw_dir = directory / 'data_raw'
    
    if not data_raw_dir.exists():
        print(f"Error: data_raw directory not found at {data_raw_dir}")
        return None
    
    # Find all agent directories
    agent_dirs = sorted([d for d in data_raw_dir.iterdir() if d.is_dir() and d.name.startswith('agent_')])
    
    print(f"\nProcessing Waypoints Files")
    print("=" * 50)
    print(f"Found {len(agent_dirs)} agent directories")
    
    # Prepare output CSV
    output_file = directory / 'aggregated_waypoints.csv'
    
    # Detect column names from the first file
    data_columns = None
    for agent_dir in agent_dirs:
        waypoints_dir = agent_dir / 'waypoints'
        if waypoints_dir.exists():
            waypoints_files = list(waypoints_dir.glob('z.dataWaypoints*'))
            if waypoints_files:
                try:
                    with open(waypoints_files[0], 'r', encoding='utf-8') as f:
                        header_line = f.readline().strip()
                        data_columns = header_line.split('#')
                        break
                except:
                    pass
    
    if data_columns is None:
        print("Error: Could not detect data format from waypoints files")
        return None
    
    print(f"Detected data columns: {data_columns}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write CSV header with detected columns
        csv_header = ['file_id', 'agent_number', 'file_number'] + data_columns
        writer.writerow(csv_header)
        
        total_files = 0
        # Process each agent directory
        for agent_dir in agent_dirs:
            agent_number = extract_agent_number(agent_dir.name)
            if agent_number is None:
                print(f"Warning: Could not extract agent number from {agent_dir.name}")
                continue
            
            waypoints_dir = agent_dir / 'waypoints'
            if not waypoints_dir.exists():
                continue
            
            # Find all waypoints files
            waypoints_files = sorted(waypoints_dir.glob('z.dataWaypoints*'))
            
            for file_path in waypoints_files:
                file_number = extract_file_number(file_path.name)
                if file_number is None:
                    print(f"Warning: Could not extract file number from {file_path.name}")
                    continue
                
                file_id = compute_file_id(agent_number, file_number)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Skip the header line
                    for line in lines[1:]:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Split by '#' separator
                        parts = line.split('#')
                        expected_parts = len(data_columns)
                        if len(parts) == expected_parts:
                            writer.writerow([file_id, agent_number, file_number] + parts)
                        else:
                            print(f"Warning: Line in {file_path.name} has {len(parts)} parts instead of {expected_parts}")
                    
                    total_files += 1
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
                    continue
        
        print(f"Total waypoints files processed: {total_files}")
    
    print(f"Aggregated waypoints data saved to {output_file}")
    return output_file

def process_rrt_files(directory):
    """Process all RRT files in the directory/data_raw/agent_XX/rrt/ structure"""
    directory = Path(directory)
    data_raw_dir = directory / 'data_raw'
    
    if not data_raw_dir.exists():
        print(f"Error: data_raw directory not found at {data_raw_dir}")
        return None
    
    # Find all agent directories
    agent_dirs = sorted([d for d in data_raw_dir.iterdir() if d.is_dir() and d.name.startswith('agent_')])
    
    print(f"\nProcessing RRT Files")
    print("=" * 50)
    print(f"Found {len(agent_dirs)} agent directories")
    
    # Prepare output CSV
    output_file = directory / 'aggregated_rrt.csv'
    
    # Detect column names from the first file
    data_columns = None
    for agent_dir in agent_dirs:
        rrt_dir = agent_dir / 'rrt'
        if rrt_dir.exists():
            rrt_files = list(rrt_dir.glob('z.dataRRT*'))
            if rrt_files:
                try:
                    with open(rrt_files[0], 'r', encoding='utf-8') as f:
                        header_line = f.readline().strip()
                        data_columns = header_line.split('#')
                        break
                except:
                    pass
    
    if data_columns is None:
        print("Error: Could not detect data format from RRT files")
        return None
    
    print(f"Detected data columns: {data_columns}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write CSV header with detected columns
        csv_header = ['file_id', 'agent_number', 'file_number'] + data_columns
        writer.writerow(csv_header)
        
        total_files = 0
        total_rows = 0
        # Process each agent directory
        for agent_dir in agent_dirs:
            agent_number = extract_agent_number(agent_dir.name)
            if agent_number is None:
                print(f"Warning: Could not extract agent number from {agent_dir.name}")
                continue
            
            rrt_dir = agent_dir / 'rrt'
            if not rrt_dir.exists():
                continue
            
            # Find all RRT files
            rrt_files = sorted(rrt_dir.glob('z.dataRRT*'))
            
            for file_path in rrt_files:
                file_number = extract_file_number(file_path.name)
                if file_number is None:
                    print(f"Warning: Could not extract file number from {file_path.name}")
                    continue
                
                file_id = compute_file_id(agent_number, file_number)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Skip the header line
                    for line in lines[1:]:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Split by '#' separator
                        parts = line.split('#')
                        expected_parts = len(data_columns)
                        if len(parts) == expected_parts:
                            writer.writerow([file_id, agent_number, file_number] + parts)
                            total_rows += 1
                        else:
                            print(f"Warning: Line in {file_path.name} has {len(parts)} parts instead of {expected_parts}")
                    
                    total_files += 1
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
                    continue
        
        print(f"Total RRT files processed: {total_files}")
        print(f"Total RRT rows: {total_rows}")
    
    print(f"Aggregated RRT data saved to {output_file}")
    return output_file

def process_lgp_files(directory):
    """Process all LGP files in the directory/data_raw/agent_XX/lgp/ structure"""
    directory = Path(directory)
    data_raw_dir = directory / 'data_raw'
    
    if not data_raw_dir.exists():
        print(f"Error: data_raw directory not found at {data_raw_dir}")
        return None
    
    # Find all agent directories
    agent_dirs = sorted([d for d in data_raw_dir.iterdir() if d.is_dir() and d.name.startswith('agent_')])
    
    print(f"\nProcessing LGP Files")
    print("=" * 50)
    print(f"Found {len(agent_dirs)} agent directories")
    
    # Prepare output CSV
    output_file = directory / 'aggregated_lgp.csv'
    
    # Detect column names from the first file
    data_columns = None
    for agent_dir in agent_dirs:
        lgp_dir = agent_dir / 'lgp'
        if lgp_dir.exists():
            lgp_files = list(lgp_dir.glob('z.dataLGP*'))
            if lgp_files:
                try:
                    with open(lgp_files[0], 'r', encoding='utf-8') as f:
                        header_line = f.readline().strip()
                        data_columns = header_line.split('#')
                        break
                except:
                    pass
    
    if data_columns is None:
        print("Error: Could not detect data format from LGP files")
        return None
    
    print(f"Detected data columns: {data_columns}")
    
    # Find the index where last field starts (should be RRTPath or similar multi-line field)
    last_field_index = len(data_columns) - 1
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write CSV header with detected columns
        csv_header = ['file_id', 'agent_number', 'file_number'] + data_columns
        writer.writerow(csv_header)
        
        total_files = 0
        total_rows = 0
        # Process each agent directory
        for agent_dir in agent_dirs:
            agent_number = extract_agent_number(agent_dir.name)
            if agent_number is None:
                print(f"Warning: Could not extract agent number from {agent_dir.name}")
                continue
            
            lgp_dir = agent_dir / 'lgp'
            if not lgp_dir.exists():
                continue
            
            # Find all LGP files
            lgp_files = sorted(lgp_dir.glob('z.dataLGP*'))
            
            for file_path in lgp_files:
                file_number = extract_file_number(file_path.name)
                if file_number is None:
                    print(f"Warning: Could not extract file number from {file_path.name}")
                    continue
                
                file_id = compute_file_id(agent_number, file_number)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split into lines
                    lines = content.split('\n')
                    
                    # Skip the header line
                    i = 1
                    while i < len(lines):
                        line = lines[i].strip()
                        
                        # Skip empty lines
                        if not line:
                            i += 1
                            continue
                        
                        # Each datapoint starts with a line containing '#' separators
                        # and may span multiple lines if the last field is multi-line
                        parts = line.split('#')
                        
                        expected_parts = len(data_columns)-1 # minus one because I forgot to delete a column in the header in the data extraction.
                        if len(parts) >= expected_parts:
                            # We have the start of a datapoint
                            # Extract all fields except the last one
                            
                            # The last field starts after the (expected_parts - 1)th '#'
                            # Find the position of that hash
                            # hash_pos = 0
                            # hash_count = 0
                            # for idx, char in enumerate(line):
                            #     if char == '#':
                            #         hash_count += 1
                            #         if hash_count == expected_parts - 1:
                            #             hash_pos = idx
                            #             break
                            
                            # # Get the last field starting from after that '#'
                            # last_field = line[hash_pos + 1:].strip()
                            
                            # Check if last field is complete (ends with ']')
                            # If not, continue reading lines until we find the closing ']'
                            
                            # Write the complete datapoint
                            writer.writerow([file_id, agent_number, file_number] + parts)
                            total_rows += 1
                        else:
                            print(f"Warning: Line {i} in {file_path.name} has {len(parts)} parts instead of {expected_parts}")
                        
                        i += 1
                    
                    total_files += 1
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"Total LGP files processed: {total_files}")
        print(f"Total LGP rows: {total_rows}")
    
    print(f"Aggregated LGP data saved to {output_file}")
    return output_file

def main(directory='.'):
    """Main function"""
    print("Data File Aggregation Script")
    print("=" * 50)
    
    # Get directory from argument
    directory = Path(directory).resolve()
    print(f"Processing directory: {directory}")
    
    # Process each type of data file
    waypoints_file = process_waypoints_files(directory)
    rrt_file = process_rrt_files(directory)
    lgp_file = process_lgp_files(directory)
    
    # Show summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    if waypoints_file:
        try:
            with open(waypoints_file, 'r') as f:
                line_count = sum(1 for line in f) - 1  # Subtract header
            print(f"Waypoints data rows: {line_count}")
        except Exception as e:
            print(f"Error reading waypoints file: {e}")
    
    if rrt_file:
        try:
            with open(rrt_file, 'r') as f:
                line_count = sum(1 for line in f) - 1  # Subtract header
            print(f"RRT data rows: {line_count}")
        except Exception as e:
            print(f"Error reading RRT file: {e}")
    
    if lgp_file:
        try:
            with open(lgp_file, 'r') as f:
                line_count = sum(1 for line in f) - 1  # Subtract header
            print(f"LGP data rows: {line_count}")
        except Exception as e:
            print(f"Error reading LGP file: {e}")
    
    # Show usage example
    print("\nUsage example:")
    print("```python")
    print("import pandas as pd")
    print("")
    print("# Load waypoints data")
    print("df_waypoints = pd.read_csv('aggregated_waypoints.csv')")
    print("# Note: Column names are read from the file headers")
    print("# Array fields (like feas, time) can be parsed with: import ast; ast.literal_eval(df_waypoints['feas'][0])")
    print("")
    print("# Load RRT data")
    print("df_rrt = pd.read_csv('aggregated_rrt.csv')")
    print("")
    print("# Load LGP data")
    print("df_lgp = pd.read_csv('aggregated_lgp.csv')")
    print("")
    print("# Filter by agent or file_id")
    print("agent_17_data = df_rrt[df_rrt['agent_number'] == 17]")
    print("specific_config = df_rrt[df_rrt['file_id'] == 17005]  # agent_17, file 5")
    print("```")

if __name__ == "__main__":
    # Accept the folder path as an argument
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = '.'
    
    main(directory)