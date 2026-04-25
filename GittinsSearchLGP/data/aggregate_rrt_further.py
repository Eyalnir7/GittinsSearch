#!/usr/bin/env python3
"""
Script to further aggregate RRT data by concatenating time arrays.

Takes aggregated_rrt.csv and creates a new file where rows with the same
file_id, agent_number, file_number, plan, and actionNum (ignoring q0 and qf)
have their time arrays concatenated.

Usage:
    python aggregate_rrt_further.py <directory>
    
Example:
    python aggregate_rrt_further.py randomBlocks
"""

import sys
import pandas as pd
import ast
from pathlib import Path
from collections import defaultdict


def parse_array_string(s):
    """Parse string representation of array to list of floats"""
    try:
        return ast.literal_eval(s)
    except:
        return []


def concatenate_time_arrays(group):
    """Concatenate all time arrays in a group"""
    all_times = []
    for time_str in group['time']:
        times = parse_array_string(time_str)
        all_times.extend(times)
    return all_times


def aggregate_rrt_further(directory):
    """Further aggregate RRT data by concatenating time arrays"""
    directory = Path(directory)
    
    input_file = directory / 'aggregated_rrt.csv'
    output_file = directory / 'aggregated_rrt_by_action.csv'
    
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return
    
    print(f"Reading RRT data from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Group by file_id, agent_number, file_number, plan, and actionNum
    grouping_cols = ['file_id', 'agent_number', 'file_number', 'plan', 'actionNum']
    
    # Check if planID exists and add it if it does
    if 'planID' in df.columns:
        grouping_cols.append('planID')
    
    # Check if feas exists
    include_feas = 'feas' in df.columns
    
    print(f"\nGrouping by: {grouping_cols}")
    
    # Group and aggregate
    aggregated_rows = []
    
    for group_key, group_df in df.groupby(grouping_cols):
        # Concatenate all time arrays
        concatenated_times = concatenate_time_arrays(group_df)
        
        # Create row with group keys
        row = {}
        for i, col in enumerate(grouping_cols):
            row[col] = group_key[i] if len(grouping_cols) > 1 else group_key
        
        # Add concatenated time array as string representation
        row['time'] = str(concatenated_times)
        
        # Add count of original rows
        row['num_trajectories'] = len(group_df)
        
        # Optionally include feas if it exists
        if include_feas:
            # Concatenate feas arrays as well
            all_feas = []
            for feas_str in group_df['feas']:
                feas = parse_array_string(feas_str)
                all_feas.extend(feas)
            row['feas'] = str(all_feas)
        
        aggregated_rows.append(row)
    
    # Create new dataframe
    result_df = pd.DataFrame(aggregated_rows)
    
    print(f"\nAggregated data shape: {result_df.shape}")
    print(f"Number of unique action instances: {len(result_df)}")
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"\nAggregated RRT data saved to: {output_file}")
    
    # Print some statistics
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(result_df)}")
    print(f"Reduction factor: {len(df) / len(result_df):.2f}x")
    
    if 'num_trajectories' in result_df.columns:
        print(f"\nTrajectories per action:")
        print(f"  Mean: {result_df['num_trajectories'].mean():.2f}")
        print(f"  Min: {result_df['num_trajectories'].min()}")
        print(f"  Max: {result_df['num_trajectories'].max()}")
    
    # Show example
    print("\n" + "="*50)
    print("Example rows from output:")
    print("="*50)
    print(result_df.head(3).to_string())
    
    print("\n" + "="*50)
    print("Usage example:")
    print("="*50)
    print("""
import pandas as pd
import ast

# Load the aggregated data
df = pd.read_csv('aggregated_rrt_by_action.csv')

# Parse time arrays
df['time_array'] = df['time'].apply(ast.literal_eval)

# Get all times for a specific action
action_times = df[df['actionNum'] == 0]['time_array'].iloc[0]
print(f"Number of time samples: {len(action_times)}")
print(f"Mean time: {sum(action_times)/len(action_times):.4f}")
""")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python aggregate_rrt_further.py <directory>")
        print("Example: python aggregate_rrt_further.py randomBlocks")
        sys.exit(1)
    
    directory = sys.argv[1]
    aggregate_rrt_further(directory)
