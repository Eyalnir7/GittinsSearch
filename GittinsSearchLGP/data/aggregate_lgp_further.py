#!/usr/bin/env python3
"""
Script to further aggregate LGP data by concatenating feas and time arrays.

Takes aggregated_waypoints.csv and creates a new file where rows with the same
file_id, agent_number, file_number, plan, and planID (ignoring RRTPath)
have their feas and time arrays concatenated.

Usage:
    python aggregate_lgp_further.py <directory>
    
Example:
    python aggregate_lgp_further.py randomBlocks
"""

import sys
import pandas as pd
import ast
from pathlib import Path
from collections import defaultdict


def parse_array_string(s):
    """Parse string representation of array to list"""
    try:
        return ast.literal_eval(s)
    except:
        return []


def collect_values(group, column_name):
    """Collect all values from a specific column into a list"""
    return group[column_name].tolist()


def aggregate_lgp_further(directory):
    """Further aggregate LGP data by concatenating feas and time arrays"""
    directory = Path(directory)
    
    input_file = directory / 'aggregated_lgp.csv'
    output_file = directory / 'aggregated_lgp_by_plan.csv'
    
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return
    
    print(f"Reading LGP data from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Group by file_id, agent_number, file_number, plan, and planID
    grouping_cols = ['file_id', 'agent_number', 'file_number', 'plan', 'planID']
    
    # Verify all grouping columns exist
    missing_cols = [col for col in grouping_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        grouping_cols = [col for col in grouping_cols if col in df.columns]
    
    print(f"\nGrouping by: {grouping_cols}")
    
    # Group and aggregate
    aggregated_rows = []
    
    for group_key, group_df in df.groupby(grouping_cols):
        # Create row with group keys
        row = {}
        for i, col in enumerate(grouping_cols):
            row[col] = group_key[i] if len(grouping_cols) > 1 else group_key
        
        # Collect feas values into array
        if 'feas' in group_df.columns:
            collected_feas = collect_values(group_df, 'feas')
            row['feas'] = str(collected_feas)
        
        # Collect time values into array
        if 'time' in group_df.columns:
            collected_times = collect_values(group_df, 'time')
            row['time'] = str(collected_times)
        
        # Add count of original rows
        row['num_rrt_paths'] = len(group_df)
        
        aggregated_rows.append(row)
    
    # Create new dataframe
    result_df = pd.DataFrame(aggregated_rows)
    
    print(f"\nAggregated data shape: {result_df.shape}")
    print(f"Number of unique plan instances: {len(result_df)}")
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"\nAggregated LGP data saved to: {output_file}")
    
    # Print some statistics
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(result_df)}")
    print(f"Reduction factor: {len(df) / len(result_df):.2f}x")
    
    if 'num_rrt_paths' in result_df.columns:
        print(f"\nRRT paths per plan:")
        print(f"  Mean: {result_df['num_rrt_paths'].mean():.2f}")
        print(f"  Min: {result_df['num_rrt_paths'].min()}")
        print(f"  Max: {result_df['num_rrt_paths'].max()}")
    
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
df = pd.read_csv('aggregated_lgp_by_plan.csv')

# Parse arrays
df['feas_array'] = df['feas'].apply(ast.literal_eval)
df['time_array'] = df['time'].apply(ast.literal_eval)

# Get all feas and times for a specific plan
plan_feas = df[df['planID'] == 9]['feas_array'].iloc[0]
plan_times = df[df['planID'] == 9]['time_array'].iloc[0]

print(f"Number of RRT paths: {len(plan_feas)}")
print(f"Feasibility rate: {sum(plan_feas)/len(plan_feas):.2%}")
print(f"Mean time: {sum(plan_times)/len(plan_times):.4f}")
""")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python aggregate_lgp_further.py <directory>")
        print("Example: python aggregate_lgp_further.py randomBlocks")
        sys.exit(1)
    
    directory = sys.argv[1]
    aggregate_lgp_further(directory)
