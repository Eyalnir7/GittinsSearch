#!/usr/bin/env python3
"""
Script to convert CSV files (aggregated_lgp.csv, aggregated_waypoints.csv, aggregated_rrt.csv)
into JSON files with transition probabilities and times.
"""

import pandas as pd
import numpy as np
import ast
import json
from scipy.stats import beta, norm
from pathlib import Path
import re


def get_chain_probs(feas_array, time_array, ci_level=0.95):
    """
    Calculate transition probabilities and times from feasibility and time arrays.
    
    Args:
        feas_array: Array of feasibility values (1 for success, 0 for failure)
        time_array: Array of time values corresponding to each feasibility result
        ci_level: Confidence interval level (default 0.95, not used in return but computed internally)
    
    Returns:
        done_transitions: Array of success transition probabilities
        done_times: Array of times for successful transitions
        fail_transitions: Array of failure transition probabilities
        fail_times: Array of times for failed transitions
    """
    feas_array = np.array(feas_array)
    time_array = np.array(time_array)
    feas_indices = feas_array == 1
    feas_times = time_array[feas_indices]
    infeas_times = time_array[~feas_indices]
    feas_values, feas_counts = np.unique(feas_times, return_counts=True)
    infeas_values, infeas_counts = np.unique(infeas_times, return_counts=True)
    
    # get all values
    all_values = np.union1d(feas_values, infeas_values)
    
    # make feas_counts sorted according to all_values, filling missing values with 0
    feas_counts_full = np.zeros_like(all_values)
    infeas_counts_full = np.zeros_like(all_values)
    for i, v in enumerate(all_values):
        if v in feas_values:
            feas_counts_full[i] = feas_counts[feas_values == v][0]
        if v in infeas_values:
            infeas_counts_full[i] = infeas_counts[infeas_values == v][0]
    
    all_counts = feas_counts_full + infeas_counts_full
    cum_sum_counts = np.cumsum(all_counts[::-1])[::-1]
    feas_array_probs = feas_counts_full / cum_sum_counts
    infeas_array_probs = infeas_counts_full / cum_sum_counts
    
    # remove the times and transition probabilities where feas_array_probs is 0 and infeas_array_probs is 0
    non_zero_feas_indices = feas_array_probs > 0
    non_zero_infeas_indices = infeas_array_probs > 0
    feas_array_probs = feas_array_probs[non_zero_feas_indices]
    infeas_array_probs = infeas_array_probs[non_zero_infeas_indices]
    feas_times = all_values[non_zero_feas_indices]
    infeas_times = all_values[non_zero_infeas_indices]

    # concat the row at the beginning of the arrays
    feas_array_probs = np.insert(feas_array_probs, 0, 0)
    infeas_array_probs = np.insert(infeas_array_probs, 0, 0)
    feas_times = np.insert(feas_times, 0, 0)
    infeas_times = np.insert(infeas_times, 0, 0)

    return feas_array_probs, feas_times, infeas_array_probs, infeas_times


def process_lgp_csv(csv_path, output_path):
    """
    Process LGP CSV file and create JSON with plan ID as key.
    
    Args:
        csv_path: Path to aggregated_lgp.csv
        output_path: Path to output JSON file
    """
    print(f"Processing LGP CSV: {csv_path}")
    
    # Read CSV - no converters needed, data will be aggregated
    df = pd.read_csv(csv_path, converters={
        'feas': ast.literal_eval,
        'time': ast.literal_eval
    })
    
    result = {}
    
    for idx, row in df.iterrows():
        plan_id = str(row['planID'])
        
        # Convert to appropriate types
        feas_list = [int(x) for x in row['feas']]
        time_list = [int(np.ceil(t / 0.01)) for t in row['time']]
        
        # Get transition probabilities and times
        done_transitions, done_times, fail_transitions, fail_times = get_chain_probs(
            feas_list, time_list
        )
        
        result[plan_id] = {
            'done_transitions': done_transitions.tolist(),
            'done_times': done_times.tolist(),
            'fail_transitions': fail_transitions.tolist(),
            'fail_times': fail_times.tolist(),
            'plan': str(row['plan'])
        }
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  Wrote {len(result)} entries to {output_path}")


def process_waypoints_csv(csv_path, output_path):
    """
    Process waypoints CSV file and create JSON with plan ID as key.
    
    Args:
        csv_path: Path to aggregated_waypoints.csv
        output_path: Path to output JSON file
    """
    print(f"Processing Waypoints CSV: {csv_path}")
    
    # Read CSV with custom converters for list columns (plan as string, not evaluated)
    df = pd.read_csv(csv_path, converters={
        'feas': ast.literal_eval,
        'time': ast.literal_eval
    })
    
    
    # Convert to appropriate types
    df['feas'] = df['feas'].apply(lambda x: [int(i) for i in x])
    df['time'] = df['time'].apply(lambda x: [float(i) for i in x])
    
    # Parse plan column - it's a string representation of a list
    df['plan'] = df['plan'].apply(lambda x: str(x))
    
    for i in range(len(df)):
        df.at[i, 'time'] = [int(np.ceil(t / 0.01)) for t in df.at[i, 'time']]
        
    result = {}
    
    for idx, row in df.iterrows():
        plan_id = str(row['planID'])
        
        # Get transition probabilities and times
        done_transitions, done_times, fail_transitions, fail_times = get_chain_probs(
            row['feas'], row['time']
        )
        
        result[plan_id] = {
            'done_transitions': done_transitions.tolist(),
            'done_times': done_times.tolist(),
            'fail_transitions': fail_transitions.tolist(),
            'fail_times': fail_times.tolist(),
            'plan': row['plan']
        }
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  Wrote {len(result)} entries to {output_path}")


def parse_list_of_lists(text):
    """
    Converts a string like:
      [[pick_touch, objectA, floor, ego], [place_straightOn_goal, objectA, ego, goalA]]
    into a real Python list of lists of strings.
    """
    # Add quotes around unquoted tokens (words, underscores, numbers)
    quoted = re.sub(r'([A-Za-z0-9_]+)', r'"\1"', text)

    # Now safely evaluate it as a Python literal
    return ast.literal_eval(quoted)

def process_rrt_csv(csv_path, output_path):
    """
    Process RRT CSV file and create JSON with {planId}_action_{actionNum} as key.
    
    Args:
        csv_path: Path to aggregated_rrt.csv
        output_path: Path to output JSON file
    """
    print(f"Processing RRT CSV: {csv_path}")
    
    # Read CSV with custom converters for list columns (plan as string, not evaluated)
    df = pd.read_csv(csv_path, converters={
        'feas': ast.literal_eval,
        'time': ast.literal_eval,
        'q0': ast.literal_eval,
        'qf': ast.literal_eval
    })
    
    # Convert to appropriate types
    df['feas'] = df['feas'].apply(lambda x: [int(i) for i in x])
    df['time'] = df['time'].apply(lambda x: [float(i) for i in x])
    for i in range(len(df)):
        df.at[i, 'time'] = [int(np.ceil(t / 0.01)) for t in df.at[i, 'time']]
    
    # Parse plan column - it's a string representation of a list
    df['plan'] = df['plan'].apply(lambda x: str(x))
    
    result = {}
    
    for idx, row in df.iterrows():
        # Create key as {planId}_action_{actionNum}
        key = f"{row['planID']}_action_{row['actionNum']}"
        
        # Get transition probabilities and times
        done_transitions, done_times, fail_transitions, fail_times = get_chain_probs(
            row['feas'], row['time']
        )
        
        result[key] = {
            'done_transitions': done_transitions.tolist(),
            'done_times': done_times.tolist(),
            'fail_transitions': fail_transitions.tolist(),
            'fail_times': fail_times.tolist(),
            'plan': row['plan'],
            'planLength': len(parse_list_of_lists(row['plan']))
        }
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  Wrote {len(result)} entries to {output_path}")


def main():
    """Main function to process all CSV files."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Define input and output paths
    lgp_csv = script_dir / "aggregated_lgp_by_plan.csv"
    waypoints_csv = script_dir / "aggregated_waypoints.csv"
    rrt_csv = script_dir / "aggregated_rrt_by_action.csv"
    
    lgp_json = script_dir / "lgp_chains.json"
    waypoints_json = script_dir / "waypoints_chains.json"
    rrt_json = script_dir / "rrt_chains.json"
    
    # Process each CSV file
    if lgp_csv.exists():
        process_lgp_csv(lgp_csv, lgp_json)
    else:
        print(f"Warning: {lgp_csv} not found")
    
    if waypoints_csv.exists():
        process_waypoints_csv(waypoints_csv, waypoints_json)
    else:
        print(f"Warning: {waypoints_csv} not found")
    
    if rrt_csv.exists():
        process_rrt_csv(rrt_csv, rrt_json)
    else:
        print(f"Warning: {rrt_csv} not found")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
