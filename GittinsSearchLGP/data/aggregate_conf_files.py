#!/usr/bin/env python3
"""
Script to aggregate configuration files from z.conf{file_id}.g files into a single JSON file
using the existing ConfigurationParsing class from DataParsing.py.

The script:
1. Finds all z.conf*.g files in the current directory
2. Uses ConfigurationParsing to parse each file into a dictionary
3. Adds file_id as an additional attribute
4. Saves all configurations as a single JSON file
"""

import os
import re
import json
from pathlib import Path
import sys
from DataParsing import ConfigurationParsing

def extract_file_number(filename):
    """Extract file number from filename like 'z.conf123.g' -> 123"""
    match = re.search(r'z\.conf(\d+)\.g$', filename)
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

def aggregate_configuration_files(directory='.'):
    """Process all z.conf*.g files in the directory/data_raw/agent_XX/configs/ structure"""
    directory = Path(directory)
    data_raw_dir = directory / 'data_raw'
    
    if not data_raw_dir.exists():
        print(f"Error: data_raw directory not found at {data_raw_dir}")
        return {}
    
    # Dictionary to store all configurations
    all_configurations = {}
    
    # Find all agent directories
    agent_dirs = sorted([d for d in data_raw_dir.iterdir() if d.is_dir() and d.name.startswith('agent_')])
    
    print(f"Found {len(agent_dirs)} agent directories")
    
    total_files = 0
    # Process each agent directory
    for agent_dir in agent_dirs:
        agent_number = extract_agent_number(agent_dir.name)
        if agent_number is None:
            print(f"Warning: Could not extract agent number from {agent_dir.name}")
            continue
        
        # Look for configs subdirectory
        configs_dir = agent_dir / 'configs'
        if not configs_dir.exists():
            print(f"Warning: configs directory not found in {agent_dir.name}")
            continue
        
        # Find all z.conf*.g files in this agent's configs directory
        conf_files = list(configs_dir.glob('z.conf*.g'))
        
        # Sort files by file number
        def sort_key(file_path):
            file_num = extract_file_number(file_path.name)
            return file_num if file_num is not None else 0
        
        conf_files.sort(key=sort_key)
        
        print(f"Processing {agent_dir.name}: {len(conf_files)} configuration files")
        
        # Process each configuration file
        for file_path in conf_files:
            file_number = extract_file_number(file_path.name)
            if file_number is None:
                print(f"Warning: Could not extract file number from {file_path.name}")
                continue
            
            # Compute file_id as concatenation of agent_number and file_number
            file_id = compute_file_id(agent_number, file_number)
            
            try:
                # Parse the configuration file using the existing parser
                config_dict = ConfigurationParsing.parse_conf_file(str(file_path))
                
                # Add file_id, agent_number, and file_number to the configuration
                config_with_id = {
                    'file_id': file_id,
                    'agent_number': agent_number,
                    'file_number': file_number,
                    'scene_config': config_dict
                }
                
                # Store in the main dictionary
                all_configurations[file_id] = config_with_id
                
                total_files += 1
                
            except Exception as e:
                print(f"Error processing {file_path.name} in {agent_dir.name}: {e}")
                continue
    
    print(f"\nTotal configurations processed: {total_files}")
    return all_configurations

def save_configurations(configurations, output_file='aggregated_configurations.json'):
    """Save configurations to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(configurations, f, indent=2, ensure_ascii=False)
        
        print(f"Aggregated configurations saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error saving configurations to {output_file}: {e}")
        return False

def load_configurations(input_file='aggregated_configurations.json'):
    """Load configurations from JSON file"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            configurations = json.load(f)
        
        print(f"Loaded {len(configurations)} configurations from {input_file}")
        return configurations
        
    except Exception as e:
        print(f"Error loading configurations from {input_file}: {e}")
        return {}

def main(directory='.'):
    """Main function"""
    print("Configuration File Aggregation Script")
    print("=" * 45)
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"Working directory: {current_dir}")
    # make current_dir the directory passed as argument. The directory argument can be relative or absolute
    current_dir = Path(directory).resolve()
    print(f"Processing directory: {current_dir}")
    
    # Process configuration files
    configurations = aggregate_configuration_files(current_dir)
    
    if not configurations:
        print("No configuration files were processed successfully!")
        return
    
    # Save to JSON
    output_file = directory + '/aggregated_configurations.json'
    success = save_configurations(configurations, output_file)
    
    if success:
        # Show some statistics
        print(f"\nSummary:")
        print(f"Total configurations: {len(configurations)}")
        
        # Show file_id range and agent info
        file_ids = sorted(configurations.keys())
        print(f"File ID range: {min(file_ids)} to {max(file_ids)}")
        
        # Show agent number range
        agent_numbers = sorted(set([configurations[fid]['agent_number'] for fid in file_ids]))
        print(f"Agent numbers: {agent_numbers}")
        
        # Show sample objects from first configuration
        if file_ids:
            first_config = configurations[file_ids[0]]['scene_config']
            print(f"Sample objects in config {file_ids[0]}: {list(first_config.keys())[:5]}...")
        
        # Show usage example
        print(f"\nUsage example:")
        print(f"```python")
        print(f"import json")
        print(f"")
        print(f"# Load all configurations")
        print(f"with open('{output_file}', 'r') as f:")
        print(f"    configs = json.load(f)")
        print(f"")
        print(f"# Access specific configuration (e.g., agent_17, file 5 -> file_id 17005)")
        print(f"config = configs['17005']['scene_config']")
        print(f"agent_num = configs['17005']['agent_number']  # 17")
        print(f"file_num = configs['17005']['file_number']    # 5")
        print(f"```")

def test_loading():
    """Test function to verify loading works correctly"""
    print("\nTesting configuration loading...")
    
    configurations = load_configurations('aggregated_configurations.json')
    
    if configurations:
        # Test accessing a specific configuration
        test_id = list(configurations.keys())[0]
        test_config = configurations[test_id]['scene_config']
        
        print(f"Test: Loaded config {test_id}")
        print(f"Objects in config: {len(test_config)}")
        
        # Try to find a goal object to test the structure
        for obj_name, obj_data in test_config.items():
            if obj_data.get('logical', {}).get('is_goal'):
                print(f"Found goal object '{obj_name}' with is_place: {obj_data['logical'].get('is_place', False)}")
                break

if __name__ == "__main__":
    # accept the folder path as an argument
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = '.'
    
    main(directory)
    
    # Test loading the created file
    if Path('aggregated_configurations.json').exists():
        test_loading()
