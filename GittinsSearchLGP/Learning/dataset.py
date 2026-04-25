"""
Dataset creation and preprocessing for heterogeneous graph constraint prediction.
"""
import torch
from torch_geometric.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import ast
import re
import os
from enums import NodeType, TaskType

# ============================================================================
# WANDB Configuration - Customize these for your WANDB workspace
# ============================================================================
WANDB_PROJECT = "randomBlocksData"          # WANDB project name
WANDB_ENTITY = "your_entity"                # WANDB entity/team name (change this)
WANDB_ARTIFACT_BASE_NAME = "randomBlocks_all"  # Base name for artifacts
# Example: artifacts will be named "{WANDB_ARTIFACT_BASE_NAME}-train", 
#          "{WANDB_ARTIFACT_BASE_NAME}-val", "{WANDB_ARTIFACT_BASE_NAME}-test"
# ============================================================================

def file_by_node_type(node_type):
    """Map node type to corresponding data file name"""
    if node_type == NodeType.WAYPOINTS:
        return "aggregated_waypoints.csv"
    elif node_type == NodeType.RRT:
        return "aggregated_rrt_by_action.csv"
    elif node_type == NodeType.LGP:
        return "aggregated_lgp_by_plan.csv"
    else:
        raise ValueError(f"Unknown node type: {node_type}")


def parse_malformed_plan(plan_str):
    """
    Parse plan strings that have unquoted identifiers like:
    [[pick_touch, objectA, floor, ego], [place_straightOn_goal, objectA, ego, goalC]]
    
    Convert them to proper Python literals with quoted strings before parsing.
    """
    # Add quotes around identifiers (words not already quoted)
    # Match word characters that are not inside quotes and not numbers
    def quote_identifier(match):
        word = match.group(0)
        # Don't quote if it's already a number
        try:
            float(word)
            return word
        except ValueError:
            return f'"{word}"'
    
    # Pattern to match unquoted words (identifiers)
    # This matches word characters that aren't preceded by a quote
    quoted_str = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', quote_identifier, plan_str)
    
    try:
        return ast.literal_eval(quoted_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing plan: {plan_str[:100]}...")
        print(f"After conversion: {quoted_str[:100]}...")
        raise e


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


def sample_by_prefix(dataframe, file_id_col='file_id', p=1.0, seed=42):
    """Sample approximately p fraction from each prefix group in a stratified manner.

    The prefix (family) is defined as the substring before the first occurrence of "__". 
    If there is no "__" in a file id, the prefix is the empty string ''.
    
    This function performs stratified sampling at the file_id level, ensuring:
      - All rows with the same file_id stay together in the sample
      - The proportion of file_ids from each prefix/family is preserved
      - Random sampling is reproducible via the seed parameter
    
    Args:
        dataframe: DataFrame to sample from
        file_id_col: Column name containing file IDs
        p: Fraction of data to sample (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame with approximately p fraction of file_ids from each prefix group
    """
    if p >= 1.0:
        return dataframe.reset_index(drop=True)

    # Work on a copy and ensure file ids are strings for grouping
    df = dataframe.copy()
    df[file_id_col] = df[file_id_col].astype(str)

    # Extract prefix (substring before '__'), empty string if none
    df['_prefix'] = df[file_id_col].apply(lambda x: x.split('__', 1)[0] if '__' in x else '')

    # Get unique file_ids with their prefix (family)
    file_id_prefix_df = df[[file_id_col, '_prefix']].drop_duplicates(subset=file_id_col)
    
    # Use train_test_split with stratification to sample file_ids
    # We use train_test_split to get the sampled portion
    sampled_file_ids, _ = train_test_split(
        file_id_prefix_df[file_id_col],
        train_size=p,
        stratify=file_id_prefix_df['_prefix'],
        random_state=seed
    )
    
    # Filter dataframe to include only rows with sampled file_ids
    result = df[df[file_id_col].isin(sampled_file_ids)].copy()
    result = result.drop(columns=['_prefix']).reset_index(drop=True)
    
    return result


class HeteroGraphDataset(Dataset):
    def __init__(self, input_path, nodeType, taskType, device=None, task=None, data_percentage=1.0, seed=42):
        """
        Initialize the dataset with the parsed dataframe and directory containing scene configurations
         Precomputes all graph data during initialization for faster retrieval
        
        Args:
            dataframe: DataFrame containing plan and feasibility data
            input_path: Path to aggregated configurations JSON file
            device: Device to place tensors on (cpu, cuda, mps). If None, auto-detects.
            task: Task type (e.g., "optimal_time_prediction_mse" or None for classification)
            data_percentage: Fraction of data to use (0.0 to 1.0)
            seed: Random seed for reproducible sampling when data_percentage < 1.0
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from ToHeteroDatav2 import to_hetero_data, get_hetero_data_input

        dataframe = pd.read_csv(input_path+file_by_node_type(nodeType), converters={'feas': ast.literal_eval, 'time': ast.literal_eval, 'plan': parse_malformed_plan})
        # treat the elements in the feas and times as numbers
        dataframe['feas'] = dataframe['feas'].apply(lambda x: [int(i) for i in x])
        dataframe['time'] = dataframe['time'].apply(lambda x: [float(i) for i in x])
        for i in range(len(dataframe)):
            dataframe.at[i, 'time'] = [int(np.ceil(t / 0.01)) for t in dataframe.at[i, 'time']]
        
        if device is None:
            device = torch.device("cpu")
        self.device = device
        dataframe = dataframe.reset_index(drop=True)
        self.scenes_dict = load_configurations(input_file=input_path+"aggregated_configurations.json")
        # take only a percentage of the data if specified
        if data_percentage < 1.0:
            print(f"Using {data_percentage*100:.1f}% of scenes (stratified per-prefix sampling at file_id level)")
            # Work on string representation of file_id for grouping and matching
            dataframe['file_id'] = dataframe['file_id'].astype(str)

            # Sample per-prefix at file_id level (all samples from same file_id stay together)
            # This handles both combined datasets (prefix present) and single-source datasets
            dataframe = sample_by_prefix(dataframe, file_id_col='file_id', p=data_percentage, seed=seed)

            # --- Diagnostic prints to verify per-prefix sampling ---
            prefix_series = dataframe['file_id'].apply(lambda x: x.split('__', 1)[0] if '__' in x else '')
            grouped = dataframe.groupby(prefix_series)
            print("Samples per prefix:")
            for prefix, group in grouped:
                display_prefix = prefix if prefix != '' else '<no_prefix>'
                first_row = group.iloc[0]
                file_id = first_row.get('file_id', 'N/A')
                plan = first_row.get('plan', 'N/A')
                plan_str = str(plan)[:140].replace('\n', ' ')
                print(f"  - {display_prefix}: {len(group)} samples, first file_id={file_id}, plan={plan_str}")
            print("--- end of per-prefix diagnostic info ---")


        # Precompute all graph data and labels
        self.data = []
        for idx, row in tqdm(dataframe.iterrows(), desc="Preprocessing graph data", total=len(dataframe)):
            file_id = row['file_id']
            scene_dict = self.scenes_dict.get(str(file_id), None).get('scene_config')
            if scene_dict is None:
                print(f"Warning: Configuration for file_id {file_id} not found. Skipping this entry.")
                continue
            if len(row['time']) == 0:
                print(f"Warning: Empty time list for file_id {file_id}. Skipping this entry.")
                continue

            action_number = None
            if "actionNum" in row.keys():
                action_number = row['actionNum']
            # Get the graph data structure
            rel_nodes, pair_edges, sink_edges = get_hetero_data_input(scene_dict, row['plan'], action_number=action_number, device=device)
            graph_data = to_hetero_data(rel_nodes, pair_edges, sink_edges, device=device)

            for node_type in graph_data.node_types:
                if hasattr(graph_data[node_type], 'x'):
                    graph_data[node_type].x.to(device)

            # Get the label - mean of feasibility values (no need to move to device here)
            if taskType == TaskType.FEASIBILITY:
                if len(row['feas']) !=0:
                    graph_data.y = torch.tensor(np.mean(row['feas']), dtype=torch.float, device=device)
                else:
                    graph_data.y = torch.tensor(0.0, dtype=torch.float, device=device)
            elif taskType == TaskType.QUANTILE_REGRESSION_FEAS:
                feas_values = [t for t, f in zip(row['time'], row['feas']) if f == 1]
                if len(feas_values) == 0:
                    continue
                graph_data.y = torch.tensor(feas_values, dtype=torch.float, device=device)
                graph_data.y_len = torch.tensor([len(feas_values)], dtype=torch.long)
            elif taskType == TaskType.QUANTILE_REGRESSION_INFEAS:
                infeas_values = [t for t, f in zip(row['time'], row['feas']) if f == 0]
                if len(infeas_values) == 0:
                    continue
                graph_data.y = torch.tensor(infeas_values, dtype=torch.float, device=device)
                graph_data.y_len = torch.tensor([len(infeas_values)], dtype=torch.long)

            # Store additional metadata
            graph_data.taskPlan = row['plan']
            graph_data.file_id = file_id
            if action_number is not None:
                graph_data.actionNum = action_number
            
            self.data.append(graph_data)
        
        print(f"Dataset initialized with {len(self.data)} samples on device: {device}")
        
    def len(self):
        return len(self.data)
        
    def get(self, idx):
        return self.data[idx]


def load_or_create_datasets(task, nodeType, percentiles=None, scenes_dir="../data/randomBlocks/", use_wandb_datasets=True, return_metadata=False, percent_data=1.0, seed=42, wandb_entity=None, wandb_project=None, wandb_artifact_base_name=None, train_artifact_name=None):
    """Load datasets from WANDB artifact or create and save them if they don't exist
    
    Args:
        scenes_dir: Path to the directory containing data files
        use_wandb_datasets: If True, try to download pre-split CSV files from WANDB and fall back to creating if it fails.
                           If False, skip WANDB and create new datasets (replacing old ones).
        return_metadata: If True, return metadata dictionary as well
        task: Task type (e.g., TaskType.FEASIBILITY)
        percent_data: Fraction of data to use (0.0 to 1.0) - only used when creating datasets from scratch
        seed: Random seed for reproducible sampling
        wandb_entity: WANDB entity/team name (uses WANDB_ENTITY if None)
        wandb_project: WANDB project name (uses WANDB_PROJECT if None)
        wandb_artifact_base_name: Base name for artifacts (uses WANDB_ARTIFACT_BASE_NAME if None)
        train_artifact_name: If provided, use this specific name for train artifact
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata) if return_metadata=True
        Tuple of (train_dataset, val_dataset, test_dataset) otherwise
    """
    import wandb
    
    # Use provided values or fall back to module defaults
    entity = wandb_entity or WANDB_ENTITY
    project = wandb_project or WANDB_PROJECT
    artifact_base_name = wandb_artifact_base_name or WANDB_ARTIFACT_BASE_NAME
    
    device = torch.device("cpu")
    device_name = "cpu"
    
    metadata = {
        "task": task.name if hasattr(task, "name") else str(task),
        "node_type": nodeType.name,
        "device": device_name,
        "source": "unknown",
        "percent_data": percent_data,
        "percentiles": percentiles,
    }
    
    # Try to download pre-split CSV datasets from WANDB
    success = True
    if use_wandb_datasets:
        try:
            print(f"Attempting to download pre-split CSV datasets from WANDB project '{project}'...")
            
            # Download the three split artifacts
            _train_art_name = train_artifact_name if train_artifact_name is not None else f'{artifact_base_name}-train'
            train_artifact = wandb.use_artifact(f'{entity}/{project}/{_train_art_name}:latest', type='dataset')
            val_artifact = wandb.use_artifact(f'{entity}/{project}/{artifact_base_name}-val:latest', type='dataset')
            test_artifact = wandb.use_artifact(f'{entity}/{project}/{artifact_base_name}-test:latest', type='dataset')
            
            train_dir = train_artifact.download()
            val_dir = val_artifact.download()
            test_dir = test_artifact.download()
            
            print(f"Downloaded artifacts from WANDB:")
            print(f"  Train: {train_artifact.version}")
            print(f"  Val: {val_artifact.version}")
            print(f"  Test: {test_artifact.version}")
            
            # Create separate HeteroGraphDataset instances for each split
            print("Creating HeteroGraphDataset instances from downloaded CSV files...")
            train_dataset = HeteroGraphDataset(
                input_path=train_dir + "/", 
                nodeType=nodeType, 
                taskType=task, 
                device=device, 
                data_percentage=percent_data,  # Data is already split, use all of it
                seed=seed
            )
            val_dataset = HeteroGraphDataset(
                input_path=val_dir + "/", 
                nodeType=nodeType, 
                taskType=task, 
                device=device, 
                data_percentage=1.0,
                seed=seed
            )
            test_dataset = HeteroGraphDataset(
                input_path=test_dir + "/", 
                nodeType=nodeType, 
                taskType=task, 
                device=device, 
                data_percentage=1.0,
                seed=seed
            )

            metadata.update({
                "source": "wandb",
                "train_artifact_version": train_artifact.version,
                "val_artifact_version": val_artifact.version,
                "test_artifact_version": test_artifact.version,
            })

            print(f"✓ Datasets loaded from WANDB artifacts")
            if return_metadata:
                return train_dataset, val_dataset, test_dataset, metadata
            return train_dataset, val_dataset, test_dataset
        
        except Exception as e:
            # Artifact doesn't exist or download failed, create new datasets
            print(f"Failed to download from WANDB: {e}")
            print(f"Falling back to local dataset creation...")
            success = False
    
    if not success or not use_wandb_datasets:
        print("Creating new datasets from local pre-split folders...")
        
        # Assume scenes_dir contains train/, val/, and test/ subdirectories
        train_dir = os.path.join(scenes_dir, "train")
        val_dir = os.path.join(scenes_dir, "val")
        test_dir = os.path.join(scenes_dir, "test")
        
        # Verify directories exist
        for split_dir, split_name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(
                    f"Expected {split_name} directory not found at {split_dir}. "
                    f"Please ensure {scenes_dir} contains train/, val/, and test/ subdirectories."
                )
        
        # Create separate HeteroGraphDataset instances for each split
        print(f"Loading from local directories:")
        print(f"  Train: {train_dir}")
        print(f"  Val: {val_dir}")
        print(f"  Test: {test_dir}")
        
        train_dataset = HeteroGraphDataset(
            input_path=train_dir + "/", 
            nodeType=nodeType, 
            taskType=task, 
            device=device, 
            data_percentage=percent_data,  # Data is already split, use all of it
            seed=seed
        )
        val_dataset = HeteroGraphDataset(
            input_path=val_dir + "/", 
            nodeType=nodeType, 
            taskType=task, 
            device=device, 
            data_percentage=1.0,
            seed=seed
        )
        test_dataset = HeteroGraphDataset(
            input_path=test_dir + "/", 
            nodeType=nodeType, 
            taskType=task, 
            device=device, 
            data_percentage=1.0,
            seed=seed
        )
        
        metadata.update({
            "source": "local",
            "train_dir": train_dir,
            "val_dir": val_dir,
            "test_dir": test_dir,
        })

        if return_metadata:
            return train_dataset, val_dataset, test_dataset, metadata
        return train_dataset, val_dataset, test_dataset
