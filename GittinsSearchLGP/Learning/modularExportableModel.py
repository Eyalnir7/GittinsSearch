import math
from typing import Dict, Tuple, List, Optional
import torch.nn as nn
import torch
from torch_geometric.nn import HeteroConv, SAGEConv
# Note: HeteroData is no longer directly used in the forward pass of the scriptable model.

# Node types that have feature data (.x)
FEATURE_NODE_TYPES = ('ssBox', 'place_frame', 'object', 'ssCylinder')
# Node types that represent constraints and typically only use positional/learnable embeddings
CONSTRAINT_NODE_TYPES = ('pick', 'place')

# The edge types are still necessary for HeteroConv
detailed_edge_types = [('object', 'close_edge', 'ssBox'), ('ssBox', 'close_edge', 'object'), ('place_frame', 'close_edge', 'ssBox'), ('ssBox', 'close_edge', 'place_frame'), ('place_frame', 'close_edge', 'object'), ('object', 'close_edge', 'place_frame'), ('pick', 'time_edge', 'place'), ('place', 'time_edge', 'pick'), ('object', 'time_edge', 'object'), ('ssBox', 'time_edge', 'ssBox'), ('place_frame', 'time_edge', 'place_frame'), ('ssCylinder', 'time_edge', 'ssCylinder'), ('object', 'pick_edge', 'pick'), ('pick', 'pick_edge', 'object'), ('place_frame', 'pick_edge', 'pick'), ('pick', 'pick_edge', 'place_frame'), ('ssCylinder', 'pick_edge', 'pick'), ('pick', 'pick_edge', 'ssCylinder'), ('object', 'place_edge', 'place'), ('place', 'place_edge', 'object'), ('ssCylinder', 'place_edge', 'place'), ('place', 'place_edge', 'ssCylinder'), ('place_frame', 'place_edge', 'place'), ('place', 'place_edge', 'place_frame')]

class SharedLinear(nn.Module):
    """
    Linear layer with shared weights and separate biases for each task.
    
    Input:
        x: Tensor of shape (batch_size, n)
    Output:
        y: Tensor of shape (batch_size, K)
           where each column corresponds to a task.
    """
    def __init__(self, in_features, num_tasks):
        super().__init__()
        self.in_features = in_features
        self.num_tasks = num_tasks
        
        # Shared weight vector (like w in your diagram)
        self.weight = nn.Parameter(torch.randn(in_features))
        
        # Separate bias per task (b1, b2, ..., bK)
        self.bias = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, x):
        # shared linear combination
        g = x @ self.weight  # shape: (batch_size,)
        
        # add task-specific bias and apply sigmoid for probabilities
        out = g.unsqueeze(1) + self.bias  # (batch_size, K)
        return out


class ScriptableHeteroConv(nn.Module):
    def __init__(self, convs: Dict[Tuple[str, str, str], nn.Module]):
        super().__init__()
        # TorchScript wants string keys in ModuleDict, so we store a parallel list
        self.edge_types: List[Tuple[str, str, str]] = list(convs.keys())
        self.convs = nn.ModuleDict({self._et_key_str(et): convs[et] for et in self.edge_types})

    def _et_key_str(self, et: Tuple[str, str, str]) -> str:
        # stable, explicit join (avoid relying on Python's tuple repr)
        return et[0] + "___" + et[1] + "___" + et[2]

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {}

        for et in self.edge_types:
            src, _, dst = et
            # Skip if any piece is missing
            if (src not in x_dict) or (dst not in x_dict):
                continue
            et_key = self._et_key_str(et)
            if et_key not in edge_index_dict:
                continue

            x_src = x_dict[src]
            x_dst = x_dict[dst]
            edge_index = edge_index_dict[et_key]

            # Find and call the corresponding conv module by iterating through ModuleDict
            found_module = False
            out_dst = torch.zeros_like(x_dst)  # Initialize with dummy value
            for key, module in self.convs.items():
                if key == et_key:
                    # IMPORTANT: call SAGEConv in bipartite mode:
                    #   - features as a tuple (x_src, x_dst)
                    #   - size=(num_src, num_dst) to set correct dim_size for aggregation
                    out_dst = module(
                        (x_src, x_dst),
                        edge_index,
                        size=(x_src.size(0), x_dst.size(0)),
                    )
                    found_module = True
            
            if not found_module:
                continue

            # sum aggregation over relations targeting the same dst type
            if dst not in out_dict:
                out_dict[dst] = out_dst
            else:
                out_dict[dst] = out_dict[dst] + out_dst

        # For dst types with no incoming edges this layer, keep previous x (identity)
        for ntype in x_dict.keys():
            if ntype not in out_dict:
                out_dict[ntype] = x_dict[ntype]
        return out_dict

class ScriptableConstraintGNN(nn.Module):
    # For TorchScript, the __init__ signature must match the original to ensure 
    # all parameters and submodules are correctly initialized.
    __annotations__ = {}
    def __init__(
        self,
        hidden_dim: int = 64,
        edge_types: List[Tuple[str, str, str]] = detailed_edge_types,
        learnable_constraint_emb: bool = False,
        pe_type: str = "cat_to_features",
        pe_dim: int = 4,
        num_message_passing_layers: int = 3,
        act: Optional[nn.Module] = None,
        dropout_rate: float = 0.2,
        skip_connection: bool = False,
        sum_over_all_nodes: bool = False,
    ):
        super(ScriptableConstraintGNN, self).__init__()
        # Store all configuration parameters
        self.skip_connection = skip_connection
        self.dropout_rate = dropout_rate
        self.sum_over_all_nodes = sum_over_all_nodes
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        self.learnable_constraint_emb = learnable_constraint_emb
        self.hidden_dim = hidden_dim

        # --- Submodule Initialization (Copied from ConstraintGNN) ---
        
        # NOTE: Device selection for PE is moved to the forward pass
        # self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        features_dim_per_type: Dict[str, int] = {
            'ssBox': 4,
            'place_frame': 4,
            'object': 4,
            'ssCylinder': 3,
        }

        self.node_embedders = nn.ModuleDict()
        self.activation = act if act is not None else nn.ReLU()
        
        for node_type, dim in features_dim_per_type.items():
            if pe_type == "cat_to_features":
                dim += pe_dim
            self.node_embedders[node_type] = nn.Linear(dim, hidden_dim)

        if not self.learnable_constraint_emb and self.pe_type == "cat_to_features":
            self.node_embedders['pick'] = nn.Linear(pe_dim, hidden_dim)
            self.node_embedders['place'] = nn.Linear(pe_dim, hidden_dim)
            
        self.constraint_embeddings = nn.Embedding(2, hidden_dim)
        # Always create constraint_embeddings for TorchScript compatibility
        if self.learnable_constraint_emb and self.pe_type == "add_to_hidden":
            self.constraint_embeddings = nn.Embedding(2, hidden_dim)
        elif self.learnable_constraint_emb and self.pe_type == "cat_to_features":
            # Note: The original code has 'hidden_dim - pe_dim', which seems correct 
            # for 'cat_to_features' with learnable constraints.
            self.constraint_embeddings = nn.Embedding(2, hidden_dim - pe_dim)

        # self.hetero_conv_list = nn.ModuleList()
        # for _ in range(num_message_passing_layers):
        #     conv_dict: Dict[Tuple[str, str, str], SAGEConv] = {}
        #     for edge_type in self.edge_types:
        #         conv_dict[edge_type] = SAGEConv(hidden_dim, hidden_dim)
        #     self.hetero_conv_list.append(HeteroConv(conv_dict, aggr='sum'))
        self.hetero_conv_list = nn.ModuleList()
        for _ in range(num_message_passing_layers):
            conv_dict: Dict[Tuple[str, str, str], SAGEConv] = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(self.hidden_dim, self.hidden_dim)
            self.hetero_conv_list.append(ScriptableHeteroConv(conv_dict))

        dropout_rate = float(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Sequential(
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 4),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        # self.to(self.device) is removed as C++ handles device placement

    # Moved from original class. Note: device is now an argument
    @torch.jit.export
    def positional_encoding(self, n_nodes: int, hidden_dim: int, device: torch.device) -> torch.Tensor:
        pe = torch.zeros(n_nodes, hidden_dim, device=device)
        position = torch.arange(0, n_nodes, dtype=torch.float, device=device).unsqueeze(1)
        # Use float for div_term to ensure scriptability on all platforms
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float) * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    # This new forward method replaces the original '_forward_single' logic
    @torch.jit.export
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor],
        times_dict: Dict[str, torch.Tensor],
        actives_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        num_pick_nodes: int,
        num_place_nodes: int
    ) -> torch.Tensor:
        
        # Get the device from an input tensor (assuming all are on the same device)
        # We assume 'pick' times is always present
        device = times_dict['pick'].device
        
        # Calculate PE length to match original implementation: num_pick_nodes * 2
        pe_len = num_pick_nodes * 2
        
        # Generate Positional Encoding
        if self.pe_type == "cat_to_features":
            pe = self.positional_encoding(pe_len, self.pe_dim, device)
        else:
            pe = self.positional_encoding(pe_len, self.hidden_dim, device)
            
        initial_embeddings: Dict[str, torch.Tensor] = {}
        out_x_dict: Dict[str, torch.Tensor] = {}

        # 1. Feature Node Embeddings
        feature_node_types = ['ssBox', 'place_frame', 'object', 'ssCylinder']
        for node_type in feature_node_types:
            # Skip if the graph does not contain this node type (empty tensor in dict)
            if node_type not in x_dict or x_dict[node_type].numel() == 0:
                continue

            features = x_dict[node_type]
            times = times_dict[node_type]
            
            # 1a. Positional Encoding Concatenation
            if self.pe_type == "cat_to_features":
                # Check if PE is needed (times.numel() > 0)
                if times.numel() > 0 and pe.numel() > 0:
                    features = torch.cat([features, pe[times.long()]], dim=1)
                
            # 1b. Masking Logic (Node features at indices 0 and 1)
            if self.mask_active_nodes:
                if node_type in actives_dict and actives_dict[node_type].numel() > 0:
                    actives = actives_dict[node_type]
                    # mask is only applied if both time > 0 and active is True
                    mask = (times.long() > 0) & (actives.to(torch.bool))
                    features = features.clone() # Clone to avoid in-place modification on input
                    # Zero out the first two feature columns (indices 0, 1)
                    features[mask, :2] = 0.0

            # 1c. Embedding - find the correct embedder by iteration
            embedder_found = False
            x = torch.zeros_like(features)  # Initialize with dummy value
            for embedder_key, embedder_module in self.node_embedders.items():
                if embedder_key == node_type:
                    x = embedder_module(features)
                    embedder_found = True
            
            if embedder_found:
                out_x_dict[node_type] = x
                if self.skip_connection:
                    initial_embeddings[node_type] = x.clone()
        
        # 2. Constraint Node Embeddings ('pick', 'place')
        constraint_node_types = ['pick', 'place']
        for node_type in constraint_node_types:
            # Determine the number of nodes for this type
            num_nodes = num_pick_nodes if node_type == 'pick' else num_place_nodes
            
            if num_nodes == 0:
                continue
                
            times = times_dict[node_type]
            
            # 2a. Learnable Constraint Embeddings
            if self.learnable_constraint_emb:
                emb_index = 0 if node_type == "pick" else 1
                constraint_indices = torch.full((num_nodes,), emb_index, dtype=torch.long, device=device)
                features = self.constraint_embeddings(constraint_indices)
                
                if self.pe_type == "cat_to_features":
                    if times.numel() > 0 and pe.numel() > 0:
                        features = torch.cat([features, pe[times.long()]], dim=1)
                
                out_x_dict[node_type] = features  # Do not use embedders for learnable constraint embeddings
            
            # 2b. PE-only Embeddings
            else: # not self.learnable_constraint_emb
                if self.pe_type == "cat_to_features":
                    # PE is the feature, which is then passed to the embedder
                    if times.numel() > 0 and pe.numel() > 0:
                        features = pe[times.long()]
                        # Find embedder by iteration
                        embedder_found = False
                        embedded_features = torch.zeros((num_nodes, self.hidden_dim), dtype=torch.float, device=device)
                        for embedder_key, embedder_module in self.node_embedders.items():
                            if embedder_key == node_type:
                                embedded_features = embedder_module(features)
                                embedder_found = True
                        out_x_dict[node_type] = embedded_features if embedder_found else torch.zeros((num_nodes, self.hidden_dim), dtype=torch.float, device=device)
                    else:
                        # Fallback for zero nodes or empty times, though num_nodes should be 0 then
                        out_x_dict[node_type] = torch.zeros((num_nodes, self.hidden_dim), dtype=torch.float, device=device)
                
                else: # pe_type == "add_to_hidden"
                    # Initialize with zeros, PE will be added later
                    out_x_dict[node_type] = torch.zeros((num_nodes, self.hidden_dim), dtype=torch.float, device=device)


        # 3. Positional Encoding Addition (for all nodes if pe_type == "add_to_hidden")
        if self.pe_type == "add_to_hidden":
            for node_type in out_x_dict.keys():
                times = times_dict.get(node_type)
                if times is not None and times.numel() > 0 and pe.numel() > 0:
                    out_x_dict[node_type] = out_x_dict[node_type] + pe[times.long()]

        # 4. Message Passing Layers
        x_dict_mp = out_x_dict
        for node_type in x_dict_mp.keys():
            x_dict_mp[node_type] = self.activation(x_dict_mp[node_type])


        for hetero_conv in self.hetero_conv_list:
            x_dict_mp = hetero_conv(x_dict_mp, edge_index_dict)
            
            # Apply activation and dropout
            for node_type in x_dict_mp.keys():
                x_dict_mp[node_type] = self.activation(x_dict_mp[node_type])
                if self.dropout_rate > 0:
                    x_dict_mp[node_type] = self.dropout(x_dict_mp[node_type])

        # 5. Global Pooling and Output Layer
        
        # Collect final embeddings
        all_embedding_list = []
        
        # Determine which node embeddings to pool
        if self.sum_over_all_nodes:
            # Need to find all node types that actually produced embeddings
            for node_type in out_x_dict.keys():
                if node_type in x_dict_mp:
                    all_embedding_list.append(x_dict_mp[node_type])
        else:
            # Use constraint node types only
            constraint_node_types_for_pooling = ['pick', 'place']
            for node_type in constraint_node_types_for_pooling:
                if node_type in x_dict_mp:
                    all_embedding_list.append(x_dict_mp[node_type])

        # Add initial embeddings for skip connection
        if self.skip_connection:
            # ssBox and ssCylinder are appended if their embeddings were successfully created
            if 'ssBox' in initial_embeddings:
                all_embedding_list.append(initial_embeddings['ssBox'])
            if 'ssCylinder' in initial_embeddings:
                all_embedding_list.append(initial_embeddings['ssCylinder'])
            
            if ('place_frame' in initial_embeddings) and ('object' in initial_embeddings):
                all_embedding_list.append(initial_embeddings['place_frame'])
                all_embedding_list.append(initial_embeddings['object'])
        
        # If no embeddings were collected (empty graph), return a tensor of zero
        if len(all_embedding_list) == 0:
            return torch.zeros(1, dtype=torch.float, device=device).squeeze()


        # Concatenate and Sum
        all_node_embeddings = torch.cat(all_embedding_list, dim=0)
        graph_embedding = all_node_embeddings.sum(dim=0, keepdim=True)

        # Final output layer
        out = self.output_layer(graph_embedding)
        return out.squeeze() # Return as 1D tensor


def forward_heteroData(model: ScriptableConstraintGNN, hetero_data) -> torch.Tensor:
    """
    Function to forward a HeteroData object through the ScriptableConstraintGNN model.
    
    This function extracts the required inputs from a HeteroData object and calls
    the model's forward method with the appropriate format.
    
    Args:
        model: The ScriptableConstraintGNN model instance
        hetero_data: A HeteroData object containing the graph structure and features
        
    Returns:
        torch.Tensor: Model prediction output
    """
    # Extract x_dict (node features)
    x_dict = {}
    feature_node_types = ['ssBox', 'place_frame', 'object', 'ssCylinder']
    for node_type in feature_node_types:
        if node_type in hetero_data.node_types and hasattr(hetero_data[node_type], 'x'):
            x_dict[node_type] = hetero_data[node_type].x
    
    # Extract times_dict
    times_dict = {}
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], 'times'):
            times_dict[node_type] = hetero_data[node_type].times
        else:
            # If times not present, create a zero tensor of appropriate size
            num_nodes = hetero_data[node_type].num_nodes
            times_dict[node_type] = torch.zeros(num_nodes, dtype=torch.long, device=hetero_data[node_type].x.device if hasattr(hetero_data[node_type], 'x') else torch.device('cpu'))
    
    # Extract actives_dict
    actives_dict = {}
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], 'actives') and hetero_data[node_type].actives is not None:
            # if hetero_data[node_type].actives is not a tensor, create a tensor of False values
            if not isinstance(hetero_data[node_type].actives, torch.Tensor):
                num_nodes = hetero_data[node_type].num_nodes
                actives_dict[node_type] = torch.zeros(num_nodes, dtype=torch.bool, device=hetero_data[node_type].x.device if hasattr(hetero_data[node_type], 'x') else torch.device('cpu'))
            else:
                actives_dict[node_type] = hetero_data[node_type].actives
        else:
            # If actives not present, create a tensor of False values
            num_nodes = hetero_data[node_type].num_nodes
            actives_dict[node_type] = torch.zeros(num_nodes, dtype=torch.bool, device=hetero_data[node_type].x.device if hasattr(hetero_data[node_type], 'x') else torch.device('cpu'))
    
    # Extract edge_index_dict - convert tuple keys to strings
    edge_index_dict = {}
    for edge_type in hetero_data.edge_types:
        # Convert tuple key to string using the same format as ScriptableHeteroConv
        edge_type_str = edge_type[0] + "___" + edge_type[1] + "___" + edge_type[2]
        edge_index_dict[edge_type_str] = hetero_data[edge_type].edge_index
    
    # Count constraint nodes
    num_pick_nodes = hetero_data['pick'].num_nodes if 'pick' in hetero_data.node_types else 0
    num_place_nodes = hetero_data['place'].num_nodes if 'place' in hetero_data.node_types else 0
    
    # Call the model's forward method
    return model.forward(
        x_dict=x_dict,
        times_dict=times_dict,
        actives_dict=actives_dict,
        edge_index_dict=edge_index_dict,
        num_pick_nodes=num_pick_nodes,
        num_place_nodes=num_place_nodes
    )


class OutputMLP(nn.Module):
    """
    Final MLP layer for constraint prediction.
    Takes a graph embedding and outputs a single constraint score.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
        act: Optional[nn.Module] = None,
        output_dim: int = 1,
    ):
        super(OutputMLP, self).__init__()
        self.activation = act if act is not None else nn.ReLU()
        self.output_layer = nn.Sequential(
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 4),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP"""
        return self.output_layer(x)


class BatchableConstraintGNN(nn.Module):
    """
    Batch-compatible version of ConstraintGNN that can handle lists of HeteroData objects for training.
    This version maintains compatibility with the original PE_GNN while being trainable.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        edge_types: List[Tuple[str, str, str]] = detailed_edge_types,
        learnable_constraint_emb: bool = False,
        pe_dim: int = 4,
        num_message_passing_layers: int = 3,
        act: Optional[nn.Module] = None,
        mask_active_nodes: bool = False,
        dropout_rate: float = 0.2,
        skip_connection: bool = False,
        device = None,
        sum_over_all_nodes: bool = False,
        output_dim: int = 1,
    ):
        super(BatchableConstraintGNN, self).__init__()
        
        # Store all configuration parameters
        self.skip_connection = skip_connection
        self.dropout_rate = dropout_rate
        self.sum_over_all_nodes = sum_over_all_nodes
        self.mask_active_nodes = mask_active_nodes
        self.pe_dim = pe_dim
        self.learnable_constraint_emb = learnable_constraint_emb
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Device setup
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Feature dimensions per node type
        features_dim_per_type: Dict[str, int] = {
            'ssBox': 4,
            'place_frame': 4,
            'object': 4,
            'ssCylinder': 3,
        }

        self.node_embedders = nn.ModuleDict()
        self.activation = act if act is not None else nn.ReLU()
        
        # Create node embedders
        for node_type, dim in features_dim_per_type.items():
            dim += pe_dim
            self.node_embedders[node_type] = nn.Linear(dim, hidden_dim)

        # Constraint embedders
        if not self.learnable_constraint_emb:
            self.node_embedders['pick'] = nn.Linear(pe_dim, hidden_dim)
            self.node_embedders['place'] = nn.Linear(pe_dim, hidden_dim)

        # Always create constraint_embeddings for TorchScript compatibility
        if learnable_constraint_emb:
            self.constraint_embeddings = nn.Embedding(2, hidden_dim - pe_dim)
        else:
            self.constraint_embeddings = nn.Embedding(2, hidden_dim)

        # Message passing layers using ScriptableHeteroConv for weight compatibility
        self.hetero_conv_list = nn.ModuleList()
        for _ in range(num_message_passing_layers):
            conv_dict: Dict[Tuple[str, str, str], SAGEConv] = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(hidden_dim, hidden_dim)
            self.hetero_conv_list.append(ScriptableHeteroConv(conv_dict))

        self.dropout = nn.Dropout(dropout_rate)
        self.output_mlp = OutputMLP(
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            act=self.activation,
            output_dim=output_dim,
        )
        
        self.to(self.device)

    def positional_encoding(self, n_nodes: int, hidden_dim: int, device: torch.device) -> torch.Tensor:
        """Generate positional encoding"""
        pe = torch.zeros(n_nodes, hidden_dim, device=device)
        position = torch.arange(0, n_nodes, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float) * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _forward_single_or_batch(self, hetero_data):
        """Forward pass for a single HeteroData object or batched HeteroData object"""
        
        device = hetero_data['pick'].times.device if 'pick' in hetero_data.node_types else self.device
        
        # Check if this is a batched HeteroData object
        is_batched = hasattr(hetero_data, 'batch_size') and hetero_data.batch_size is not None and hetero_data.batch_size > 1
        
        if is_batched:
            # Handle batched HeteroData object
            batch_size = hetero_data.batch_size
            
            # Get constraint node counts per graph from batch information
            # For batched data, we need to calculate nodes per graph, not total nodes
            if 'pick' in hetero_data.node_types and hasattr(hetero_data['pick'], 'ptr'):
                # ptr contains cumulative sum of nodes per graph
                pick_ptr = hetero_data['pick'].ptr
                num_pick_nodes_per_graph = (pick_ptr[1:] - pick_ptr[:-1]).max().item()  # Max pick nodes in any graph
                total_pick_nodes = hetero_data['pick'].num_nodes
            else:
                num_pick_nodes_per_graph = 0
                total_pick_nodes = 0
            
            if 'place' in hetero_data.node_types and hasattr(hetero_data['place'], 'ptr'):
                place_ptr = hetero_data['place'].ptr
                num_place_nodes_per_graph = (place_ptr[1:] - place_ptr[:-1]).max().item()  # Max place nodes in any graph
                total_place_nodes = hetero_data['place'].num_nodes
            else:
                num_place_nodes_per_graph = 0
                total_place_nodes = 0
        else:
            # Handle single HeteroData object (original logic)
            batch_size = 1
            num_pick_nodes_per_graph = hetero_data['pick'].num_nodes if 'pick' in hetero_data.node_types else 0
            num_place_nodes_per_graph = hetero_data['place'].num_nodes if 'place' in hetero_data.node_types else 0
            total_pick_nodes = num_pick_nodes_per_graph
            total_place_nodes = num_place_nodes_per_graph
        
        # Generate PE
        pe_len = num_pick_nodes_per_graph * 2
        pe = self.positional_encoding(pe_len, self.pe_dim, device)
            
        initial_embeddings: Dict[str, torch.Tensor] = {}
        out_x_dict: Dict[str, torch.Tensor] = {}

        # 1. Feature Node Embeddings
        feature_node_types = ['ssBox', 'place_frame', 'object', 'ssCylinder']
        for node_type in feature_node_types:
            if node_type not in hetero_data.node_types or hetero_data[node_type].num_nodes == 0:
                continue

            features = hetero_data[node_type].x
            times = hetero_data[node_type].times
            
            # 1a. Positional Encoding Concatenation
            if times.numel() > 0 and pe.numel() > 0:
                features = torch.cat([features, pe[times.long()]], dim=1)
                
            # 1b. Masking Logic
            if self.mask_active_nodes and hasattr(hetero_data[node_type], 'actives') and hetero_data[node_type].actives is not None:
                actives = hetero_data[node_type].actives
                mask = (times.long() > 0) & (actives.to(torch.bool))
                features = features.clone()
                features[mask, :2] = 0.0

            # 1c. Embedding
            x = self.node_embedders[node_type](features)

            out_x_dict[node_type] = x
            if self.skip_connection:
                initial_embeddings[node_type] = x.clone()
        
        # 2. Constraint Node Embeddings
        constraint_node_types = ['pick', 'place']
        for node_type in constraint_node_types:
            if node_type == 'pick':
                num_nodes_this_type = total_pick_nodes if is_batched else num_pick_nodes_per_graph
            else:  # place
                num_nodes_this_type = total_place_nodes if is_batched else num_place_nodes_per_graph
            
            if num_nodes_this_type == 0:
                continue
                
            times = hetero_data[node_type].times
            
            if self.learnable_constraint_emb:
                emb_index = 0 if node_type == "pick" else 1
                constraint_indices = torch.full((num_nodes_this_type,), emb_index, dtype=torch.long, device=device)
                features = self.constraint_embeddings(constraint_indices)
                
                if times.numel() > 0 and pe.numel() > 0:
                    features = torch.cat([features, pe[times.long()]], dim=1)
                
                out_x_dict[node_type] = features
            else:
                if times.numel() > 0 and pe.numel() > 0:
                    features = pe[times.long()]
                    out_x_dict[node_type] = self.node_embedders[node_type](features)
                else:
                    out_x_dict[node_type] = torch.zeros((num_nodes_this_type, self.hidden_dim), dtype=torch.float, device=device)

        # 4. Message Passing Layers
        x_dict_mp = out_x_dict
        
        # Apply activations
        for node_type in x_dict_mp.keys():
            x_dict_mp[node_type] = self.activation(x_dict_mp[node_type])

        # Convert edge_index_dict keys from tuples to strings for ScriptableHeteroConv
        edge_index_dict_str = {}
        for edge_type, edge_index in hetero_data.edge_index_dict.items():
            edge_type_str = edge_type[0] + "___" + edge_type[1] + "___" + edge_type[2]
            edge_index_dict_str[edge_type_str] = edge_index
        
        for hetero_conv in self.hetero_conv_list:
            x_dict_mp = hetero_conv(x_dict_mp, edge_index_dict_str)
            
            for node_type in x_dict_mp.keys():
                x_dict_mp[node_type] = self.activation(x_dict_mp[node_type])
                if self.dropout_rate > 0:
                    x_dict_mp[node_type] = self.dropout(x_dict_mp[node_type])

        # 5. Global Pooling
        if is_batched:
            # Handle batched pooling - need to separate embeddings by graph
            batch_outputs = []
            
            for batch_idx in range(batch_size):
                graph_embedding_list = []
                
                # Pool embeddings for current graph
                if self.sum_over_all_nodes:
                    for node_type in out_x_dict.keys():
                        if node_type in x_dict_mp and node_type in hetero_data.node_types:
                            # Get batch indices for this node type
                            if hasattr(hetero_data[node_type], 'batch'):
                                node_batch_mask = hetero_data[node_type].batch == batch_idx
                                if node_batch_mask.any():
                                    graph_embedding_list.append(x_dict_mp[node_type][node_batch_mask])
                else:
                    # Use constraint nodes only
                    constraint_node_types_for_pooling = ['pick', 'place']
                    for node_type in constraint_node_types_for_pooling:
                        if node_type in x_dict_mp and node_type in hetero_data.node_types:
                            if hasattr(hetero_data[node_type], 'batch'):
                                node_batch_mask = hetero_data[node_type].batch == batch_idx
                                if node_batch_mask.any():
                                    graph_embedding_list.append(x_dict_mp[node_type][node_batch_mask])
                
                # Add skip connections for current graph
                if self.skip_connection:
                    skip_node_types = ['ssBox', 'ssCylinder']
                    if ('place_frame' in initial_embeddings) and ('object' in initial_embeddings):
                        skip_node_types.extend(['place_frame', 'object'])
                    
                    for node_type in skip_node_types:
                        if node_type in initial_embeddings and node_type in hetero_data.node_types:
                            if hasattr(hetero_data[node_type], 'batch'):
                                node_batch_mask = hetero_data[node_type].batch == batch_idx
                                if node_batch_mask.any():
                                    graph_embedding_list.append(initial_embeddings[node_type][node_batch_mask])
                
                # Pool embeddings for this graph
                if len(graph_embedding_list) == 0:
                    batch_outputs.append(torch.zeros(1, dtype=torch.float, device=device))
                else:
                    graph_embeddings = torch.cat(graph_embedding_list, dim=0)
                    graph_embedding = graph_embeddings.sum(dim=0, keepdim=True)
                    out = self.output_mlp(graph_embedding)
                    batch_outputs.append(out.squeeze())
            
            return torch.stack(batch_outputs)
            
        else:
            # Handle single graph (original logic)
            all_embedding_list = []
            
            if self.sum_over_all_nodes:
                for node_type in out_x_dict.keys():
                    if node_type in x_dict_mp:
                        all_embedding_list.append(x_dict_mp[node_type])
            else:
                constraint_node_types_for_pooling = ['pick', 'place']
                for node_type in constraint_node_types_for_pooling:
                    if node_type in x_dict_mp:
                        all_embedding_list.append(x_dict_mp[node_type])

            # Skip connections
            if self.skip_connection:
                if 'ssBox' in initial_embeddings:
                    all_embedding_list.append(initial_embeddings['ssBox'])
                if 'ssCylinder' in initial_embeddings:
                    all_embedding_list.append(initial_embeddings['ssCylinder'])
                if ('place_frame' in initial_embeddings) and ('object' in initial_embeddings):
                    all_embedding_list.append(initial_embeddings['place_frame'])
                    all_embedding_list.append(initial_embeddings['object'])
            
            if len(all_embedding_list) == 0:
                return torch.zeros(1, dtype=torch.float, device=device).squeeze()

            all_node_embeddings = torch.cat(all_embedding_list, dim=0)
            graph_embedding = all_node_embeddings.sum(dim=0, keepdim=True)

            out = self.output_mlp(graph_embedding)
            return out.squeeze()

    def forward(self, hetero_data_batch):
        """
        Forward pass for batch of HeteroData objects from DataLoader
        
        Args:
            hetero_data_batch: Either:
                - List of individual HeteroData objects from custom collate function
                  (as used in train_WANDB.py training setup)
                - Single batched HeteroData object from PyTorch Geometric's default batching
                - Single HeteroData object for individual inference
            
        Returns:
            torch.Tensor: Batch of predictions, shape (batch_size,)
        """
        if isinstance(hetero_data_batch, list):
            # This is the expected input: list of individual HeteroData objects
            outputs = []
            for hetero_data in hetero_data_batch:
                output = self._forward_single_or_batch(hetero_data)
                outputs.append(output)
            return torch.stack(outputs)
        else:
            # Single HeteroData object or batched HeteroData object 
            return self._forward_single_or_batch(hetero_data_batch)



    def copy_weights_to_scriptable(self, scriptable_model):
        """
        Copy trained weights from this model to a ScriptableConstraintGNN model
        
        Args:
            scriptable_model: ScriptableConstraintGNN instance to copy weights to
        """
        # Copy node embedders
        for key, module in self.node_embedders.items():
            if key in scriptable_model.node_embedders:
                scriptable_model.node_embedders[key].load_state_dict(module.state_dict())
        
        # Copy constraint embeddings if both have them
        if hasattr(self, 'constraint_embeddings') and hasattr(scriptable_model, 'constraint_embeddings'):
            scriptable_model.constraint_embeddings.load_state_dict(self.constraint_embeddings.state_dict())
        
        # Copy hetero conv layers - both now use ScriptableHeteroConv with same structure
        for i, (batch_conv, script_conv) in enumerate(zip(self.hetero_conv_list, scriptable_model.hetero_conv_list)):
            # Both use ScriptableHeteroConv, so direct state dict copy works
            script_conv.load_state_dict(batch_conv.state_dict())
        
        # Copy output layer
        scriptable_model.output_layer.load_state_dict(self.output_mlp.output_layer.state_dict())
        
        print("✓ Successfully copied weights from BatchableConstraintGNN to ScriptableConstraintGNN")