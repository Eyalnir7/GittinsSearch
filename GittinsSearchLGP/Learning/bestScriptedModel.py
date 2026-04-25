import math
from typing import Dict, Tuple, List
import torch.nn as nn
import torch
from torch_geometric.nn import SAGEConv

FEATURE_NODE_TYPES = ('ssBox', 'place_frame', 'object', 'ssCylinder')
# Node types that represent constraints and typically only use positional/learnable embeddings
CONSTRAINT_NODE_TYPES = ('pick', 'place')

# The edge types are still necessary for HeteroConv
detailed_edge_types = [('object', 'close_edge', 'ssBox'), ('ssBox', 'close_edge', 'object'), ('place_frame', 'close_edge', 'ssBox'), ('ssBox', 'close_edge', 'place_frame'), ('place_frame', 'close_edge', 'object'), ('object', 'close_edge', 'place_frame'), ('pick', 'time_edge', 'place'), ('place', 'time_edge', 'pick'), ('object', 'time_edge', 'object'), ('ssBox', 'time_edge', 'ssBox'), ('place_frame', 'time_edge', 'place_frame'), ('ssCylinder', 'time_edge', 'ssCylinder'), ('object', 'pick_edge', 'pick'), ('pick', 'pick_edge', 'object'), ('place_frame', 'pick_edge', 'pick'), ('pick', 'pick_edge', 'place_frame'), ('ssCylinder', 'pick_edge', 'pick'), ('pick', 'pick_edge', 'ssCylinder'), ('object', 'place_edge', 'place'), ('place', 'place_edge', 'object'), ('ssCylinder', 'place_edge', 'place'), ('place', 'place_edge', 'ssCylinder'), ('place_frame', 'place_edge', 'place'), ('place', 'place_edge', 'place_frame')]


class OutputHead(nn.Module):
    """Output head for graph-level predictions.
    
    Args:
        hidden_dim: Dimension of input embeddings
        output_dim: Dimension of output predictions
        dropout_rate: Dropout rate for regularization
    """
    def __init__(self, hidden_dim: int, output_dim: int = 1, dropout_rate: float = 0.0, activation=nn.ReLU()):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.layer = nn.Sequential(
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 4),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        # Only squeeze if output_dim is 1
        if self.output_dim == 1:
            return out.squeeze(-1)
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
        hidden_dim: int = 128,
        pe_dim: int = 4,
        num_message_passing_layers: int = 3,
        dropout_rate: float = 0.1420325863085976,
        output_dim: int = 1,
        use_layer_norm: bool = False,
        activation = nn.ReLU()
    ):
        super(ScriptableConstraintGNN, self).__init__()
        # Store all configuration parameters
        self.dropout_rate = dropout_rate
        self.pe_dim = pe_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_layer_norm = bool(use_layer_norm)

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
        self.activation = activation
        
        for node_type, dim in features_dim_per_type.items():
            dim += pe_dim
            self.node_embedders[node_type] = nn.Linear(dim, hidden_dim)
        self.node_embedders['pick'] = nn.Linear(pe_dim, hidden_dim)
        self.node_embedders['place'] = nn.Linear(pe_dim, hidden_dim)

        self.input_norms = nn.ModuleDict()
        self.mp_norms = nn.ModuleDict()
        if self.use_layer_norm:
            for node_type in ['ssBox', 'place_frame', 'object', 'ssCylinder', 'pick', 'place']:
                self.input_norms[node_type] = nn.LayerNorm(hidden_dim)
                self.mp_norms[node_type] = nn.LayerNorm(hidden_dim)


        edge_types: List[Tuple[str, str, str]] = [('object', 'close_edge', 'ssBox'), ('ssBox', 'close_edge', 'object'), ('place_frame', 'close_edge', 'ssBox'), ('ssBox', 'close_edge', 'place_frame'), ('place_frame', 'close_edge', 'object'), ('object', 'close_edge', 'place_frame'), ('pick', 'time_edge', 'place'), ('place', 'time_edge', 'pick'), ('object', 'time_edge', 'object'), ('ssBox', 'time_edge', 'ssBox'), ('place_frame', 'time_edge', 'place_frame'), ('ssCylinder', 'time_edge', 'ssCylinder'), ('object', 'pick_edge', 'pick'), ('pick', 'pick_edge', 'object'), ('place_frame', 'pick_edge', 'pick'), ('pick', 'pick_edge', 'place_frame'), ('ssCylinder', 'pick_edge', 'pick'), ('pick', 'pick_edge', 'ssCylinder'), ('object', 'place_edge', 'place'), ('place', 'place_edge', 'object'), ('ssCylinder', 'place_edge', 'place'), ('place', 'place_edge', 'ssCylinder'), ('place_frame', 'place_edge', 'place'), ('place', 'place_edge', 'place_frame')]
        self.hetero_conv_list = nn.ModuleList()
        for _ in range(num_message_passing_layers):
            conv_dict: Dict[Tuple[str, str, str], SAGEConv] = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(self.hidden_dim, self.hidden_dim)
            self.hetero_conv_list.append(ScriptableHeteroConv(conv_dict))

        dropout_rate = float(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = OutputHead(hidden_dim, output_dim, dropout_rate, self.activation)
        
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
    
    @torch.jit.export
    def positional_encoding_from_positions(
        self, positions: torch.Tensor, pe_dim: int, device: torch.device
    ) -> torch.Tensor:
        # positions: [N] long/int or float
        pos = positions.to(dtype=torch.float, device=device).unsqueeze(1)  # [N, 1]

        pe = torch.zeros((pos.size(0), pe_dim), device=device, dtype=torch.float)

        div_term = torch.exp(
            torch.arange(0, pe_dim, 2, device=device, dtype=torch.float) *
            (-math.log(10000.0) / pe_dim)
        )  # [pe_dim/2]

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe

    # This new forward method replaces the original '_forward_single' logic
    @torch.jit.export
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        times_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],   # NEW
    ) -> torch.Tensor:
            
        # Get the device from an input tensor (assuming all are on the same device)
        device = x_dict['ssCylinder'].device
            
        initial_embeddings: Dict[str, torch.Tensor] = {}
        out_x_dict: Dict[str, torch.Tensor] = {}

        ######################################################################################################################################
        # 1. Feature Node Embeddings
        feature_node_types = ['ssBox', 'place_frame', 'object', 'ssCylinder']
        for node_type in feature_node_types:
            ##################################################################################
            # create the features with PE
            if node_type not in x_dict:
                continue
            
            features = x_dict[node_type]
            # Skip if no nodes
            if features.size(0) == 0:
                continue

            times = times_dict[node_type]
            pe_feats = self.positional_encoding_from_positions(times, self.pe_dim, device)
            if times.numel() > 0 and pe_feats.numel() > 0:
                features = torch.cat([features, pe_feats], dim=1)
            ##################################################################################

            ##################################################################################
            # Embed using the corresponding embedder (TorchScript-safe loop)
            embedder_found = False
            for embedder_key, embedder_module in self.node_embedders.items():
                if embedder_key == node_type:
                    x = embedder_module(features)
                    out_x_dict[node_type] = x
                    initial_embeddings[node_type] = x.clone()
            ##################################################################################
        
        ##################################################################################
        # embed pick and place nodes
        constraint_node_types = ['pick', 'place']
        for node_type in constraint_node_types:
            # Skip if node type doesn't exist in times_dict
            if node_type not in times_dict:
                continue
            
            times = times_dict[node_type]
            # Skip if no nodes of this type
            if times.numel() == 0:
                continue
                
            features = self.positional_encoding_from_positions(times, self.pe_dim, device)

            # TorchScript-safe lookup in node_embedders
            for embedder_key, embedder_module in self.node_embedders.items():
                if embedder_key == node_type:
                    x = embedder_module(features)
                    out_x_dict[node_type] = x
        ##################################################################################
        # apply activations
        x_dict_mp = out_x_dict
        for node_type in x_dict_mp.keys():
            if self.use_layer_norm:
                for norm_key, norm_module in self.input_norms.items():
                    if norm_key == node_type:
                        x_dict_mp[node_type] = norm_module(x_dict_mp[node_type])
            x_dict_mp[node_type] = self.activation(x_dict_mp[node_type])

        ######################################################################################################################################
        # Message Passing Layers
        for hetero_conv in self.hetero_conv_list:
            x_dict_mp = hetero_conv(x_dict_mp, edge_index_dict)
            
            # Apply activation and dropout
            for node_type in x_dict_mp.keys():
                if self.use_layer_norm:
                    for norm_key, norm_module in self.mp_norms.items():
                        if norm_key == node_type:
                            x_dict_mp[node_type] = norm_module(x_dict_mp[node_type])
                x_dict_mp[node_type] = self.activation(x_dict_mp[node_type])
                if self.dropout_rate > 0:
                    x_dict_mp[node_type] = self.dropout(x_dict_mp[node_type])

        ######################################################################################################################################
        # Graph-level pooling using batch_dict["ssCylinder"] (all graphs have cylinder nodes)
        b = batch_dict["ssCylinder"]
        if b.numel() == 0:
            num_graphs = 1
        else:
            num_graphs = int(b.max().item()) + 1

        graph_embedding = torch.zeros(
            (num_graphs, self.hidden_dim),
            device=device,
            dtype=torch.float,
        )

        # Pool final constraint embeddings (after message passing)
        for node_type in ['pick', 'place']:
            if node_type in x_dict_mp and node_type in batch_dict and x_dict_mp[node_type].numel() > 0:
                graph_embedding.index_add_(
                    0,
                    batch_dict[node_type].to(device=device),
                    x_dict_mp[node_type],
                )

        # Pool selected initial embeddings (your existing logic)
        for node_type in ['ssBox', 'ssCylinder', 'place_frame', 'object']:
            if node_type in initial_embeddings and node_type in batch_dict and initial_embeddings[node_type].numel() > 0:
                graph_embedding.index_add_(
                    0,
                    batch_dict[node_type].to(device=device),
                    initial_embeddings[node_type],
                )

        out = self.output_layer(graph_embedding)   # [num_graphs, output_dim]
        return out
 

def forward_heteroBatch(model: ScriptableConstraintGNN, hetero_batch) -> torch.Tensor:
    x_dict = {}
    times_dict = {}
    batch_dict = {}

    for node_type in hetero_batch.node_types:
        # x (only for feature types that have x)
        if hasattr(hetero_batch[node_type], 'x'):
            x_dict[node_type] = hetero_batch[node_type].x

        # times (required by your model; create zeros if missing)
        num_nodes = hetero_batch[node_type].num_nodes
        device = hetero_batch[node_type].x.device if hasattr(hetero_batch[node_type], 'x') else torch.device('cpu')
        if hasattr(hetero_batch[node_type], 'times'):
            times_dict[node_type] = hetero_batch[node_type].times
        else:
            times_dict[node_type] = torch.zeros(num_nodes, dtype=torch.long, device=device)

        # batch vector (required for pooling)
        if hasattr(hetero_batch[node_type], 'batch'):
            batch_dict[node_type] = hetero_batch[node_type].batch
        else:
            # single-graph fallback: all zeros
            batch_dict[node_type] = torch.zeros(num_nodes, dtype=torch.long, device=device)

    edge_index_dict = {}
    for edge_type in hetero_batch.edge_types:
        edge_type_str = edge_type[0] + "___" + edge_type[1] + "___" + edge_type[2]
        edge_index_dict[edge_type_str] = hetero_batch[edge_type].edge_index

    return model.forward(
        x_dict=x_dict,
        times_dict=times_dict,
        edge_index_dict=edge_index_dict,
        batch_dict=batch_dict,
    )

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
        edge_index_dict=edge_index_dict,
        num_pick_nodes=num_pick_nodes,
        num_place_nodes=num_place_nodes
    )