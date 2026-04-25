from torch_geometric.data import HeteroData
import torch
import re
import ast
import numpy as np

FEATURE_DIMS = {
    'ssBox': 4,
    'place_frame': 4,
    'object': 4,
    'ssCylinder': 3,
}

def get_close_obstacles(scene_dict, object, collision_pairs, threshold=1.0):
    # search in the dictionary for objects with 'absolute_pose' less than threshold distance from the given object
    frozenset_collision_pairs = set()
    for pair in collision_pairs:
        frozenset_collision_pairs.add(frozenset([pair[0],pair[1]]))
    if object not in scene_dict:
        return []
    close_obstacles = []
    object_pos = np.array(scene_dict[object]['absolute_pose'])
    for key, value in scene_dict.items():
        if key==object or not value.get("absolute_pose", None) or (not frozenset([key, object]) in frozenset_collision_pairs and "goal" not in key):
            continue
        obs_pose = np.array(value['absolute_pose'])
        if np.linalg.norm(obs_pose - object_pos) < threshold:
            close_obstacles.append(key)
    return close_obstacles

def get_pairCollision(scene_dict):
    collision_pairs = set()
    for key, value in scene_dict.items():
        if value.get("joint", None):
            if key != "ego":
                collision_pairs.add(frozenset((key, "ego")))
            for key2, value2 in scene_dict.items():
                if key2 != key and "wall" in key2.lower():
                    collision_pairs.add(frozenset((key, key2)))
    return [tuple(pair) for pair in collision_pairs]

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

def get_features(node_dict, device="cpu"):
    if node_dict['shape'] == "ssBox":
        size = node_dict['size'][:2]
    if node_dict['shape'] == "ssCylinder":
        size = [node_dict['size'][0]]
    position = node_dict['absolute_pose'][:2]
    return torch.tensor(position + size, dtype=torch.float, device=device)

def get_hetero_data_input(scene_dict, task_plan, device=None, action_number=None):
    relevant_nodes = {"ssBox": {'names': [], 'features': [], 'times': []},
                      "place_frame": {'names': [], 'features': [], 'times': []},
                      "object": {'names': [], 'features': [], 'times': []},
                      "ssCylinder": {'names': [], 'features': [], 'times': []},
                      "pick": {'names': [], 'features': [], 'times': []},
                      "place": {'names': [], 'features': [], 'times': []}}
    pair_edges = {"close_edge": {'directed': False, 'edges': []},
                  "time_edge": {'directed': True, 'edges': []},}
    sink_edges = {"pick_edge": {'directed': False, 'edges': []},
                  "place_edge": {'directed': False, 'edges': []}}
    objects_to_times = {}
    
    collision_pairs = get_pairCollision(scene_dict)

    def add_object_to_relevant(obj, i):
        node_type = scene_dict[obj]['shape']
        logical = scene_dict[obj].get("logical")
        if logical:
            is_object = logical.get("is_object", False)
            if is_object:
                node_type = "object"
            is_place = logical.get("is_place", False)
            if is_place:
                node_type = "place_frame"

        features = get_features(scene_dict[obj], device=device)
        relevant_nodes[node_type]['names'].append(obj+f"_{i}")
        relevant_nodes[node_type]['features'].append(features)
        relevant_nodes[node_type]['times'].append(i)
        if obj not in objects_to_times:
            objects_to_times[obj] = set()
        objects_to_times[obj].add(i)
    
    parsed_task_plan = []
    for action in task_plan:
        action_type = 'pick' if 'pick' in action[0] else 'place'
        parsed_task_plan.append([action_type] + action[1:])

    if action_number is not None:
        if action_number == 0:
            parsed_task_plan = [parsed_task_plan[0]]
        else:
            parsed_task_plan = [parsed_task_plan[action_number-1], parsed_task_plan[action_number]]

    for i, action in enumerate(parsed_task_plan):
        relevant_nodes[action[0]]['names'].append(action[0]+f"_{i}")
        relevant_nodes[action[0]]['times'].append(i)
        pick_place_objects = [action[0]+f"_{i}"]
        if i < len(parsed_task_plan)-1:
            pair_edges["time_edge"]['edges'].append((action[0]+f"_{i}", parsed_task_plan[i+1][0]+f"_{i+1}"))
        for object in action[1:]:
            add_object_to_relevant(object, i)
            pick_place_objects.append(object+f"_{i}")
            close_obstacles = get_close_obstacles(scene_dict, object, collision_pairs)

            for obs in close_obstacles:
                add_object_to_relevant(obs, i)
                pair_edges["close_edge"]['edges'].append((object+f"_{i}", obs+f"_{i}"))

        sink_edges[f"{action[0]}_edge"]['edges'].append(tuple(pick_place_objects))
    
    for obj, times in objects_to_times.items():
        times = sorted(list(times))
        for i in range(len(times) - 1):
            pair_edges["time_edge"]['edges'].append((obj+f"_{times[i]}", obj+f"_{times[i+1]}"))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # make the times and actives lists into tensors
    for node_type in relevant_nodes.keys():
        relevant_nodes[node_type]['times'] = torch.tensor(relevant_nodes[node_type]['times'], dtype=torch.long, device=device)
        # if 'actives' in relevant_nodes[node_type]:
        #     relevant_nodes[node_type]['actives'] = torch.tensor(relevant_nodes[node_type]['actives'], dtype=torch.bool).to(device)

    return relevant_nodes, pair_edges, sink_edges

        

def to_hetero_data(relevant_nodes, pair_edges, sink_edges, device=None):
    """_summary_

    :param relevant_nodes: A dictionary where each key is a node type. The values are dictionaries with the following keys: 'names', 'features', 'times'. The values of 'name' are lists of node names, and the values of 'features' are lists of feature vectors corresponding to each node. The 'time' key is the timestep of the task plan when the node is relevant. This will be used later for positional embedding.
    :param pair_edges: A dictionary where the keys are the name of the edge type (for example, 'close_edge' or 'time_edge') and the values are dictionaries with two keys: 'directed' (bool) and 'edges' (list of tuples). Each tuple in 'edges' is (name1_node, name2_node).
    :param sink_edges: Similar to pair_Edges in terms of the datastructure (dictionary of dictionaries...), but each edge tuple is (sink_node, name1_node, ..., nameN_node). name{i}_node are the nodes connected to the sink_node.
    """
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    data = HeteroData()
    name_to_type = {}

    # Add nodes
    for node_type, nodes_info in relevant_nodes.items():
        names = nodes_info['names']
        for name in names:
            name_to_type[name] = node_type
        features = nodes_info['features']
        data[node_type].num_nodes = len(names)
        data[node_type].names = names
        data[node_type].times = nodes_info['times']
        data[node_type].actives = nodes_info.get('actives', [])
        if len(features) > 0:
            # add the features to data[node_type].x. The features are a list of tensors, so we need to stack them
            data[node_type].x = torch.stack(features)
        else:
            if node_type in FEATURE_DIMS:
                data[node_type].x = torch.empty((0, FEATURE_DIMS[node_type]), dtype=torch.float, device=device)

    edge_lists = {}

    def add_edge(src_type, rel, dst_type, src, dst):
        """Utility to accumulate edges by type"""
        key = (src_type, rel, dst_type)
        if key not in edge_lists:
            edge_lists[key] = [[], []]
        edge_lists[key][0].append(src)
        edge_lists[key][1].append(dst)

    for edge_type, edge_info in pair_edges.items():
        for edge in edge_info['edges']:
            node1, node2 = edge
            type1 = name_to_type[node1]
            type2 = name_to_type[node2]
            idx1 = relevant_nodes[type1]['names'].index(node1)
            idx2 = relevant_nodes[type2]['names'].index(node2)
            add_edge(type1, edge_type, type2, idx1, idx2)
            if edge_info.get('directed', False) == False:
                add_edge(type2, edge_type, type1, idx2, idx1)

    for edge_type, sink_edges_info in sink_edges.items():
        for edge in sink_edges_info['edges']:
            sink_node = edge[0]
            connected_nodes = edge[1:]
            sink_type = name_to_type.get(sink_node)
            for conn_node in connected_nodes:
                conn_type = name_to_type.get(conn_node)
                idx_sink = relevant_nodes[sink_type]['names'].index(sink_node)
                idx_conn = relevant_nodes[conn_type]['names'].index(conn_node)
                add_edge(conn_type, edge_type, sink_type, idx_conn, idx_sink)
                if sink_edges_info.get('directed', False) == False:
                    add_edge(sink_type, edge_type, conn_type, idx_sink, idx_conn)

    for (src_type, rel, dst_type), (src, dst) in edge_lists.items():
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_index = torch.unique(edge_index, dim=1)  # Do unique on CPU
        edge_index = edge_index.to(device)
        data[src_type, rel, dst_type].edge_index = edge_index

    return data

if __name__ == "__main__":
    import DataParsing as DP
    print("Testing ToHeteroData module...")
    df = DP.DataFrameParsing.load_all_datasets(4, 4)  # Load only
    print(df.head())
    scene_config = DP.ConfigurationParsing.parse_conf_file("../data_blockObj/z.conf4.g")
    rel_nodes, pair_edges, sink_edges = get_hetero_data_input(scene_config, parse_list_of_lists(df.iloc[0]['plan']))
    hetero_data_example = to_hetero_data(rel_nodes, pair_edges, sink_edges)
    print(list(hetero_data_example.edge_types))