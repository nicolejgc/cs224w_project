"""
VesselGraph Data Loader for DAR Transfer Learning

SIMPLE SETUP:
    1. Inspect the data:
       uv run src/build_data.py inspect-vessel ./VesselGraph
       
    2. Convert to DAR format:
       uv run src/build_data.py vessel ../VesselGraph --output ./src/data/vessel

VesselGraph edge features:
- length: vessel segment length
- distance: Euclidean distance between nodes  
- curveness: vessel curveness (1.0 = straight)

Reference: https://arxiv.org/pdf/2302.04496
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import json
import sys
import numpy as np


def load_vesselgraph(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load VesselGraph data from VesselGraph repository.
    
    Args:
        path: Path to submodule of VesselGraph repo (e.g., "./VesselGraph")
        
    Returns:
        edge_index: [2, num_edges] array of edges
        node_features: [num_nodes, num_features] array (3D positions)
        edge_features: Dict with 'length', 'distance', 'curveness', etc.
        
    Usage:
        edge_index, node_feat, edge_feat = load_vesselgraph("./VesselGraph")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Path not found: {path}\n\n"
            f"Correct the path to VesselGraph repo first\n"
            f"Original VesselGraph repo: https://github.com/jocpae/VesselGraph.git"
        )
    
    # Check if this is a VesselGraph repo (has source/ directory)
    if not (path / "source").exists():
        raise RuntimeError(
            f"Not a VesselGraph repo: {path}\n\n"
            f"Expected source/ directory.\n"
            f"Original VesselGraph repo: https://github.com/jocpae/VesselGraph.git"
        )
    
    return _load_from_vesselgraph_repo(path)


def _load_from_vesselgraph_repo(repo_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load using VesselGraph's own loaders."""
    print(f"Found VesselGraph repo at: {repo_path}")
    
    # Add VesselGraph source paths so internal imports work
    # vessap_utils.py is in source/pytorch_dataset/, and link_dataset.py 
    # does "from vessap_utils import *", so we need that dir in sys.path
    paths_to_add = [
        str(repo_path / "source"),
        str(repo_path / "source" / "pytorch_dataset"),  # for vessap_utils
        str(repo_path),
    ]
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)
    
    try:
        # Fix PyTorch 2.6+ compatibility: allow torch_geometric classes in torch.load
        import torch
        try:
            from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
            from torch_geometric.data.storage import GlobalStorage
            torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
        except ImportError:
            pass  # torch_geometric not installed or different version
        
        from pytorch_dataset.link_dataset import LinkVesselGraph
        
        # Load synthetic_1 with spatial features
        name = "synthetic_graph_1"
        print(f"Loading {name}...")
        
        data_root = str(repo_path / "data")
        dataset = LinkVesselGraph(name=name, root=data_root)
        graph = dataset[0]
        
        return _extract_graph_data(graph)
        
    except ImportError as e:
        print(f"VesselGraph loader import failed: {e}")
        raise
    
    except Exception as e:
        print(f"VesselGraph loader failed: {e}")
        raise


def _extract_graph_data(graph) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Extract numpy arrays from PyG Data object or OGB-style graph dict."""
    # Show what's in the graph object
    if hasattr(graph, 'keys'):
        print(f"Graph keys: {list(graph.keys())}\n")
    else:
        print(f"Graph attributes: {[k for k in dir(graph) if not k.startswith('_') and not callable(getattr(graph, k, None))][:15]}")
    
    edge_index = np.array(graph['edge_index'])
    
    # Node features
    if 'x' in graph and graph['x'] is not None:
        node_features = np.array(graph['x'])
    elif 'node_feat' in graph and graph['node_feat'] is not None:
        node_features = np.array(graph['node_feat'])
    else:
        num_nodes = graph.get('num_nodes', int(edge_index.max()) + 1)
        node_features = np.zeros((num_nodes, 3))
    
    # Print node feature column names if available
    if 'node_attr_keys' in graph and graph['node_attr_keys'] is not None:
        print(f"node_attr_keys (column names): {list(graph['node_attr_keys'])}\n")
    
    # Edge features - check edge_attr_keys for actual column names
    edge_features = {}
    if 'edge_attr' in graph and graph['edge_attr'] is not None:
        edge_attr = np.array(graph['edge_attr'])
        # print(f"Raw edge_attr shape: {edge_attr.shape}")
        # print(f"Raw edge_attr (first 3 edges):\n{edge_attr[:3]}")
        
        # Get actual column names from edge_attr_keys if available
        if 'edge_attr_keys' in graph and graph['edge_attr_keys'] is not None:
            keys = list(graph['edge_attr_keys'])
            print(f"edge_attr_keys (column names): {keys}\n")
            # Map columns by name
            for i, key in enumerate(keys):
                if i < edge_attr.shape[1]:
                    edge_features[key] = edge_attr[:, i]
        elif edge_attr.ndim == 2 and edge_attr.shape[1] >= 3:
            # Fallback to assumed order
            print(f"No edge_attr_keys found, using assumed order: [0]=length, [1]=distance, [2]=avgRadiusAvg, [3]=roundnessAvg, [4]=curveness")
            edge_features['length'] = edge_attr[:, 0]
            edge_features['distance'] = edge_attr[:, 1]
            if edge_attr.shape[1] >= 5:
                edge_features['avgRadiusAvg'] = edge_attr[:, 2]
                edge_features['roundnessAvg'] = edge_attr[:, 3]
                edge_features['curveness'] = edge_attr[:, 4]
    
    return edge_index, node_features, edge_features


def vesselgraph_to_dar_format(
    edge_index: np.ndarray,
    node_features: np.ndarray,
    edge_features: Dict[str, np.ndarray],
    source_node: int,
    target_node: int,
    run_algorithm: bool = True,
    pad_to: int = 64,  # Pad to fixed size for batching
) -> Dict:
    """
    Convert VesselGraph data to DAR training format.
    
    If run_algorithm=True, runs Ford-Fulkerson to generate proper hints and outputs.
    Otherwise, returns format with empty hints/outputs (for inference).
    
    All matrices are padded to pad_to x pad_to for consistent batching.
    """
    num_nodes = node_features.shape[0]
    
    # Ensure we don't exceed pad_to
    if num_nodes > pad_to:
        raise ValueError(f"Subgraph has {num_nodes} nodes but pad_to={pad_to}")
    
    # Create adjacency matrix (padded)
    adj = np.zeros((pad_to, pad_to))
    for src, dst in edge_index.T:
        adj[src, dst] = 1
        adj[dst, src] = 1
    
    # Convert edge list to matrix (padded)
    def to_matrix(edge_values):
        mat = np.zeros((pad_to, pad_to))
        for i, (src, dst) in enumerate(edge_index.T):
            mat[src, dst] = edge_values[i]
            mat[dst, src] = edge_values[i]
        return mat
    
    def normalize(mat):
        if mat.max() > mat.min():
            return (mat - mat.min()) / (mat.max() - mat.min())
        return mat
    
    # Get features (with fallbacks)
    if 'length' in edge_features:
        length_mat = normalize(to_matrix(edge_features['length']))
    else:
        lengths = np.linalg.norm(node_features[edge_index[0]] - node_features[edge_index[1]], axis=1)
        length_mat = normalize(to_matrix(lengths))
    
    if 'distance' in edge_features:
        distance_mat = normalize(to_matrix(edge_features['distance']))
    else:
        distance_mat = length_mat.copy()
    
    if 'curveness' in edge_features:
        curveness_mat = normalize(to_matrix(edge_features['curveness']))
    else:
        curveness_mat = np.zeros_like(adj)
    
    if run_algorithm:
        # Run Ford-Fulkerson to get hints and outputs
        # The algorithm will run on the padded matrices (isolated nodes don't affect min-cut)
        from utils.data.algorithms.graphs import ford_fulkerson_mincut_vessel
        
        f, probes = ford_fulkerson_mincut_vessel(
            length_mat, distance_mat, curveness_mat, adj,
            source_node, target_node
        )
        
        # probes is already in DAR format (ProbesDict)
        return probes
    else:
        # Return format without running algorithm (for inference)
        s = np.zeros(num_nodes)
        s[source_node] = 1
        t = np.zeros(num_nodes)
        t[target_node] = 1
        pos = np.arange(num_nodes) / num_nodes
        
        return {
            "input": {
                "node": {"pos": pos, "s": s, "t": t},
                "edge": {
                    "length": length_mat,
                    "distance": distance_mat,
                    "curveness": curveness_mat,
                    "adj": adj,
                },
                "graph": {}
            },
            "hint": {"node": {}, "edge": {}, "graph": {}},
            "output": {"node": {}, "edge": {}}
        }



def sample_subgraph(
    edge_index: np.ndarray,
    node_features: np.ndarray, 
    edge_features: Dict[str, np.ndarray],
    source: int,
    target: int,
    max_nodes: int = 64,
    num_hops: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], int, int]:
    """
    Extract a subgraph around source and target nodes using BFS.
    
    Returns subgraph with at most max_nodes nodes, plus remapped source/target indices.
    """
    from collections import deque
    
    # Build adjacency list
    num_nodes = node_features.shape[0]
    adj_list = [[] for _ in range(num_nodes)]
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edge_index.T):
        adj_list[u].append(v)
        adj_list[v].append(u)
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx
    
    # BFS from both source and target
    visited = set([source, target])
    queue = deque([(source, 0), (target, 0)])
    
    while queue and len(visited) < max_nodes:
        node, depth = queue.popleft()
        if depth >= num_hops:
            continue
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
                if len(visited) >= max_nodes:
                    break
    
    # Create node mapping (old -> new indices)
    old_to_new = {old: new for new, old in enumerate(sorted(visited))}
    new_to_old = {new: old for old, new in old_to_new.items()}
    
    # Extract subgraph edges
    sub_edges = []
    sub_edge_indices = []
    for old_u in visited:
        for old_v in adj_list[old_u]:
            if old_v in visited and old_u < old_v:  # avoid duplicates
                sub_edges.append([old_to_new[old_u], old_to_new[old_v]])
                sub_edge_indices.append(edge_to_idx[(old_u, old_v)])
    
    if not sub_edges:
        # Fallback: just source and target with no edges
        sub_edge_index = np.array([[0], [1]])
        sub_node_features = node_features[[source, target]]
        sub_edge_features = {k: np.array([0.0]) for k in edge_features}
        return sub_edge_index, sub_node_features, sub_edge_features, 0, 1
    
    sub_edges = np.array(sub_edges)
    sub_edge_index = np.vstack([sub_edges[:, 0], sub_edges[:, 1]])
    # Make undirected
    sub_edge_index = np.hstack([sub_edge_index, sub_edge_index[::-1]])
    sub_edge_indices = sub_edge_indices + sub_edge_indices  # duplicate for reverse edges
    
    # Extract node features
    old_indices = [new_to_old[i] for i in range(len(visited))]
    sub_node_features = node_features[old_indices]
    
    # Extract edge features
    sub_edge_features = {}
    for name, values in edge_features.items():
        sub_edge_features[name] = values[sub_edge_indices]
    
    new_source = old_to_new[source]
    new_target = old_to_new[target]
    
    return sub_edge_index, sub_node_features, sub_edge_features, new_source, new_target


def create_dar_dataset_from_vesselgraph(
    vesselgraph_path: str,
    output_path: str,
    num_samples: int = 500,
    algorithm: str = "ford_fulkerson_mincut_vessel",
    max_nodes: int = 64,  # Subgraph size limit
):
    """
    Create DAR training dataset from VesselGraph.
    
    Args:
        vesselgraph_path: Path to submodule of VesselGraph repo
        output_path: Where to save the DAR dataset
        num_samples: Number of source/target pairs to generate
        algorithm: Algorithm spec name
        max_nodes: Maximum nodes per subgraph (default 64, like CLRS)
    """
    # Load data
    edge_index, node_features, edge_features = load_vesselgraph(vesselgraph_path)
    num_nodes = node_features.shape[0]
    
    print(f"Loaded: {num_nodes} nodes, {edge_index.shape[1]} edges")
    print(f"Features: {list(edge_features.keys())}")
    print(f"Sampling subgraphs with max {max_nodes} nodes each...")
    
    # Generate random source/target pairs and sample subgraphs
    np.random.seed(42)
    samples = []
    for i in range(num_samples):
        s, t = np.random.choice(num_nodes, 2, replace=False)
        
        # Sample subgraph around source/target
        sub_edge_index, sub_node_feat, sub_edge_feat, new_s, new_t = sample_subgraph(
            edge_index, node_features, edge_features, int(s), int(t), max_nodes=max_nodes
        )
        
        sample = vesselgraph_to_dar_format(sub_edge_index, sub_node_feat, sub_edge_feat, new_s, new_t, pad_to=max_nodes)
        samples.append(sample)
        
        if (i + 1) % 100 == 0:
            print(f"  Created {i + 1}/{num_samples} samples")
    
    # Save splits
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_end = int(0.8 * num_samples)
    val_end = int(0.9 * num_samples)
    
    with open(output_path / f"train_{algorithm}.pkl", 'wb') as f:
        pickle.dump(samples[:train_end], f)
    with open(output_path / f"val_{algorithm}.pkl", 'wb') as f:
        pickle.dump(samples[train_end:val_end], f)
    with open(output_path / f"test_{algorithm}.pkl", 'wb') as f:
        pickle.dump(samples[val_end:], f)
    
    with open(output_path / "config.json", 'w') as f:
        json.dump({
            "algorithm": algorithm,
            "num_samples": num_samples,
            "original_num_nodes": num_nodes,
            "max_subgraph_nodes": max_nodes,
            "vessel_features": ["length", "distance", "curveness"]
        }, f, indent=2)
    
    print(f"\n✓ Created DAR dataset at {output_path}")
    print(f"  Train: {train_end}, Val: {val_end - train_end}, Test: {num_samples - val_end}")


def inspect_vesselgraph(path: str):
    """
    Inspect VesselGraph data and print available features.
    
    Args:
        path: Path to submodule of VesselGraph repo
    """
    print(f"\n{'='*70}")
    print(f"INSPECTING VESSELGRAPH")
    print(f"{'='*70}\n")
    print(f"Path: {path}\n")
    
    try:
        edge_index, node_features, edge_features = load_vesselgraph(path)
        
        print(f"Graph Statistics:")
        print(f"  Nodes: {node_features.shape[0]:,}")
        print(f"  Edges: {edge_index.shape[1]:,}")
        print(f"  Node feature shape: {node_features.shape}")
        
        print(f"\nEdge Features:")
        if edge_features:
            for name, values in edge_features.items():
                print(f"  - {name}:")
                print(f"      Shape: {values.shape}")
                print(f"      Min: {values.min():.4f}, Max: {values.max():.4f}, Mean: {values.mean():.4f}")
        else:
            print("  (No edge features found)")
        
        return edge_index, node_features, edge_features
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None
