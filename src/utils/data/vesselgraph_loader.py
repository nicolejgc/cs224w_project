"""
VesselGraph Data Loader for DAR Transfer Learning

SIMPLE SETUP:
    1. Clone VesselGraph:
       git clone https://github.com/jocpae/VesselGraph.git ../VesselGraph
       
    2. Inspect the data:
       uv run src/build_data.py inspect-vessel ../VesselGraph
       
    3. Convert to DAR format:
       uv run src/build_data.py vessel ../VesselGraph --output ./src/data/vessel

VesselGraph edge features:
- length: vessel segment length
- distance: Euclidean distance between nodes  
- curvedness: vessel curvedness

Reference: https://arxiv.org/pdf/2302.04496
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json
import sys


def load_vesselgraph(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load VesselGraph data from cloned repository.
    
    Args:
        path: Path to cloned VesselGraph repo (e.g., "../VesselGraph")
        
    Returns:
        edge_index: [2, num_edges] array of edges
        node_features: [num_nodes, num_features] array (3D positions)
        edge_features: Dict with 'length', 'distance', 'curvedness'
        
    Usage:
        git clone https://github.com/jocpae/VesselGraph.git ../VesselGraph
        edge_index, node_feat, edge_feat = load_vesselgraph("../VesselGraph")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Path not found: {path}\n\n"
            f"Clone VesselGraph first:\n"
            f"  git clone https://github.com/jocpae/VesselGraph.git {path}"
        )
    
    # Check if this is a VesselGraph repo (has source/ directory)
    if (path / "source").exists():
        return _load_from_vesselgraph_repo(path)
    
    # Try loading from numpy files directly
    result = _load_from_numpy_files(path)
    if result is not None:
        return result
    
    raise RuntimeError(
        f"Could not find VesselGraph data in: {path}\n\n"
        f"Expected VesselGraph repo with source/ directory.\n"
        f"Clone: git clone https://github.com/jocpae/VesselGraph.git"
    )


def _load_from_vesselgraph_repo(repo_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load using VesselGraph's own loaders."""
    print(f"Found VesselGraph repo at: {repo_path}")
    
    # Add VesselGraph source to path
    source_path = str(repo_path / "source")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
        sys.path.insert(0, str(repo_path))
    
    try:
        from pytorch_dataset.link_dataset import LinkVesselGraph
        
        # Load synthetic_1 with spatial features
        name = "ogbl-vessel_synth_1_spatial"
        print(f"Loading {name}...")
        
        data_root = str(repo_path / "data")
        dataset = LinkVesselGraph(name=name, root=data_root)
        graph = dataset[0]
        
        return _extract_graph_data(graph)
        
    except ImportError as e:
        print(f"VesselGraph loader import failed: {e}")
        print("Trying to load from numpy files instead...")
        return _load_from_numpy_files(repo_path / "data")
    except Exception as e:
        print(f"VesselGraph loader failed: {e}")
        raise


def _load_from_numpy_files(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    """Load from raw numpy files."""
    
    # Search for edge_index.npy in various locations
    search_paths = [
        path,
        path / "raw",
        path / "processed",
        path / "ogbl-vessel_synth_1_spatial" / "raw",
    ]
    
    for data_dir in search_paths:
        edge_index_path = data_dir / "edge_index.npy"
        if edge_index_path.exists():
            print(f"Loading from: {data_dir}")
            
            edge_index = np.load(edge_index_path)
            
            # Load node features
            node_features = None
            for name in ["node_feat.npy", "node-feat.npy", "pos.npy"]:
                if (data_dir / name).exists():
                    node_features = np.load(data_dir / name)
                    break
            
            if node_features is None:
                num_nodes = int(edge_index.max()) + 1
                node_features = np.zeros((num_nodes, 3))
            
            # Load edge features
            edge_features = {}
            
            # Try combined edge_attr
            for name in ["edge_attr.npy", "edge-attr.npy"]:
                if (data_dir / name).exists():
                    edge_attr = np.load(data_dir / name)
                    if edge_attr.ndim == 2 and edge_attr.shape[1] >= 3:
                        edge_features['length'] = edge_attr[:, 0]
                        edge_features['distance'] = edge_attr[:, 1]
                        edge_features['curvedness'] = edge_attr[:, 2]
                    break
            
            return edge_index, node_features, edge_features
    
    return None


def _extract_graph_data(graph: dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Extract numpy arrays from OGB-style graph dict."""
    edge_index = np.array(graph['edge_index'])
    
    # Node features
    if 'x' in graph and graph['x'] is not None:
        node_features = np.array(graph['x'])
    elif 'node_feat' in graph and graph['node_feat'] is not None:
        node_features = np.array(graph['node_feat'])
    else:
        num_nodes = graph.get('num_nodes', int(edge_index.max()) + 1)
        node_features = np.zeros((num_nodes, 3))
    
    # Edge features: [length, distance, curvedness]
    edge_features = {}
    if 'edge_attr' in graph and graph['edge_attr'] is not None:
        edge_attr = np.array(graph['edge_attr'])
        if edge_attr.ndim == 2 and edge_attr.shape[1] >= 3:
            edge_features['length'] = edge_attr[:, 0]
            edge_features['distance'] = edge_attr[:, 1]
            edge_features['curvedness'] = edge_attr[:, 2]
    
    return edge_index, node_features, edge_features


def vesselgraph_to_dar_format(
    edge_index: np.ndarray,
    node_features: np.ndarray,
    edge_features: Dict[str, np.ndarray],
    source_node: int,
    target_node: int,
) -> Dict:
    """Convert VesselGraph data to DAR training format."""
    num_nodes = node_features.shape[0]
    
    # Create adjacency matrix
    adj = np.zeros((num_nodes, num_nodes))
    for src, dst in edge_index.T:
        adj[src, dst] = 1
        adj[dst, src] = 1
    
    # Convert edge list to matrix
    def to_matrix(edge_values):
        mat = np.zeros((num_nodes, num_nodes))
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
    
    if 'curvedness' in edge_features:
        curvedness_mat = normalize(to_matrix(edge_features['curvedness']))
    else:
        curvedness_mat = np.zeros_like(adj)
    
    # Source/target one-hot
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
                "curvedness": curvedness_mat,
                "adj": adj,
            },
            "graph": {}
        },
        "hint": {"node": {}, "edge": {}, "graph": {}},
        "output": {"node": {}, "edge": {}}
    }


def create_dar_dataset_from_vesselgraph(
    vesselgraph_path: str,
    output_path: str,
    num_samples: int = 500,
    algorithm: str = "ford_fulkerson_mincut_vessel",
):
    """
    Create DAR training dataset from VesselGraph.
    
    Args:
        vesselgraph_path: Path to cloned VesselGraph repo
        output_path: Where to save the DAR dataset
        num_samples: Number of source/target pairs to generate
        algorithm: Algorithm spec name
    """
    # Load data
    edge_index, node_features, edge_features = load_vesselgraph(vesselgraph_path)
    num_nodes = node_features.shape[0]
    
    print(f"Loaded: {num_nodes} nodes, {edge_index.shape[1]} edges")
    print(f"Features: {list(edge_features.keys())}")
    
    # Generate random source/target pairs
    np.random.seed(42)
    samples = []
    for i in range(num_samples):
        s, t = np.random.choice(num_nodes, 2, replace=False)
        sample = vesselgraph_to_dar_format(edge_index, node_features, edge_features, int(s), int(t))
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
            "num_nodes": num_nodes,
            "vessel_features": ["length", "distance", "curvedness"]
        }, f, indent=2)
    
    print(f"\n✓ Created DAR dataset at {output_path}")
    print(f"  Train: {train_end}, Val: {val_end - train_end}, Test: {num_samples - val_end}")


def inspect_vesselgraph(path: str):
    """
    Inspect VesselGraph data and print available features.
    
    Args:
        path: Path to cloned VesselGraph repo
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
        
        print(f"\nNode Features (first 3):")
        print(node_features[:3])
        
        return edge_index, node_features, edge_features
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


# For backwards compatibility
VESSELGRAPH_DATASETS = {
    "synthetic_1", "synthetic_2", "synthetic_3",
    "BALBc_no1", "C57BL6_no1", "CD1-E_no1",
}
