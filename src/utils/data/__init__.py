from .loader import load_dataset, Loader


def adj_mat(features):
    for inp in features.inputs:
        if inp.name == "adj":
            return inp.data


def edge_attr_mat(features):
    """
    Get edge capacity matrix.
    
    For standard algorithms: returns 'A' (capacity)
    For vessel transfer learning: returns adjacency (model learns capacity via encoders)
    """
    # First try to find 'A' (standard capacity)
    for inp in features.inputs:
        if inp.name == "A":
            return inp.data
    
    # For vessel data: use adjacency as placeholder
    # The model learns effective capacity through the vessel feature encoders
    # (length, distance, curveness) rather than hardcoding a formula
    for inp in features.inputs:
        if inp.name == "adj":
            return inp.data
    
    return None
