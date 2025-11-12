"""I/O utilities for saving and loading data."""
import json
import pickle
from pathlib import Path
from typing import Any, Union


def dump(obj: Any, path: Union[Path, str]):
    """Save object to file (supports .pkl, .pickle, .json, .pth, .yaml, .yml extensions)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    ext = path.suffix.lower()
    
    if ext in ['.pkl', '.pickle']:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    elif ext == '.json':
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)
    elif ext == '.pth':
        import torch
        torch.save(obj, path)
    elif ext in ['.yaml', '.yml']:
        import yaml
        with open(path, 'w') as f:
            yaml.safe_dump(obj, f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def load(path: Union[Path, str]) -> Any:
    """Load object from file (supports .pkl, .pickle, .json, .pth, .yaml, .yml extensions)"""
    path = Path(path)
    
    ext = path.suffix.lower()
    
    if ext in ['.pkl', '.pickle']:
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif ext == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif ext == '.pth':
        import torch
        return torch.load(path)
    elif ext in ['.yaml', '.yml']:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

