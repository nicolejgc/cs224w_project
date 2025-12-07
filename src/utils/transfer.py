"""
Transfer Learning Utilities for DAR

This module implements the transfer learning pipeline from the DAR paper:
1. Train on synthetic graphs with capacity features
2. Replace capacity encoder with domain-specific encoders (vessel features)
3. Freeze backbone, train new encoders on synthetic data
4. Fine-tune on real-world data

Reference: https://arxiv.org/pdf/2302.04496
"""

import torch
from torch.nn import Module, ModuleDict, Linear
from typing import List, Dict, Optional
import copy


def freeze_model(model: Module, exclude: Optional[List[str]] = None):
    """
    Freeze all parameters in a model except those in exclude list.
    
    Args:
        model: The PyTorch model to freeze
        exclude: List of parameter name patterns to keep trainable
                 e.g., ["encoders.length", "encoders.curvature"]
    
    Example:
        # Freeze everything except new encoders (all existing encoders like pos, s, t are frozen)
        freeze_model(net, exclude=["encoders.length", "encoders.distance", "encoders.curvedness"])
    """
    exclude = exclude or []
    
    for name, param in model.named_parameters():
        should_freeze = True
        for pattern in exclude:
            if pattern in name:
                should_freeze = False
                break
        
        param.requires_grad = not should_freeze
        
    # Print frozen/trainable status
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def unfreeze_model(model: Module):
    """Unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True


def replace_encoder(
    model: Module,
    old_encoder_name: str,
    new_encoder_names: List[str],
    in_features: int = 1,
    num_hidden: int = 64,
    combine_method: str = "concat"  # "concat" or "sum"
):
    """
    Replace an encoder with new domain-specific encoders.
    
    This implements the encoder swapping from the DAR paper where:
    - The capacity encoder (A) is removed
    - New encoders for vessel features are added
    
    Args:
        model: The model containing encoders (e.g., MFNet_Impl or EncodeProcessDecode_Impl)
        old_encoder_name: Name of encoder to remove (e.g., "A" for capacity)
        new_encoder_names: Names of new encoders (e.g., ["length", "curvature", "bifurc_dist"])
        in_features: Input dimension for each new encoder
        num_hidden: Hidden dimension (should match model's num_hidden)
        combine_method: How to combine multiple encoders ("concat" or "sum")
    
    Returns:
        Modified model with new encoders
    
    Example:
        replace_encoder(
            model.net_.net,  # Access the EncodeProcessDecode_Impl
            old_encoder_name="A",
            new_encoder_names=["length", "curvature", "bifurc_dist"],
            num_hidden=64
        )
    """
    # Access the encoders ModuleDict
    if hasattr(model, 'net_'):
        # MF_Net wrapper
        encoders = model.net_.net.encoders
        num_hidden = model.net_.net.num_hidden
    elif hasattr(model, 'net'):
        # MFNet_Impl wrapper
        encoders = model.net.encoders
        num_hidden = model.net.num_hidden
    elif hasattr(model, 'encoders'):
        # Direct EncodeProcessDecode_Impl
        encoders = model.encoders
        num_hidden = model.num_hidden
    else:
        raise ValueError("Could not find encoders in model")
    
    # Remove old encoder if it exists
    if old_encoder_name in encoders:
        print(f"Removing encoder: {old_encoder_name}")
        del encoders[old_encoder_name]
    
    # Add new encoders
    for name in new_encoder_names:
        print(f"Adding new encoder: {name} (in_features={in_features}, out_features={num_hidden})")
        encoders[name] = Linear(in_features, num_hidden, bias=False)
    
    return model


def add_skip_connection_encoder(
    model: Module,
    encoder_name: str,
    in_features: int,
    num_hidden: int
):
    """
    Add an encoder that will be used with skip connections.
    
    This is used in Phase 3 of DAR transfer learning where
    edge embeddings are combined with skip connections.
    
    Args:
        model: The model to modify
        encoder_name: Name for the new encoder
        in_features: Input feature dimension
        num_hidden: Output dimension (should match model)
    """
    if hasattr(model, 'net_'):
        encoders = model.net_.net.encoders
    elif hasattr(model, 'net'):
        encoders = model.net.encoders
    elif hasattr(model, 'encoders'):
        encoders = model.encoders
    else:
        raise ValueError("Could not find encoders in model")
    
    encoders[encoder_name] = Linear(in_features, num_hidden, bias=False)
    print(f"Added skip connection encoder: {encoder_name}")


def save_backbone(model: Module, path: str, exclude_encoders: Optional[List[str]] = None):
    """
    Save model weights, optionally excluding certain encoders.
    
    Args:
        model: Model to save
        path: Path to save weights
        exclude_encoders: Encoder names to exclude from saving
    """
    exclude_encoders = exclude_encoders or []
    
    state_dict = model.state_dict()
    
    # Remove excluded encoders
    keys_to_remove = []
    for key in state_dict.keys():
        for encoder_name in exclude_encoders:
            if f"encoders.{encoder_name}" in key:
                keys_to_remove.append(key)
                break
    
    for key in keys_to_remove:
        del state_dict[key]
        print(f"Excluded from save: {key}")
    
    torch.save(state_dict, path)
    print(f"Saved backbone to: {path}")


def load_backbone(model: Module, path: str, strict: bool = False):
    """
    Load backbone weights, allowing for missing/extra keys.
    
    Args:
        model: Model to load weights into
        path: Path to saved weights
        strict: If False, allows missing keys (for new encoders)
    """
    state_dict = torch.load(path, map_location='cpu')
    
    # Load with strict=False to allow new encoders
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    
    if missing:
        print(f"Missing keys (expected for new encoders): {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    
    print(f"Loaded backbone from: {path}")


def get_trainable_params(model: Module) -> List[str]:
    """Get names of trainable parameters."""
    return [name for name, param in model.named_parameters() if param.requires_grad]


def get_frozen_params(model: Module) -> List[str]:
    """Get names of frozen parameters."""
    return [name for name, param in model.named_parameters() if not param.requires_grad]


# ============================================================================
# Brain Vessel Feature Specs
# ============================================================================

def get_vessel_feature_specs():
    """
    Get CLRS specs for brain vessel features.
    
    These replace the capacity (A) feature from synthetic training:
    - length: Edge feature (vessel segment length)
    - curvature: Edge feature (vessel curvature)
    - bifurc_dist: Edge feature (distance between bifurcation points)
    
    Returns:
        Dict of feature specs compatible with CLRS
    """
    import clrs
    
    return {
        # New vessel-specific edge features
        "length": (clrs.Location.EDGE, clrs.Type.SCALAR),
        "curvature": (clrs.Location.EDGE, clrs.Type.SCALAR),
        "bifurc_dist": (clrs.Location.EDGE, clrs.Type.SCALAR),
        
        # Keep standard graph structure features
        "adj": (clrs.Location.EDGE, clrs.Type.MASK),
        "s": (clrs.Location.NODE, clrs.Type.MASK_ONE),
        "t": (clrs.Location.NODE, clrs.Type.MASK_ONE),
    }


# ============================================================================
# Transfer Learning Pipeline
# ============================================================================

class TransferPipeline:
    """
    Complete transfer learning pipeline for DAR.
    
    Usage:
        pipeline = TransferPipeline(model)
        
        # Phase 1: Already done - model trained on synthetic data with capacity
        
        # Phase 2: Replace encoders
        pipeline.replace_capacity_with_vessel_features()
        
        # Phase 3: Freeze backbone, train new encoders
        pipeline.freeze_for_encoder_training()
        pipeline.train_on_synthetic(dataloader, epochs=100)
        
        # Phase 4: Fine-tune on real data
        pipeline.unfreeze_for_finetuning(layers_to_unfreeze=["processor"])
        pipeline.train_on_real(dataloader, epochs=50)
    """
    
    def __init__(self, model: Module):
        self.model = model
        self.original_state = None
    
    def save_checkpoint(self, path: str):
        """Save full model checkpoint."""
        torch.save(self.model.state_dict(), path)
    
    def load_checkpoint(self, path: str):
        """Load full model checkpoint."""
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
    
    def replace_capacity_with_vessel_features(
        self,
        vessel_features: List[str] = ["length", "curvature", "bifurc_dist"]
    ):
        """
        Phase 2: Replace capacity encoder with vessel feature encoders.
        """
        print("\n" + "="*60)
        print("PHASE 2: Replacing capacity encoder with vessel features")
        print("="*60)
        
        replace_encoder(
            self.model,
            old_encoder_name="A",
            new_encoder_names=vessel_features
        )
        
        print(f"\nNew encoders added: {vessel_features}")
        print("Capacity encoder (A) removed")
    
    def freeze_for_encoder_training(
        self,
        new_encoder_names: List[str] = ["length", "curvature", "bifurc_dist"]
    ):
        """
        Phase 3: Freeze all weights except new encoders.
        """
        print("\n" + "="*60)
        print("PHASE 3: Freezing backbone for encoder training")
        print("="*60)
        
        exclude_patterns = [f"encoders.{name}" for name in new_encoder_names]
        freeze_model(self.model, exclude=exclude_patterns)
    
    def unfreeze_for_finetuning(self, layers_to_unfreeze: Optional[List[str]] = None):
        """
        Phase 4: Optionally unfreeze some layers for fine-tuning.
        
        Args:
            layers_to_unfreeze: Patterns to match for unfreezing
                               e.g., ["processor", "decoders"]
        """
        print("\n" + "="*60)
        print("PHASE 4: Preparing for fine-tuning")
        print("="*60)
        
        if layers_to_unfreeze is None:
            # Unfreeze everything
            unfreeze_model(self.model)
            print("All parameters unfrozen")
        else:
            # Unfreeze specific layers
            for name, param in self.model.named_parameters():
                for pattern in layers_to_unfreeze:
                    if pattern in name:
                        param.requires_grad = True
                        break
            
            trainable = get_trainable_params(self.model)
            print(f"Unfrozen layers matching: {layers_to_unfreeze}")
            print(f"Total trainable params: {len(trainable)}")

