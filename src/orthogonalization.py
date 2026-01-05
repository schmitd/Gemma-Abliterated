"""
Orthogonalization module for applying weight modifications to reduce refusal behavior.
Contains the core logic for computing and applying refusal directions.
"""

import logging
import torch
from tqdm import tqdm
from typing import Optional

logger = logging.getLogger(__name__)

def get_orthogonalized_matrix(matrix: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Orthogonalize matrix with respect to direction (blog post method), silently skipping invalid inputs.
    
    Args:
        matrix: Weight matrix to orthogonalize
        direction: Refusal direction vector
        
    Returns:
        Orthogonalized matrix or original matrix if orthogonalization fails
    """
    try:
        norm = direction.norm()
        if norm == 0 or torch.isnan(norm):
            return matrix
        direction = direction / norm
        if direction.device != matrix.device:
            direction = direction.to(matrix.device)
        proj = torch.outer(direction, direction)
        if torch.isnan(proj).any():
            return matrix
        result_matrix = matrix - matrix @ proj
        if torch.isnan(result_matrix).any():
            return matrix
        return result_matrix
    except Exception as e:
        logger.warning(f"Error in orthogonalization: {e}")
        return matrix

def compute_direct_refusal_direction(model, layer: int) -> Optional[torch.Tensor]:
    """
    Compute refusal direction: harmful_mean - harmless_mean. Returns None if invalid.
    
    Args:
        model: The abliterator model
        layer: Layer index to compute direction for
        
    Returns:
        Normalized refusal direction vector or None if computation fails
    """
    act_key = f"blocks.{layer}.hook_resid_pre"
    if act_key not in model.harmful or act_key not in model.harmless:
        act_key = f"blocks.{layer}.hook_resid_post"
    if act_key not in model.harmful or act_key not in model.harmless:
        return None
    
    try:
        harmful_mean = torch.mean(model.harmful[act_key], dim=0)
        harmless_mean = torch.mean(model.harmless[act_key], dim=0)
        if torch.isnan(harmful_mean).any() or torch.isnan(harmless_mean).any():
            return None
        refusal_dir = harmful_mean - harmless_mean
        norm_val = refusal_dir.norm()
        if norm_val == 0 or torch.isnan(norm_val):
            return None
        return refusal_dir / norm_val
    except Exception as e:
        logger.warning(f"Error computing refusal direction for layer {layer}: {e}")
        return None

def apply_weight_orthogonalization(model) -> None:
    """
    Apply weight orthogonalization using blog post methodology (skip layer 0).
    
    Args:
        model: The abliterator model to modify
    """
    num_layers = model.model.cfg.n_layers
    logger.info(f"Starting orthogonalization of {num_layers} layers")
    
    for layer in tqdm(range(num_layers), desc="Orthogonalizing layers"):
        if layer == 0:
            continue
        refusal_dir = compute_direct_refusal_direction(model, layer)
        if refusal_dir is None:
            logger.warning(f"Skipping layer {layer}: could not compute refusal direction")
            continue
        try:
            block = model.model.blocks[layer]
            # Orthogonalize attention output weights
            block.attn.W_O.data = get_orthogonalized_matrix(block.attn.W_O.data, refusal_dir)
            # Orthogonalize MLP output weights
            block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out.data, refusal_dir)
        except Exception as e:
            logger.error(f"Error orthogonalizing layer {layer}: {e}")
    
    logger.info("Orthogonalization complete")
