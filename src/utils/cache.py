"""
src/utils/cache.py
-----------------
Activation caching helpers for saving and loading MLP activations.
"""

from pathlib import Path
import torch
from typing import Dict, List, Optional


def save_activation_cache(cache_dict: dict, path: str) -> None:
    """
    Save activation cache to disk.
    
    Args:
        cache_dict: Dictionary with 'mlp_in', 'mlp_out', 'prompts', etc.
        path: Path to save to
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_dict, path)
    print(f"Saved activation cache to {path}")
    
    # Print summary
    if 'mlp_in' in cache_dict:
        n_layers = len(cache_dict['mlp_in'])
        n_tokens = cache_dict['mlp_in'][0].shape[0]
        print(f"  Layers: {n_layers}")
        print(f"  Tokens: {n_tokens:,}")


def load_activation_cache(path: str, device: str = "cpu") -> dict:
    """
    Load activation cache from disk.
    
    Args:
        path: Path to cache file
        device: Device to load tensors to
        
    Returns:
        Dictionary with activation data
    """
    cache = torch.load(path, map_location=device, weights_only=False)
    print(f"Loaded activation cache from {path}")
    
    # Print summary
    if 'mlp_in' in cache:
        n_layers = len(cache['mlp_in'])
        n_tokens = cache['mlp_in'][0].shape[0]
        model_name = cache.get('model_name', 'unknown')
        print(f"  Model: {model_name}")
        print(f"  Layers: {n_layers}")
        print(f"  Tokens: {n_tokens:,}")
    
    return cache


def merge_activation_caches(cache_paths: List[str], output_path: str) -> None:
    """
    Merge multiple activation caches into one.
    
    Args:
        cache_paths: List of paths to cache files
        output_path: Path to save merged cache
    """
    merged = None
    
    for path in cache_paths:
        cache = load_activation_cache(path, device="cpu")
        
        if merged is None:
            merged = cache
        else:
            # Concatenate along token dimension
            for layer_idx in range(len(cache['mlp_in'])):
                merged['mlp_in'][layer_idx] = torch.cat([
                    merged['mlp_in'][layer_idx],
                    cache['mlp_in'][layer_idx]
                ], dim=0)
                
                merged['mlp_out'][layer_idx] = torch.cat([
                    merged['mlp_out'][layer_idx],
                    cache['mlp_out'][layer_idx]
                ], dim=0)
            
            # Merge prompts
            if 'prompts' in cache:
                merged['prompts'].extend(cache['prompts'])
    
    save_activation_cache(merged, output_path)


def subsample_cache(
    cache: dict,
    max_tokens: Optional[int] = None,
    max_prompts: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Subsample activation cache for faster experimentation.
    
    Args:
        cache: Activation cache dictionary
        max_tokens: Maximum tokens to keep (if set)
        max_prompts: Maximum prompts to keep (if set)
        seed: Random seed
        
    Returns:
        Subsampled cache
    """
    if max_tokens is None and max_prompts is None:
        return cache
    
    torch.manual_seed(seed)
    
    n_tokens = cache['mlp_in'][0].shape[0]
    
    if max_tokens is not None and n_tokens > max_tokens:
        # Random sample of tokens
        indices = torch.randperm(n_tokens)[:max_tokens]
        indices = indices.sort()[0]  # Keep temporal order
        
        new_cache = {
            'mlp_in': [cache['mlp_in'][i][indices] for i in range(len(cache['mlp_in']))],
            'mlp_out': [cache['mlp_out'][i][indices] for i in range(len(cache['mlp_out']))],
            'model_name': cache.get('model_name', 'unknown'),
        }
        
        if 'prompts' in cache:
            # Estimate which prompts are included (approximate)
            new_cache['prompts'] = cache['prompts'][:len(indices) // 10]
        
        return new_cache
    
    return cache


def compute_cache_statistics(cache: dict) -> dict:
    """
    Compute statistics about activation cache.
    
    Args:
        cache: Activation cache
        
    Returns:
        Dictionary of statistics
    """
    n_layers = len(cache['mlp_in'])
    n_tokens = cache['mlp_in'][0].shape[0]
    d_model = cache['mlp_in'][0].shape[1]
    
    stats = {
        'n_layers': n_layers,
        'n_tokens': n_tokens,
        'd_model': d_model,
        'model_name': cache.get('model_name', 'unknown'),
    }
    
    # Per-layer statistics
    for layer_idx in range(n_layers):
        mlp_in = cache['mlp_in'][layer_idx]
        mlp_out = cache['mlp_out'][layer_idx]
        
        stats[f'layer_{layer_idx}_in_mean'] = mlp_in.mean().item()
        stats[f'layer_{layer_idx}_in_std'] = mlp_in.std().item()
        stats[f'layer_{layer_idx}_out_mean'] = mlp_out.mean().item()
        stats[f'layer_{layer_idx}_out_std'] = mlp_out.std().item()
    
    return stats