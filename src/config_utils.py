"""
Utility functions for handling mmengine Config objects.

This module provides lightweight utilities that can be imported without
heavy dependencies like JAX or Haiku.
"""

from mmengine.config import Config, ConfigDict


def config_to_dict(obj):
    """
    Recursively convert mmengine Config/ConfigDict objects to native Python dictionaries.
    
    This function handles nested structures including:
    - mmengine.config.Config objects
    - mmengine.config.ConfigDict objects (dict subclass used internally by mmengine)
    - Regular dictionaries (including nested ones)
    - Lists (with potential nested Config objects)
    - Tuples (preserved as tuples)
    
    Args:
        obj: The object to convert. Can be a Config, ConfigDict, dict, list, 
             tuple, or any other type.
        
    Returns:
        A native Python object with all Config/ConfigDict instances converted to dicts.
        
    Example:
        >>> from mmengine.config import Config
        >>> cfg = Config({'optimizer': {'class_': 'adam', 'kwargs': {'lr': 0.01}}})
        >>> result = config_to_dict(cfg)
        >>> type(result)
        <class 'dict'>
        >>> type(result['optimizer'])
        <class 'dict'>
    """
    # Handle mmengine Config objects
    if isinstance(obj, Config):
        # Config._cfg_dict is the underlying dict, but we can also iterate
        # Convert to dict and recursively process
        return {key: config_to_dict(value) for key, value in obj.items()}
    
    # Handle regular dictionaries (including ConfigDict which is a dict subclass)
    # We explicitly check for dict AFTER Config since Config might also be dict-like
    elif isinstance(obj, dict):
        return {key: config_to_dict(value) for key, value in obj.items()}
    
    # Handle lists
    elif isinstance(obj, list):
        return [config_to_dict(item) for item in obj]
    
    # Handle tuples (preserve as tuple)
    elif isinstance(obj, tuple):
        return tuple(config_to_dict(item) for item in obj)
    
    # Return other types as-is (int, float, str, bool, None, etc.)
    else:
        return obj

