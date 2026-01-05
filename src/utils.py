"""
Utility functions for the Gemma ablation project.
Contains helper functions used across multiple modules.
"""

import re
import logging
from typing import List, Set
from transformers import AutoTokenizer

from .config import POSITIVE_STRINGS, NEGATIVE_STRINGS, REFUSAL_PHRASES

# Configure logging
logger = logging.getLogger(__name__)

def get_token_id_set(tokenizer: AutoTokenizer, strings: List[str]) -> Set[int]:
    """
    Convert a list of strings to their token IDs.
    
    Args:
        tokenizer: The tokenizer to use for encoding
        strings: List of strings to convert to token IDs
        
    Returns:
        Set of token IDs
    """
    token_ids: Set[int] = set()
    for s in strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        token_ids.update(ids)
    return token_ids

def format_prompts(prompts: List[str], template: str) -> List[str]:
    """
    Format prompts using the specified chat template.
    
    Args:
        prompts: List of raw prompts
        template: Chat template string with {instruction} placeholder
        
    Returns:
        List of formatted prompts
    """
    return [template.format(instruction=p) for p in prompts]

def get_response_text(generation_result) -> str:
    """
    Standardize response extraction from model.generate output.
    
    Args:
        generation_result: Output from model.generate
        
    Returns:
        Standardized response text
    """
    if isinstance(generation_result, list):
        return generation_result[0] if generation_result else ""
    return str(generation_result)

def create_refusal_regex() -> re.Pattern:
    """
    Create a regex pattern for detecting refusal phrases.
    
    Returns:
        Compiled regex pattern for refusal detection
    """
    pattern = r"(" + r"|".join([re.escape(p) for p in REFUSAL_PHRASES]) + r")"
    return re.compile(pattern, re.IGNORECASE)

def setup_logging(level: str = "INFO", format_str: str = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        format_str: Logging format string
    """
    if format_str is None:
        format_str = "%(asctime)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def validate_inputs(prompts: List[str], min_length: int = 1) -> bool:
    """
    Validate input prompts.
    
    Args:
        prompts: List of prompts to validate
        min_length: Minimum required length
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(prompts, list):
        logger.error("Prompts must be a list")
        return False
    
    if len(prompts) < min_length:
        logger.error(f"Must provide at least {min_length} prompt(s)")
        return False
    
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip():
            logger.error(f"Prompt {i} is empty or not a string")
            return False
    
    return True
