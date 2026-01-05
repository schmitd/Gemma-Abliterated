#!/usr/bin/env python3
"""
Model saving script for the Gemma ablation project.
Saves the abliterated model in various formats using the modular structure.
"""

import sys
import logging

from src.utils import setup_logging
from src.config import MODEL_ID, GEMMA3_CHAT_TEMPLATE, POSITIVE_STRINGS, NEGATIVE_STRINGS
from src.model_manager import ModelManager
from src.orthogonalization import apply_weight_orthogonalization
from src.model_saver import ModelSaver

def main():
    """Save the abliterated model."""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Loading model: {MODEL_ID}")
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Create and configure model
        model = model_manager.create_model()
        
        # Reset to clean state
        model_manager.reset_model()
        
        # Apply orthogonalization
        logger.info("Applying weight orthogonalization...")
        apply_weight_orthogonalization(model)
        
        # Initialize model saver
        saver = ModelSaver()
        
        # Save model info
        model_info = {
            "Model": MODEL_ID,
            "Modified": "True",
            "Modification": "Weight orthogonalization for refusal reduction",
            "Chat template": GEMMA3_CHAT_TEMPLATE,
            "Positive tokens": POSITIVE_STRINGS,
            "Negative tokens": NEGATIVE_STRINGS,
        }
        
        # Save in HuggingFace format
        saver.save_huggingface_format(model, model_manager.tokenizer, model_info)
        
        # Create Ollama Modelfile
        saver.create_ollama_modelfile()
        
        # Save abliterator cache
        saver.save_abliterator_cache(model)
        
        # Print usage instructions
        print(saver.get_usage_instructions())
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
