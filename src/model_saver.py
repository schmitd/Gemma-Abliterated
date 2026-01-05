"""
Model saver module for saving models in various formats.
Handles saving abliterated models for different use cases.
"""

import os
import json
import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

class ModelSaver:
    """Handles saving models in different formats."""
    
    def __init__(self, output_dir: str = "abliterated_gemma_2b"):
        """
        Initialize the model saver.
        
        Args:
            output_dir: Directory to save the model
        """
        self.output_dir = output_dir
        
    def save_huggingface_format(self, model, tokenizer, model_info: dict = None) -> None:
        """
        Save model in HuggingFace format.
        
        Args:
            model: The abliterator model
            tokenizer: The tokenizer
            model_info: Additional model information
        """
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Saving model to {self.output_dir}...")
        
        # Save the model state dict
        torch.save(model.model.state_dict(), os.path.join(self.output_dir, "pytorch_model.bin"))
        
        # Save the model config
        config = model.model.cfg
        config_dict = {
            "architectures": ["GemmaForCausalLM"],
            "model_type": "gemma",
            "vocab_size": len(tokenizer),
            "hidden_size": config.d_model,
            "intermediate_size": config.d_mlp,
            "num_hidden_layers": config.n_layers,
            "num_attention_heads": config.n_heads,
            "max_position_embeddings": config.n_ctx,
            "rms_norm_eps": config.eps,
            "rope_theta": getattr(config, 'rope_theta', 10000.0),
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save the tokenizer
        tokenizer.save_pretrained(self.output_dir)
        
        # Save model info
        if model_info:
            with open(os.path.join(self.output_dir, "model_info.txt"), "w") as f:
                for key, value in model_info.items():
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"Model saved successfully to {self.output_dir}")
    
    def create_ollama_modelfile(self, model_name: str = "abliterated-gemma") -> None:
        """
        Create a Modelfile for Ollama.
        
        Args:
            model_name: Name for the Ollama model
        """
        lines = [
            "FROM ./",
            'TEMPLATE """<start_of_turn>user',
            '{{.Input}}<end_of_turn>',
            '<start_of_turn>model',
            '"""',
            'PARAMETER stop "<end_of_turn>"',
            'PARAMETER stop "<start_of_turn>"',
            'PARAMETER temperature 0.7',
            'PARAMETER top_p 0.9',
            'PARAMETER top_k 40',
            'PARAMETER repeat_penalty 1.1',
            'SYSTEM You are a helpful AI assistant. You have been modified to be less likely to refuse requests while maintaining helpfulness and safety.'
        ]
        
        modelfile_content = '\n'.join(lines)
        
        with open(os.path.join(self.output_dir, "Modelfile"), "w") as f:
            f.write(modelfile_content)
        
        logger.info(f"Modelfile created in {self.output_dir}")
    
    def save_abliterator_cache(self, model, cache_fname: str = None) -> None:
        """
        Save the abliterator cache and modifications.
        
        Args:
            model: The abliterator model
            cache_fname: Filename for the cache
        """
        if cache_fname is None:
            cache_fname = os.path.join(self.output_dir, "abliterator_cache.pth")
        
        logger.info(f"Saving abliterator cache to {cache_fname}")
        model.save_activations(cache_fname)
    
    def get_usage_instructions(self) -> str:
        """
        Get usage instructions for the saved model.
        
        Returns:
            Formatted usage instructions
        """
        instructions = f"""
MODEL SAVED SUCCESSFULLY
========================
Model saved to: {self.output_dir}

To use with Python/abliterator:
1. Use the load_abliterated_model() function from use_abliterated_model.py
2. Or load the cache file: cache_fname='{self.output_dir}/abliterator_cache.pth'

To use with Ollama (if supported):
1. cd {self.output_dir}
2. ollama create abliterated-gemma -f Modelfile
3. ollama run abliterated-gemma

The model has been modified to reduce refusal behavior.
Test it with prompts that would normally be refused.
"""
        return instructions
