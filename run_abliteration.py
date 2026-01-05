#!/usr/bin/env python3
"""
Main experiment runner for the Gemma ablation project.
Uses the modular codebase to run the complete ablation experiment.
"""

import sys
import logging

from src.utils import setup_logging
from src.config import TEST_SET_SIZE
from src.model_manager import ModelManager
from src.orthogonalization import apply_weight_orthogonalization
from src.evaluation import benchmark_model, compare_results
from src.model_saver import ModelSaver

def main():
    """Run the complete ablation experiment."""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Gemma ablation experiment")
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Create and configure model
        model = model_manager.create_model()
        
        # Reset to clean state for baseline
        model_manager.reset_model()
        
        # Get test sets
        harmful_test, harmless_test = model_manager.get_test_sets(TEST_SET_SIZE)
        
        print("\n" + "="*50)
        print("BASELINE BENCHMARK (Before Ablation)")
        print("="*50)
        
        # Run baseline benchmarks
        baseline_harmful = benchmark_model(model, harmful_test, "Baseline - Harmful Prompts")
        baseline_harmless = benchmark_model(model, harmless_test, "Baseline - Harmless Prompts")
        
        print("\n" + "="*50)
        print("APPLYING WEIGHT ORTHOGONALIZATION")
        print("="*50)
        
        # Apply orthogonalization
        apply_weight_orthogonalization(model)
        
        print("\n" + "="*50)
        print("POST-ORTHOGONALIZATION BENCHMARK")
        print("="*50)
        
        # Run post-orthogonalization benchmarks
        ablated_harmful = benchmark_model(model, harmful_test, "Orthogonalized - Harmful Prompts")
        ablated_harmless = benchmark_model(model, harmless_test, "Orthogonalized - Harmless Prompts")
        
        print("\n" + "="*50)
        print("COMPARISON RESULTS")
        print("="*50)
        
        # Compare results
        compare_results(baseline_harmful, ablated_harmful, "Harmful Prompts")
        compare_results(baseline_harmless, ablated_harmless, "Harmless Prompts")
        
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETE")
        print("="*50)
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Error in experiment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
