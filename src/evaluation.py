"""
Evaluation module for benchmarking and testing model performance.
Contains functions for scoring prompts and measuring refusal rates.
"""

import logging
from typing import List, Tuple, Dict
from tqdm import tqdm
import torch

from .config import (
    DEFAULT_SAMPLED_TOKENS, DEFAULT_MAX_GEN_TOKENS, 
    DEFAULT_BATCH_SIZE, GEMMA3_CHAT_TEMPLATE
)
from .utils import format_prompts, get_response_text, create_refusal_regex, validate_inputs

logger = logging.getLogger(__name__)

def score_prompts(model, prompts: List[str], 
                  sampled_token_ct: int = DEFAULT_SAMPLED_TOKENS) -> Tuple[float, float]:
    """
    Score prompts for refusal vs acceptance behavior.
    
    Args:
        model: The abliterator model
        prompts: List of prompts to score
        sampled_token_ct: Number of tokens to sample for scoring
        
    Returns:
        Tuple of (refusal_score, acceptance_score)
    """
    if not validate_inputs(prompts):
        return 0.0, 0.0
    
    try:
        toks = model.model.tokenizer(format_prompts(prompts, GEMMA3_CHAT_TEMPLATE), 
                                    padding=True, truncation=False, return_tensors="pt").input_ids
        logits, _ = model.run_with_cache(toks, max_new_tokens=sampled_token_ct, drop_refusals=False)
        negative_per_batch, positive_per_batch = model.measure_scores_from_logits(logits, sampled_token_ct, measure="max")
        negative = float(torch.mean(negative_per_batch))
        positive = float(torch.mean(positive_per_batch))
        return negative, positive
    except Exception as e:
        logger.error(f"Error in score_prompts: {e}")
        return 0.0, 0.0

def refusal_text_rate(model, prompts: List[str], 
                     max_gen_tokens: int = DEFAULT_MAX_GEN_TOKENS, 
                     batch_size: int = DEFAULT_BATCH_SIZE) -> float:
    """
    Calculate the rate of refusal text in generated responses.
    
    Args:
        model: The abliterator model
        prompts: List of prompts to test
        max_gen_tokens: Maximum tokens to generate per prompt
        batch_size: Batch size for generation
        
    Returns:
        Fraction of responses containing refusal phrases
    """
    if not validate_inputs(prompts):
        return 0.0
    
    all_texts = []
    refusal_regex = create_refusal_regex()
    
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating refusal text samples"):
            batch_prompts = prompts[i:i+batch_size]
            gens = model.generate(batch_prompts, max_tokens_generated=max_gen_tokens, 
                                stop_at_eos=True, drop_refusals=False)
            if isinstance(gens, list):
                all_texts.extend(gens)
            else:
                all_texts.append(gens)
        
        hits = sum(1 for t in all_texts if refusal_regex.search(str(t)) is not None)
        return hits / max(1, len(all_texts))
    except Exception as e:
        logger.error(f"Error in refusal_text_rate: {e}")
        return 0.0

def benchmark_model(model, test_prompts: List[str], 
                   name: str = "Model") -> Dict[str, float]:
    """
    Benchmark model performance on test prompts.
    
    Args:
        model: The abliterator model to benchmark
        test_prompts: List of test prompts
        name: Name for the benchmark run
        
    Returns:
        Dictionary with benchmark results
    """
    if not validate_inputs(test_prompts):
        logger.warning("No valid test prompts provided for benchmarking")
        return {"negative": 0.0, "positive": 0.0, "text_refusal": 0.0}
    
    print(f"\n=== {name} Benchmark (fast) ===")
    
    try:
        negative, positive = score_prompts(model, test_prompts, sampled_token_ct=DEFAULT_SAMPLED_TOKENS)
        text_refusal = refusal_text_rate(model, test_prompts, 
                                       max_gen_tokens=DEFAULT_MAX_GEN_TOKENS, 
                                       batch_size=DEFAULT_BATCH_SIZE)
        
        print(f"Refusal Score: {negative:.4f}")
        print(f"Acceptance Score: {positive:.4f}")
        ratio = (positive / negative) if negative > 0 else float("inf")
        print(f"Ratio (Acceptance/Refusal): {ratio:.4f}")
        print(f"Refusal Text Rate: {text_refusal:.3f}")

        print("\nExample generations:")
        for i, prompt in enumerate(test_prompts[:2]):
            try:
                response = model.generate(prompt, max_tokens_generated=DEFAULT_MAX_GEN_TOKENS, 
                                        stop_at_eos=True, drop_refusals=False)
                response_text = get_response_text(response)
                print(f"Prompt {i+1}: {prompt[:100]}...")
                print(f"Response: {response_text}")
                print()
            except Exception as e:
                logger.error(f"Error generating for prompt {i+1}: {e}")
                print(f"Prompt {i+1}: Error - {e}")
                print()

        return {"negative": negative, "positive": positive, "text_refusal": text_refusal}
    
    except Exception as e:
        logger.error(f"Error in benchmark_model: {e}")
        return {"negative": 0.0, "positive": 0.0, "text_refusal": 0.0}

def compare_results(baseline_results: Dict[str, float], 
                   ablated_results: Dict[str, float], 
                   prompt_type: str = "Prompts") -> None:
    """
    Compare baseline and ablated results.
    
    Args:
        baseline_results: Results from baseline model
        ablated_results: Results from ablated model
        prompt_type: Type of prompts being compared
    """
    print(f"\n{prompt_type}:")
    print(f"  Baseline - Refusal: {baseline_results['negative']:.4f}, "
          f"Acceptance: {baseline_results['positive']:.4f}, "
          f"Text Refusal: {baseline_results['text_refusal']:.3f}")
    print(f"  Ablated - Refusal: {ablated_results['negative']:.4f}, "
          f"Acceptance: {ablated_results['positive']:.4f}, "
          f"Text Refusal: {ablated_results['text_refusal']:.3f}")
    
    # Calculate changes
    refusal_change = ablated_results['negative'] - baseline_results['negative']
    acceptance_change = ablated_results['positive'] - baseline_results['positive']
    text_refusal_change = ablated_results['text_refusal'] - baseline_results['text_refusal']
    
    print(f"  Change - Refusal: {refusal_change:+.4f}, "
          f"Acceptance: {acceptance_change:+.4f}, "
          f"Text Refusal: {text_refusal_change:+.3f}")
    
    # Calculate percent changes
    if baseline_results['negative'] > 0:
        refusal_pct = (refusal_change / baseline_results['negative']) * 100
        print(f"  Refusal % Change: {refusal_pct:+.1f}%")
    
    if baseline_results['text_refusal'] > 0:
        text_refusal_pct = (text_refusal_change / baseline_results['text_refusal']) * 100
        print(f"  Text Refusal % Change: {text_refusal_pct:+.1f}%")
