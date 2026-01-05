#!/usr/bin/env python3
"""
Finetuning script for Gemma models.
Performs Supervised Fine-Tuning (SFT) for factual prompts, 
using the original Gemma as a baseline.
"""

import os
import torch
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import MODEL_ID as DEFAULT_MODEL_ID, get_device

def setup_finetuning():
    """Set up the finetuning environment and logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    return logger

def main():
    logger = setup_finetuning()
    
    # Configuration
    # We use google/gemma-2b (base) as the baseline if requested, 
    # but default to the config's model if it's already set to a base version.
    # For this script, we'll specifically target the base gemma-2b to demonstrate 
    # SFT from scratch.
    base_model_id = "google/gemma-2b" 
    dataset_id = "sciq" # Science questions for factual grounding
    output_dir = "./gemma-2b-factual-sft"
    
    logger.info(f"Loading base model: {base_model_id}")
    
    # 1. Quantization configuration (4-bit for efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # 2. Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Recommended for SFT
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 3. Prepare model for PEFT (LoRA)
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    logger.info("Model prepared for PEFT training")

    # 4. Load factual dataset
    logger.info(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id, split="train[:2000]") # Use a subset for faster demonstration
    
    def formatting_func(example):
        """Format the dataset for instruction tuning."""
        return f"Question: {example['question']}\nAnswer: {example['correct_answer']}"

    # 5. Training Configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=100,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        bf16=True,
        push_to_hub=False,
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        args=training_args,
    )

    # 7. Start Training
    logger.info("Starting SFT training...")
    trainer.train()

    # 8. Save the resulting model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Finetuning complete!")

if __name__ == "__main__":
    main()
