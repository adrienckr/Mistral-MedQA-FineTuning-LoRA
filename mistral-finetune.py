import argparse
import logging
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tokenize_function(examples, tokenizer):
    # Process batches: create a list of prompts and full texts
    prompts = [
        f"### Question:\n{input_text}\n\n### Answer:\n"
        for input_text in examples["input_text"]
    ]
    full_texts = [
        prompt + output_text
        for prompt, output_text in zip(prompts, examples["output_text"])
    ]
    
    # Tokenize the batch of full texts
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors=None
    )
    
    # Labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main(args):
    # Define model name
    model_name = args.model_name
    logger.info("Loading dataset...")
    dataset = load_dataset("thesven/SyntheticMedicalQA-4336")
    dataset = dataset.rename_column("question", "input_text")
    dataset = dataset.rename_column("response", "output_text")
    
    # Split dataset into train & validation sets
    split_dataset = dataset["train"].train_test_split(test_size=0.2)
    
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Configuring 8-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_skip_modules=["lm_head"]
    )
    
    logger.info("Initializing base model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float32  # Use float32 for stability
    )
    
    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    logger.info("Configuring LoRA settings...")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    
    logger.info("Creating PEFT model...")
    model = get_peft_model(model, peft_config)
    
    # Ensure trainable parameters use float32
    logger.info("Ensuring trainable parameters are in float32...")
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    logger.info("Tokenizing the datasets...")
    tokenized_datasets = split_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=split_dataset["train"].column_names
    )
    
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,  # Reduced batch size
        gradient_accumulation_steps=16,  # Increased gradient accumulation
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=50,
        warmup_steps=100,
        gradient_checkpointing=True,
        fp16=False,  # Disable mixed precision for stability
        bf16=False,
        optim="adamw_torch",  # Use standard AdamW optimizer
        max_grad_norm=0.3,
        weight_decay=0.01,
        remove_unused_columns=False
    )
    
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"]
    )
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving the trained model...")
    model.save_pretrained(args.save_model_dir)
    logger.info(f"Model saved to: {args.save_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-Tune Mistral-7B on Synthetic Medical QA with LoRA"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="alecocc/mistral-8b-SFT-medqa-graph-cot",
        help="Name of the base model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mistral_medical_finetuned_lora",
        help="Directory for training outputs",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="./mistral_medical_finetuned_lora_final",
        help="Directory to save the final model",
    )
    args = parser.parse_args()
    main(args)