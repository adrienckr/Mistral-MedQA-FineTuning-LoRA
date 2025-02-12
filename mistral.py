from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
from transformers import BitsAndBytesConfig

# Define model name
model_name = "alecocc/mistral-8b-SFT-medqa-graph-cot"

# Load dataset and rename columns
dataset = load_dataset("thesven/SyntheticMedicalQA-4336")
dataset = dataset.rename_column("question", "input_text")
dataset = dataset.rename_column("response", "output_text")

# Split into train & validation
split_dataset = dataset["train"].train_test_split(test_size=0.2)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=["lm_head"]
)

# Initialize base model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float32  # Use float32 for stability
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
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

# Create PEFT model
model = get_peft_model(model, peft_config)

# Enable gradient computation
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

def tokenize_function(examples):
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

# Apply tokenization with batched processing
tokenized_datasets = split_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=split_dataset["train"].column_names
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./mistral_medical_finetuned_lora",
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
    fp16=False,  # Disable mixed precision
    bf16=False,
    optim="adamw_torch",  # Use standard AdamW optimizer
    max_grad_norm=0.3,
    weight_decay=0.01,
    remove_unused_columns=False
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# Print trainable parameters
model.print_trainable_parameters()

# Start training
trainer.train()

# Save the trained model
model.save_pretrained("./mistral_medical_finetuned_lora_final")