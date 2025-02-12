# Fine-Tuning Mistral-7B on Synthetic Medical QA with LoRA

## Overview
This repository contains a script to fine-tune the [Mistral-7B](https://huggingface.co/alecocc/mistral-7B-SFT-medqa-graph-cot) model on the [SyntheticMedicalQA-4336](https://huggingface.co/datasets/thesven/SyntheticMedicalQA-4336) dataset using LoRA (Low-Rank Adaptation). The model is trained with 8-bit quantization for memory efficiency and leverages the PEFT (Parameter Efficient Fine-Tuning) library for optimizing training.

## Features
- Uses **8-bit quantization** to reduce memory consumption
- Fine-tunes the **Mistral-7B** model for **medical question answering**
- Implements **LoRA** to improve fine-tuning efficiency
- Includes **gradient checkpointing** and **gradient accumulation** for optimized training
- Utilizes **AdamW optimizer** for stable updates

## Requirements
Make sure you have the required libraries installed:

```bash
pip install datasets transformers peft torch bitsandbytes
```

## Dataset
The script uses the **SyntheticMedicalQA-4336** dataset:
- Questions are mapped to `input_text`
- Responses are mapped to `output_text`
- The dataset is split into **80% training** and **20% validation**

## Model & Training Configuration
### **Model Name**
- **Base Model**: `alecocc/mistral-7B-SFT-medqa-graph-cot`
- **Tokenizer**: Uses **AutoTokenizer** with EOS token padding

### **Quantization**
- Uses **8-bit quantization** with `BitsAndBytesConfig`
- Enables **FP32 CPU offload** for stability

### **LoRA Configuration**
- Applies LoRA on attention projection layers (`q_proj, k_proj, v_proj, etc.`)
- **Rank (r):** 16, **Alpha:** 32, **Dropout:** 0.1

### **Training Arguments**
- **Batch Size:** 1 (with gradient accumulation of 16 steps)
- **Learning Rate:** `1e-4`
- **Epochs:** 3
- **Evaluation Steps:** 50
- **Gradient Checkpointing:** Enabled
- **Optimizer:** AdamW
- **Mixed Precision (FP16/BF16):** Disabled for stability

## How to Use
### **1. Load and Preprocess Dataset**
```python
from datasets import load_dataset

dataset = load_dataset("thesven/SyntheticMedicalQA-4336")
dataset = dataset.rename_column("question", "input_text")
dataset = dataset.rename_column("response", "output_text")
split_dataset = dataset["train"].train_test_split(test_size=0.2)
```

### **2. Initialize Tokenizer & Quantized Model**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

model_name = "alecocc/mistral-7B-SFT-medqa-graph-cot"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_7Bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=["lm_head"]
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float32
)
```

### **3. Apply LoRA Fine-Tuning**
```python
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, peft_config)
```

### **4. Tokenization & Data Preparation**
```python
def tokenize_function(examples):
    prompts = [f"### Question:\n{input_text}\n\n### Answer:\n" for input_text in examples["input_text"]]
    full_texts = [prompt + output_text for prompt, output_text in zip(prompts, examples["output_text"])]
    tokenized = tokenizer(full_texts, truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = split_dataset.map(tokenize_function, batched=True, remove_columns=split_dataset["train"].column_names)
```

### **5. Set Training Parameters & Train Model**
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./mistral_medical_finetuned_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=50,
    warmup_steps=100,
    gradient_checkpointing=True,
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    max_grad_norm=0.3,
    weight_decay=0.01,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()
```

### **6. Save the Trained Model**
```python
model.save_pretrained("./mistral_medical_finetuned_lora_final")
```

## Results & Expected Outcomes
- The fine-tuned model will generate improved medical responses.
- The use of **LoRA** ensures efficient fine-tuning without modifying the entire model.
- The trained model can be further tested and deployed for medical question answering tasks.



