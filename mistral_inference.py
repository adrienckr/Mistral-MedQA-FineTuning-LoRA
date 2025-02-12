import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 100) -> str:
    input_text = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.replace(input_text, "").strip()
    return answer

def main(args):
    logger.info("Loading fine-tuned model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.save_model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.save_model_dir)
    model.eval()
    logger.info("Model loaded successfully. Ready for inference.")
    while True:
        question = input("Enter your question (or type 'quit' to exit): ")
        if question.lower().strip() == "quit":
            break
        answer = generate_answer(model, tokenizer, question, max_new_tokens=args.max_new_tokens)
        print("\nAnswer:", answer, "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for the fine-tuned Mistral-7B model with LoRA"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="alecocc/mistral-8b-SFT-medqa-graph-cot",
        help="Name of the base model"
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="./mistral_medical_finetuned_lora_final",
        help="Directory where the fine-tuned model is saved"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate for each answer"
    )
    args = parser.parse_args()
    main(args)
