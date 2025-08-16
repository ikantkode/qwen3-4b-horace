from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Set maximum sequence length (suitable for RAG tasks)
max_seq_length = 2048

# Load the quantized Qwen3-4B-Instruct model from Hugging Face, it will auto-download
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,  # Auto-detect float16/bfloat16
    load_in_4bit=True  # Fits 12GB VRAM
)

# Apply LoRA adapters for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

# Load a sample dataset for testing
dataset = load_dataset("squad", split="train")

# Format dataset for RAG
def format_rag_example(example):
    return {
        "text": f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer: {example['answers']['text'][0]}"
    }

dataset = dataset.map(format_rag_example)

# Set up the trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Short for testing
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs"
    )
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_qwen3_4b")
tokenizer.save_pretrained("fine_tuned_qwen3_4b")
