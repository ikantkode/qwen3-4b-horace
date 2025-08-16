from unsloth import FastLanguageModel
import torch

# Load the fine-tuned model from the local folder
model, tokenizer = FastLanguageModel.from_pretrained(
    "./fine_tuned_qwen3_4b",  # Local folder path
    load_in_4bit=True  # Same 4-bit quantization as fine-tuning
)

# Prepare the model for generating responses
FastLanguageModel.for_inference(model)

# Test with a RAG-style prompt (context + question)
prompt = """Context: University life in the UK involves attending lectures, seminars, and tutorials, living in halls or private accommodation, and managing independent study. Freshers' week is a time to socialize and join student societies.
Question: What should I expect during my first week at a UK university?"""
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.8)
print(tokenizer.decode(outputs[0]))
