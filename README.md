# Fine-Tuning Qwen3-4B-Instruct for Construction Industry RAG App

This guide explains how to fine-tune the Qwen3-4B-Instruct model for a Retrieval-Augmented Generation (RAG) app focused on the construction industry, using Unsloth on an Ubuntu server with an NVIDIA RTX 3060 (12GB VRAM). It’s written for beginners with no fine-tuning experience, covering how to create a construction dataset, fine-tune the model, and test it. The dataset will include categories like safety, materials, regulations, equipment, project management, and sustainability.

## Prerequisites
- **Hardware**: Intel i7 4th gen processor, NVIDIA RTX 3060 with 12GB VRAM, 16GB RAM.
- **Operating System**: Ubuntu-based system accessed via SSH.
- **Software**: NVIDIA drivers and CUDA (version 11.8 or 12.1) must be installed.
- **Internet**: Needed to download the model and datasets.
- **Environment**: A Python virtual environment named `unsloth_env` with Python 3.11.

## Step 1: Check Your System
1. **Verify GPU and CUDA**:
   - Run the command `nvidia-smi` in your terminal.
   - You should see a table showing your RTX 3060, CUDA version (like 12.8), and available VRAM (~11.6GB).
   - If it fails (e.g., says "command not found"), ask your system admin to install NVIDIA drivers and CUDA.

2. **Check Python Version**:
   - Run `python --version`.
   - It should show Python 3.10, 3.11, or 3.12 (e.g., Python 3.11.0). If not, install Python 3.11:
     - Update package lists: `sudo apt update`.
     - Install Python: `sudo apt install python3.11 python3.11-venv python3-pip`.

3. **Activate Virtual Environment**:
   - Ensure you’re in the `unsloth_env` environment: `source ~/unsloth_env/bin/activate`.
   - Your terminal prompt should show `(unsloth_env)`.

4. **Test Internet Access**:
   - Run `ping huggingface.co`.
   - You should see responses like "64 bytes from ...". Press Ctrl+C to stop.
   - If it fails, check your network settings or contact your system admin.

## Step 2: Install Necessary Software
- Install Unsloth and other required libraries in the virtual environment:
  - Run: `pip install unsloth transformers datasets trl peft bitsandbytes accelerate`.
- If you see errors about CUDA compatibility, install a specific Unsloth version:
  - Run: `pip install "unsloth[cu121-torch240]" -U`.

## Step 3: Create a Construction Industry Dataset
You’ll create a dataset with prompt-response pairs for construction topics like safety, materials, regulations, equipment, project management, and sustainability. The dataset will be a JSONL file (one JSON object per line).

1. **Make a Dataset Folder**:
   - Run: `mkdir ~/construction_dataset`.
   - Move to the folder: `cd ~/construction_dataset`.

2. **Create the Dataset File**:
   - Open a text editor: `nano construction_data.jsonl`.
   - Write entries in this format: each line is a JSON object with a "text" field containing "Prompt: [your question or instruction]\nResponse: [the answer]".
   - Example entries to start (add 20-50 more for your categories):
     - Safety: Prompt asking about heavy machinery safety protocols, with a response listing certification, inspections, and protective equipment.
     - Materials: Prompt about properties of reinforced concrete, with a response explaining its strength and durability.
     - Regulations: Prompt about waste management rules, with a response mentioning EPA guidelines.
     - Foundation: Prompt about calculating load-bearing capacity, with a response detailing soil tests and formulas.
     - Sustainability: Prompt about sustainable materials, with a response suggesting recycled steel or bamboo.
     - Project Management: Prompt about project management steps, with a response outlining scope, timeline, and software use.
     - Equipment: Prompt about crane maintenance, with a response listing inspection and certification steps.
   - Add entries for your specific categories (e.g., worker training, building codes). Aim for 50-100 total entries.
   - Save: Press Ctrl+O, Enter, Ctrl+X.

3. **Check the Dataset**:
   - Run: `ls` to confirm `construction_data.jsonl` exists.
   - Run: `head -n 5 construction_data.jsonl` to view the first 5 lines.
   - If there’s a JSON error, edit the file again to fix formatting.

## Step 4: Set Up the Fine-Tuning Script
1. **Go to the Finetune Folder**:
   - Run: `cd ~/finetune`.

2. **Create or Edit the Fine-Tuning Script**:
   - Open: `nano finetune.py`.
   - Write a script that:
     - Loads the Qwen3-4B-Instruct model from Hugging Face (`unsloth/Qwen3-4B-unsloth-bnb-4bit`).
     - Applies LoRA adapters for efficient fine-tuning.
     - Loads your dataset from `/home/shakespear/construction_dataset/construction_data.jsonl`.
     - Trains for 60 steps (takes 10-30 minutes).
     - Saves the model to `fine_tuned_qwen3_4b`.
   - Save: Ctrl+O, Enter, Ctrl+X.

## Step 5: Run Fine-Tuning
- Run: `python finetune.py`.
- This will:
  - Download the model (~2.56GB).
  - Load your dataset.
  - Train for 60 steps.
  - Save the fine-tuned model to `fine_tuned_qwen3_4b`.
- If errors occur:
  - **Memory error**: Check VRAM with `nvidia-smi`. Edit `finetune.py` to reduce batch size (e.g., from 2 to 1) and re-run.
  - **Dataset error**: Verify the dataset path.
  - **Model download error**: Manually download with `huggingface-cli download unsloth/Qwen3-4B-unsloth-bnb-4bit --local-dir qwen3_model`, then update `finetune.py` to use `./qwen3_model`.

- Check the output: `ls fine_tuned_qwen3_4b`. Expect files like `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`.

## Step 6: Test the Fine-Tuned Model
1. **Create a Test Script**:
   - Open: `nano test.py`.
   - Write a script that:
     - Loads the fine-tuned model from `fine_tuned_qwen3_4b`.
     - Uses a construction-related prompt (e.g., asking about sustainable materials).
     - Generates a response (up to 200 tokens).
   - Save: Ctrl+O, Enter, Ctrl+X.

2. **Run the Test**:
   - Run: `python test.py`.
   - Check the output for a construction-related response.
   - If it fails, verify the model path or share the error.

## Step 7: Plan for RAG
- This fine-tuning prepares the model to answer construction questions. For a RAG app, you’ll need a retriever to fetch relevant documents (e.g., from safety manuals or building codes).
- Share details about your data sources or categories to set up a retriever.

## Troubleshooting
- **Model Files**: Keep large files (`fine_tuned_qwen3_4b`, `qwen3_model`) local or upload to Hugging Face.
- **Errors**: Share any errors in issues from `finetune.py` or `test.py` for help.
