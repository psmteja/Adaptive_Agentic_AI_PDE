"""
Step 9: Run Qwen on one of our FNO-planner prompts.

- Reads the first example from:
      llm_data/fno_planner_train_multi.jsonl
- Sends the `prompt` to Qwen/Qwen2.5-7B-Instruct
- Prints the model's response

This is pre-fine-tuning: we're just checking that
Qwen can talk about FNO configs on our prompts.
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


LLM_DATA_FILE = "llm_data/fno_planner_train_multi.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # you can change to another Qwen model if desired


def load_first_prompt(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"LLM data file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
        if not line:
            raise ValueError("LLM data file is empty.")
        obj = json.loads(line)

    prompt = obj["prompt"]
    print("[Step9] Loaded first prompt from LLM data.")
    return prompt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Step9] Using device: {device}")

    # 1) Load prompt from our generated training data
    prompt = load_first_prompt(LLM_DATA_FILE)
    print("\n[Step9] Prompt (truncated):")
    print("--------------------------------------------------")
    print(prompt[:600] + ("..." if len(prompt) > 600 else ""))
    print("--------------------------------------------------\n")

    # 2) Load Qwen model + tokenizer
    print(f"[Step9] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # 3) Build chat-style input (Qwen uses chat template in HF)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant that designs energy-efficient Fourier Neural "
                "Operators (FNOs) for PDE surrogate modeling. Follow the user's instructions "
                "and output clear reasoning and a JSON config when requested."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Use the tokenizer's chat template to build the input string
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # 4) Generate
    print("[Step9] Generating response from Qwen...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # deterministic for now
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # The chat template usually includes the prompt; we just print everything for now
    print("\n[Step9] Full Qwen output:")
    print("==================================================")
    print(full_text)
    print("==================================================")

    print("\n[Step9] Done ✅")


if __name__ == "__main__":
    main()
