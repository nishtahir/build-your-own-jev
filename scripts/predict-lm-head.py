import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
options = ["A", "B", "C", "D", "E"]

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoint")
parser.add_argument("--input", default="input.json")
args = parser.parse_args()

# the tokenizer is shared, the finetune only changes the lm head
tokenizer = AutoTokenizer.from_pretrained(model_name)


def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype="auto", device_map="auto"
    )


# each entry is (name, model, temperature)
predictors = [("base", load_model(model_name), 1.0)]

checkpoint = Path(args.checkpoint)
if checkpoint.exists():
    finetuned = load_model(checkpoint)
    predictors.append(("finetuned", finetuned, 1.0))

    # temperature fit by calibrate-lm-head.py
    temperature_path = checkpoint / "temperature.json"
    if temperature_path.exists():
        temperature = json.loads(temperature_path.read_text())["temperature"]
        predictors.append(("calibrated", finetuned, temperature))

# the token the model would emit for each option as the first assistant token
option_token_ids = [
    tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options
]


def format_prompt(item):
    prompt = item["question"] + "\n"
    for opt in options:
        prompt += f"{opt}. {item[opt]}\n"
    prompt += "Answer:"
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def predict(model, temperature, item):
    model_inputs = tokenizer([format_prompt(item)], return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        logits = model(**model_inputs).logits[0, -1]
    # constrained decoding: only the option tokens are allowed
    option_logits = logits[option_token_ids].float()
    return torch.softmax(option_logits / temperature, dim=-1)


with open(args.input) as f:
    data = json.load(f)

for item in data:
    print(item["question"])
    for name, model, temperature in predictors:
        probs = predict(model, temperature, item)
        best = probs.argmax().item()
        print(f"{name}: {options[best]}. {item[options[best]]}")
        for opt, prob in zip(options, probs.tolist()):
            print(f"  {opt}: {prob:.4f}  {item[opt]}")
