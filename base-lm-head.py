import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen3-1.7B"
model_name = "Qwen/Qwen3-4B-Instruct-2507"
options = ["A", "B", "C", "D"]

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# load and format the input as simple text
with open("input.json") as f:
    item = json.load(f)[0]

prompt = item["question"] + "\n"
for opt in options:
    prompt += f"{opt}. {item[opt]}\n"
prompt += "Answer:"

# prepare the model input
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
# [[12,3,4,,5,6,6,78], [12345678]]
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


# predict exactly one token's worth of logits
with torch.no_grad():
    logits = model(**model_inputs).logits[0, -1]

# mask off everything except the multiple choice option tokens
option_token_ids = {opt: tokenizer.encode(f" {opt}", add_special_tokens=False)[0] for opt in options}

mask = torch.full_like(logits, float("-inf"))
for tid in option_token_ids.values():
    mask[tid] = logits[tid]
probs = torch.softmax(mask, dim=-1)

# print the options with their probabilities
for opt in options:
    print(f"{opt}: {probs[option_token_ids[opt]].item():.4f}")
