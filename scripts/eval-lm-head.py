import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
options = ["A", "B", "C", "D", "E"]

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="dataset/validation.jsonl")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

# load the tokenizer and the model
# left padding so the last position of every row is a real token
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# the token the model would emit for each option as the first assistant token
option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]


def format_prompt(item):
    prompt = item["question"] + "\n"
    for opt in options:
        prompt += f"{opt}. {item[opt]}\n"
    prompt += "Answer:"
    messages = [
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


def predict(items):
    texts = [format_prompt(item) for item in items]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**model_inputs).logits[:, -1]
    # constrained decoding: only the option tokens are allowed
    choices = logits[:, option_token_ids].argmax(dim=-1).tolist()
    return [options[c] for c in choices]


with open(args.data) as f:
    data = [json.loads(line) for line in f]
if args.limit is not None:
    data = data[:args.limit]

# sort by length so each batch has minimal padding
data.sort(key=lambda item: len(format_prompt(item)))

correct = 0
seen = 0
progress = tqdm(total=len(data), unit="answer")
for start in range(0, len(data), args.batch_size):
    batch = data[start:start + args.batch_size]
    correct += sum(p == item["answer"] for p, item in zip(predict(batch), batch))
    seen += len(batch)
    progress.update(len(batch))
    progress.set_postfix(accuracy=f"{correct / seen:.4f}")
progress.close()

print(f"accuracy: {correct}/{len(data)} = {correct / len(data):.4f}")
