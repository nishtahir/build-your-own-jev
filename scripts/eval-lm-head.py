import argparse
import json
from typing import cast

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

model_name = "Qwen/Qwen3-1.7B"
options = ["A", "B", "C", "D", "E"]

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="dataset/validation.jsonl")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

# load the tokenizer and the model
# left padding so the last position of every row is a real token
tokenizer = cast(
    PreTrainedTokenizerBase,
    AutoTokenizer.from_pretrained(snapshot_download(model_name), padding_side="left"),
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

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
    data = data[: args.limit]

# sort by length so each batch has minimal padding
data.sort(key=lambda item: len(format_prompt(item)))

predictions = []
labels = []
progress = tqdm(total=len(data), unit="answer")
for start in range(0, len(data), args.batch_size):
    batch = data[start : start + args.batch_size]
    predictions += predict(batch)
    labels += [item["answer"] for item in batch]
    progress.update(len(batch))
    progress.set_postfix(
        accuracy=f"{sum(p == l for p, l in zip(predictions, labels)) / len(labels):.4f}"
    )
progress.close()

print(f"{'':>8}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
f1s = []
for opt in options:
    tp = sum(p == opt and l == opt for p, l in zip(predictions, labels))
    predicted = predictions.count(opt)
    support = labels.count(opt)
    precision = tp / predicted if predicted else 0.0
    recall = tp / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f1s.append(f1)
    print(f"{opt:>8}{precision:>10.4f}{recall:>10.4f}{f1:>10.4f}{support:>10}")

correct = sum(p == l for p, l in zip(predictions, labels))
print(f"accuracy: {correct}/{len(labels)} = {correct / len(labels):.4f}")
print(f"macro f1: {sum(f1s) / len(f1s):.4f}")
