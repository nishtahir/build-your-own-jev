import argparse
import json
import random

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
options = ["A", "B", "C", "D", "E"]

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="dataset/train.jsonl")
parser.add_argument("--output", default="checkpoint")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--learning-rate", type=float, default=1e-4)
args = parser.parse_args()

# load the tokenizer and the model
# left padding so the last position of every row is a real token
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# untie the lm head from the input embeddings: give it its own copy of the
# weights so training it leaves the embeddings untouched
model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
model.config.tie_word_embeddings = False

# freeze the backbone - only the lm head gets trained
model.requires_grad_(False)
model.lm_head.requires_grad_(True)

optimizer = torch.optim.Adam(model.lm_head.parameters(), lr=args.learning_rate)
loss_fn = nn.CrossEntropyLoss()

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


def option_logits(items):
    texts = [format_prompt(item) for item in items]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    # the frozen backbone needs no gradients, only the lm head does
    with torch.no_grad():
        hidden = model.model(**model_inputs).last_hidden_state[:, -1]
    # constrained to the option tokens, matching constrained decoding at eval time
    return model.lm_head(hidden)[:, option_token_ids]


with open(args.data) as f:
    data = [json.loads(line) for line in f]
if args.limit is not None:
    data = data[: args.limit]

random.seed(0)
for epoch in range(args.epochs):
    random.shuffle(data)
    total_loss = 0.0
    progress = tqdm(
        range(0, len(data), args.batch_size), desc=f"epoch {epoch}", unit="batch"
    )
    for start in progress:
        batch = data[start : start + args.batch_size]
        labels = torch.tensor(
            [options.index(item["answer"]) for item in batch], device=model.device
        )

        loss = loss_fn(option_logits(batch).float(), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    print(f"epoch {epoch}: avg train loss {total_loss / len(data):.4f}")

model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)
print(f"saved checkpoint to {args.output}")
