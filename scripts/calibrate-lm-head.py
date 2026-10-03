import argparse
import json

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

options = ["A", "B", "C", "D", "E"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="checkpoint")
parser.add_argument("--data", default="dataset/validation.jsonl")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

# load the tokenizer and the model
# left padding so the last position of every row is a real token
tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype="auto", device_map="auto"
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


def option_logits(items):
    texts = [format_prompt(item) for item in items]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**model_inputs).logits[:, -1]
    # constrained decoding: only the option tokens are allowed
    return logits[:, option_token_ids].float().cpu()


def expected_calibration_error(probs, labels, bins=15):
    confidences, predictions = probs.max(dim=-1)
    correct = (predictions == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            gap = (confidences[in_bin].mean() - correct[in_bin].mean()).abs()
            ece += in_bin.float().mean() * gap
    return float(ece)


with open(args.data) as f:
    data = [json.loads(line) for line in f]
if args.limit is not None:
    data = data[: args.limit]

# sort by length so each batch has minimal padding
data.sort(key=lambda item: len(format_prompt(item)))

# collect the option logits once, temperature is fit on these afterwards
all_logits = []
for start in tqdm(range(0, len(data), args.batch_size), unit="batch"):
    all_logits.append(option_logits(data[start : start + args.batch_size]))
logits = torch.cat(all_logits)
labels = torch.tensor([options.index(item["answer"]) for item in data])

# fit a single scalar temperature by minimizing NLL, parameterized as
# log(T) so that T stays positive
loss_fn = nn.CrossEntropyLoss()
log_temperature = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=200)


def closure():
    optimizer.zero_grad()
    loss = loss_fn(logits / log_temperature.exp(), labels)
    loss.backward()
    return loss


optimizer.step(closure)
temperature = log_temperature.exp().item()

# dividing logits by a positive scalar never changes the argmax, so accuracy
# is untouched - only the confidence of the probabilities moves
accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
print(f"accuracy: {accuracy:.4f}")
for name, t in [("before", 1.0), ("after", temperature)]:
    scaled = logits / t
    probs = torch.softmax(scaled, dim=-1)
    nll = loss_fn(scaled, labels).item()
    ece = expected_calibration_error(probs, labels)
    print(f"{name}: temperature {t:.4f}, NLL {nll:.4f}, ECE {ece:.4f}")

output = f"{args.model}/temperature.json"
with open(output, "w") as f:
    json.dump({"temperature": temperature}, f)
print(f"saved temperature to {output}")
