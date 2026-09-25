import json

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Instruct-2507"
options = ["A", "B", "C", "D"]
epochs = 3
learning_rate = 1e-3

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# replace the LM head with a fresh, untrained classification head: one
# output feature per multiple choice option instead of one per vocab token
hidden_size = model.config.hidden_size
model.lm_head = nn.Linear(hidden_size, len(options), bias=False).to(model.device, model.dtype)
print("lm_head params:", model.lm_head.weight.numel())

# freeze the backbone - only the new head gets trained
model.requires_grad_(False)
model.lm_head.requires_grad_(True)

optimizer = torch.optim.Adam(model.lm_head.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


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


def predict_logits(item):
    text = format_prompt(item)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    return model(**model_inputs).logits[0, -1]


# load the labeled dataset and split 80:20 into train/test
with open("training.json") as f:
    data = json.load(f)

split = int(len(data) * 0.8)
train_data = data[:split]
test_data = data[split:]
print(f"train: {len(train_data)}, test: {len(test_data)}")

# small training loop: one example at a time, backbone frozen
for epoch in range(epochs):
    total_loss = 0.0
    for item in train_data:
        label = torch.tensor(options.index(item["answer"]), device=model.device)

        logits = predict_logits(item)
        loss = loss_fn(logits.float().unsqueeze(0), label.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"epoch {epoch}: avg train loss {total_loss / len(train_data):.4f}")

# evaluate accuracy on the held-out test split
correct = 0
with torch.no_grad():
    for item in test_data:
        logits = predict_logits(item)
        predicted = options[logits.argmax().item()]
        correct += predicted == item["answer"]
print(f"test accuracy: {correct}/{len(test_data)}")

# run inference over input.json with the trained head
with open("input.json") as f:
    inference_data = json.load(f)

with torch.no_grad():
    for item in inference_data:
        logits = predict_logits(item)
        probs = torch.softmax(logits.float(), dim=-1)
        print(item["question"])
        for opt, prob in zip(options, probs.tolist()):
            print(f"  {opt}: {prob:.4f}")
