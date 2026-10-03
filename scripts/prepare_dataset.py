import json
from pathlib import Path

from datasets import load_dataset

DATASET_DIR = Path(__file__).parent.parent / "dataset"


def format_example(example: dict) -> dict:
    row = {"question": example["question"]}
    for label, text in zip(example["choices"]["label"], example["choices"]["text"]):
        row[label] = text
    row["answer"] = example["answerKey"]
    return row


def main() -> None:
    DATASET_DIR.mkdir(exist_ok=True)
    splits = load_dataset("tau/commonsense_qa", cache_dir=DATASET_DIR / "cache")

    for split_name, split in splits.items():
        path = DATASET_DIR / f"{split_name}.jsonl"
        with path.open("w") as f:
            for example in split:
                f.write(json.dumps(format_example(example)) + "\n")
        print(f"Wrote {len(split)} rows to {path}")


if __name__ == "__main__":
    main()
