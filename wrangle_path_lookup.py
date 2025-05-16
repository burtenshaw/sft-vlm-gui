from ast import literal_eval

import json

from datasets import DatasetDict, load_dataset, Dataset

# curl -X GET "https://huggingface.co/api/datasets/OpenGVLab/GUI-Odyssey?full=true" \
#   -H "Authorization: Bearer hf_xxx" \
#   -o /Users/ben/code/sft-vlm-gui/dataset_config.json

data = json.load(open("data/dataset_config.json"))["siblings"]

SCREENSHOT_PATHS = {}
DATASET_STUB = "https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey/resolve/main/"

for obj in data:
    if not obj["rfilename"].startswith("screenshot"):
        continue
    path = obj["rfilename"]
    url = f"{DATASET_STUB}{path}"
    name = path.split("/")[-1]
    SCREENSHOT_PATHS[name] = url

SYSTEM_PROMPT = open("system_prompt.md").read()

def path_to_url(path: str, idx: int = 0) -> str:
    """
    Convert a local file path to a URL.
    """
    return SCREENSHOT_PATHS.get(path, None)


def prepare_interleaved_dataset(dataset: Dataset, max_samples: int = 10) -> Dataset:
    """
    Prepare the dataset for training by interleaving the images and text.
    """

    wrangled_samples = []

    for n, sample in enumerate(dataset):
        if len(wrangled_samples) >= max_samples:
            break
        sample_messages = []
        steps = literal_eval(sample["steps"])
        if not isinstance(steps, list):
            print(f"Steps is not a list for sample {n}. Skipping...")
            continue
        for step in steps:
            image_path = step["screenshot"]
            image_url = path_to_url(image_path)
            if image_url is None:
                print(f"Image URL not found for {image_path}. Skipping...")
                sample_messages = []
                break
            step_string = json.dumps(step)
            sample_messages.append(
                {
                    "type": "image",
                    "image": image_url,
                }
            )
            sample_messages.append(
                {
                    "type": "text",
                    "text": step_string,
                }
            )
        if not sample_messages:
            print(f"No messages for sample {n}. Skipping...")
            continue
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": sample_messages,
            }
        ]
        new_sample = {**dict(sample)}
        new_sample["messages"] = json.dumps(messages)
        print(f"Adding sample {n}")
        wrangled_samples.append(new_sample)
    dataset = Dataset.from_list(wrangled_samples)
    return dataset


dataset = load_dataset(path="OpenGVLab/GUI-Odyssey", split="all", streaming=True)
dataset = prepare_interleaved_dataset(dataset, max_samples=10)
print(dataset[0])
dataset.save_to_disk("data/mini_dataset")