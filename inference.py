import json
import io
import os
import zipfile
import requests
import tempfile

import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# For multi-image example
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    url = image.get("text")
                    response = requests.get(url=url)
                    image = Image.open(io.BytesIO(response.content))
                    image_inputs.append(image.convert("RGB"))
    return image_inputs


def process_and_plot_images(content_messages: list[dict]) -> dict[int, str]:
    """
    Loads images from content_messages, creates single image plots with normalized labels,
    saves them as temporary image files, and returns a list of their paths.
    """
    tmp_image_paths = {}

    for i, content in enumerate(content_messages):
        if isinstance(content, dict) and "image" in content:
            url = content.get("image")
            response = requests.get(url=url)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            # Normalize label (assuming label is present in content)
            label = content.get("label", 0)  # Default to 0 if no label
            normalized_label = min(max(label, 0), 1000)  # Clamp between 0 and 1000

            # Create a single image plot
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(np.array(image))
            ax.set_title(f"Label: {normalized_label}")
            ax.axis("off")

            # Save the plot to a temporary file
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(tmp_file.name, bbox_inches="tight")
            plt.close(fig)

            tmp_image_paths[i] = tmp_file.name

    return tmp_image_paths


model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path="google/gemma-3-4b-it", 
    trust_remote_code=True
)
processor.tokenizer.padding_side = "right"
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path="google/gemma-3-4b-it",
    trust_remote_code=True,
    **model_kwargs
)

dataset = load_from_disk(dataset_path="data/mini_dataset")

sample = dataset[0]

print("===")

messages = json.loads(sample["messages"])

content_messages = messages[-1]["content"].copy()
tmp_image_paths = process_and_plot_images(content_messages)
for i, message in enumerate(messages):
    if "image" in content_messages[i]:
        content_messages[i]["image"] = tmp_image_paths.get(i)
    
for i in range(len(content_messages)):
    
    print(f"=== {i} ===")
    last_message = content_messages.pop(-1)
    last_screenshot = content_messages[-1]
    messages[-1]["content"] = content_messages
    print(last_message)
    print(last_screenshot)

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    
    content_messages.pop(-1)
    
    print(decoded)

    print("=== - ===")

# Example usage
tmp_image_paths = process_and_plot_images(content_messages)
print("Temporary image paths:", tmp_image_paths)
