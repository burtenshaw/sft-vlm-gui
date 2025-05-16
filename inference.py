import json
import io
import os
import zipfile
import requests

import torch
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

messages = json.loads(sample["messages"])

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
# Save the model