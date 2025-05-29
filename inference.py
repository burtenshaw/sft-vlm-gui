import json
import io
import requests
import tempfile

import pandas as pd
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
    adds horizontal grid lines, axis labels, and ticks with labels from 0 to 1000 in intervals of 100,
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
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            
            # Add grid lines
            ax.grid(visible=True, which='major', axis='both', linestyle='--', linewidth=0.5, color='gray')
            
            # Set ticks and labels
            ax.set_xticks(np.linspace(0, image.size[0], 11))  # 11 ticks for 0 to 1000
            ax.set_yticks(np.linspace(0, image.size[1], 11))
            ax.set_xticklabels(range(0, 1001, 100), rotation=45)
            ax.set_yticklabels(range(0, 1001, 100), rotation=45)
            
            ax.axis("on")  # Keep axes visible

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



def strip_generation(generation: str) -> str:
    """ strip the generation down to just the json """
    generation = generation.split("```json")[1]
    generation = generation.split("```")[0]
    try:
        generation = json.loads(generation)
    except json.JSONDecodeError:
        print(f"Error parsing JSON: {generation}")
        return None
    return generation
    
def evaluate_gui_response(model_response, expected_response=None):
    """
    Evaluates a GUI agent response based on the system prompt requirements.
    
    Args:
        model_response (str): The response from the model, expected to be a JSON string
        expected_response (dict, optional): Ground truth expected response for comparison
        
    Returns:
        dict: Evaluation results containing format_correct, action_valid, and score
    """
    try:
        # Parse the model response
        if isinstance(model_response, str):
            try:
                parsed_response = json.loads(model_response)
            except json.JSONDecodeError:
                # Try to extract JSON if wrapped in markdown code blocks
                if "```json" in model_response:
                    model_response = model_response.split("```json")[1].split("```")[0].strip()
                    parsed_response = json.loads(model_response)
                else:
                    return {"format_correct": False, "error": "Invalid JSON format", "score": 0}
        else:
            parsed_response = model_response
        
        # Check required fields
        required_fields = ["action", "info", "ps"]
        missing_fields = [field for field in required_fields if field not in parsed_response]
        if missing_fields:
            return {"format_correct": False, "error": f"Missing fields: {missing_fields}", "score": 0}
        
        # Validate action type
        valid_actions = ["CLICK", "SCROLL", "LONG_PRESS", "TYPE", "COMPLETE", "IMPOSSIBLE", "HOME", "BACK"]
        if parsed_response["action"] not in valid_actions:
            return {"format_correct": False, "error": f"Invalid action: {parsed_response['action']}", "score": 0}
        
        # Validate info format based on action
        action = parsed_response["action"]
        info = parsed_response["info"]
        
        if action in ["CLICK", "LONG_PRESS"]:
            if not (isinstance(info, list) and len(info) == 2 and all(isinstance(coord, (int, float)) for coord in info)):
                if not (isinstance(info, str) and info in ["KEY_HOME", "KEY_BACK", "KEY_RECENT"]):
                    return {"format_correct": False, "error": "Invalid coordinates format for CLICK/LONG_PRESS", "score": 0}
        elif action == "SCROLL":
            if not (isinstance(info, list) and len(info) == 2 and 
                    all(isinstance(coord_pair, list) and len(coord_pair) == 2 for coord_pair in info)):
                return {"format_correct": False, "error": "Invalid coordinates format for SCROLL", "score": 0}
        elif action == "TYPE":
            if not (isinstance(info, list) and len(info) == 1 and isinstance(info[0], str)):
                return {"format_correct": False, "error": "Invalid text format for TYPE", "score": 0}
        else:  # COMPLETE, IMPOSSIBLE, HOME, BACK
            if info != "":
                return {"format_correct": False, "error": f"Info should be empty for {action}", "score": 0}
        
        # Calculate score
        score = 1.0
        
        # If we have expected response, compare with it
        if expected_response:
            action_match = parsed_response["action"] == expected_response["action"]
            
            if not action_match:
                score *= 0.5
            
            # For actions with coordinates, check proximity rather than exact match
            if action in ["CLICK", "LONG_PRESS"] and action_match:
                if isinstance(info, list) and isinstance(expected_response["info"], list):
                    distance = sum((a - b)**2 for a, b in zip(info, expected_response["info"]))**0.5
                    proximity_score = max(0, 1 - distance/1000)  # Normalize by max coordinate value
                    score *= proximity_score
            
        return {
            "format_correct": True, 
            "action_valid": True,
            "score": score
        }
    
    except Exception as e:
        return {"format_correct": False, "error": str(e), "score": 0}

dataset = load_from_disk(dataset_path="data/mini_dataset")

MAX_SAMPLES = 10

results = []

for sample_i, sample in enumerate(dataset):

    print(f"=== SAMPLE {sample_i} ===")

    messages = json.loads(sample["messages"])

    content_messages = messages[-1]["content"].copy()
    tmp_image_paths = process_and_plot_images(content_messages)

    for i, message in enumerate(content_messages):
        if "image" in content_messages[i]:
            content_messages[i]["image"] = tmp_image_paths.get(i)
            

    for i in range(len(content_messages)):
        if len(content_messages) <= 1:
            break
        print(f"=== STEP {i} ===")
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
        generation = strip_generation(decoded)
        evaluation = evaluate_gui_response(generation, last_message)
        print(evaluation)
        print(generation)

        result = {**evaluation, "generation": generation, "step": i, "sample_id": sample_i, "last_message": last_message}

        results.append(result)

        content_messages.pop(-1)
        print("=== - ===")

df = pd.DataFrame(results)
df.to_json("results.json", orient="records")