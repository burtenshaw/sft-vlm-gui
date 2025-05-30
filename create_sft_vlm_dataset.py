#!/usr/bin/env python3
"""
Create a TRL SFT-compatible dataset for VLM training from GUI-Odyssey dataset.

This script loads the GUI-Odyssey dataset, processes it to create interleaved image-text
conversations, downloads images as PIL objects, and converts to TRL SFT format.

Example usage:
python create_sft_vlm_dataset.py \
    --dataset_name OpenGVLab/GUI-Odyssey \
    --output_dataset_path data/sft_vlm_dataset \
    --max_samples 10 \
    --max_workers 4
"""

import io
import json
import requests
import os
import hashlib
from urllib.parse import urlparse
import time
import argparse
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from ast import literal_eval

import torch
from datasets import Dataset, DatasetDict, Features, Value, Sequence, load_from_disk, load_dataset, Image as DatasetImage
from PIL import Image

# Global cache for images
IMAGE_CACHE = {}
CACHE_DIR = "image_cache"

def get_cache_path(url: str) -> str:
    """Generate a cache file path for a given URL."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Create a hash of the URL for the filename
    url_hash = hashlib.md5(url.encode()).hexdigest()
    parsed_url = urlparse(url)
    extension = os.path.splitext(parsed_url.path)[1] or '.png'
    
    return os.path.join(CACHE_DIR, f"{url_hash}{extension}")

def download_image_with_retry(url: str, max_retries: int = 3, timeout: int = 10) -> Image.Image:
    """Download image with retry logic and caching."""
    # Check memory cache first
    if url in IMAGE_CACHE:
        return IMAGE_CACHE[url]
    
    # Check disk cache
    cache_path = get_cache_path(url)
    if os.path.exists(cache_path):
        try:
            image = Image.open(cache_path).convert("RGB")
            IMAGE_CACHE[url] = image
            return image
        except Exception as e:
            print(f"Error loading cached image {cache_path}: {e}")
            # Remove corrupted cache file
            try:
                os.remove(cache_path)
            except:
                pass
    
    # Download with retry logic
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Load image
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            # Cache to disk
            try:
                image.save(cache_path)
            except Exception as e:
                print(f"Warning: Could not cache image to disk: {e}")
            
            # Cache in memory
            IMAGE_CACHE[url] = image
            return image
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt == max_retries - 1:
                # Return a small placeholder image on final failure
                print(f"Failed to download {url} after {max_retries} attempts, using placeholder")
                placeholder = Image.new("RGB", (224, 224), color=(128, 128, 128))
                return placeholder
            
            # Wait before retry
            time.sleep(1)
    
    # This shouldn't be reached, but just in case
    return Image.new("RGB", (224, 224), color=(128, 128, 128))

def load_dataset_config(config_path: str = "data/dataset_config.json") -> Dict[str, str]:
    """Load the dataset configuration to create URL mappings."""
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        screenshot_paths = {}
        dataset_stub = "https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey/resolve/main/"
        
        for obj in data["siblings"]:
            if not obj["rfilename"].startswith("screenshot"):
                continue
            path = obj["rfilename"]
            url = f"{dataset_stub}{path}"
            name = path.split("/")[-1]
            screenshot_paths[name] = url
        
        print(f"Loaded {len(screenshot_paths)} screenshot URL mappings")
        return screenshot_paths
        
    except Exception as e:
        print(f"Warning: Could not load dataset config from {config_path}: {e}")
        print("Will try to build URLs dynamically")
        return {}

def path_to_url(path: str, screenshot_paths: Dict[str, str]) -> Optional[str]:
    """Convert a local file path to a URL."""
    if screenshot_paths:
        return screenshot_paths.get(path, None)
    else:
        # Fallback: construct URL dynamically
        dataset_stub = "https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey/resolve/main/"
        return f"{dataset_stub}screenshot/{path}"

def load_system_prompt(prompt_path: str = "system_prompt.md") -> str:
    """Load the system prompt from file."""
    try:
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load system prompt from {prompt_path}: {e}")
        return "You are a helpful assistant that can interact with smartphone interfaces."

def prepare_interleaved_dataset(dataset: Dataset, screenshot_paths: Dict[str, str], system_prompt: str, max_samples: int = 10) -> List[Dict[str, Any]]:
    """
    Prepare the dataset for training by interleaving the images and text.
    """
    wrangled_samples = []

    for n, sample in enumerate(dataset):
        if len(wrangled_samples) >= max_samples:
            break
            
        sample_messages = []
        
        try:
            steps = literal_eval(sample["steps"]) if isinstance(sample["steps"], str) else sample["steps"]
        except:
            print(f"Could not parse steps for sample {n}. Skipping...")
            continue
            
        if not isinstance(steps, list):
            print(f"Steps is not a list for sample {n}. Skipping...")
            continue
            
        for step in steps:
            image_path = step["screenshot"]
            image_url = path_to_url(image_path, screenshot_paths)
            if image_url is None:
                print(f"Image URL not found for {image_path}. Skipping...")
                sample_messages = []
                break
                
            step_string = json.dumps(step)
            sample_messages.append({
                "type": "image",
                "image": image_url,
            })
            sample_messages.append({
                "type": "text",
                "text": step_string,
            })
            
        if not sample_messages:
            print(f"No messages for sample {n}. Skipping...")
            continue
            
        task = sample["task"]
        instruction = sample["instruction"]
        app = sample["app"]
        
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"App: {app}\nTask: {task}\nInstruction: {instruction}",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": sample_messages,
            }
        ]
        
        wrangled_samples.append({
            "episode_id": str(n),
            "messages": messages
        })
        print(f"Processed sample {n}")
        
    print(f"Prepared {len(wrangled_samples)} samples from dataset")
    return wrangled_samples

def process_images_in_content(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process content list to download images and convert URLs to PIL objects."""
    processed_content = []
    
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "image" and "image" in item:
                # Download image and replace URL with PIL object
                image_url = item["image"]
                try:
                    pil_image = download_image_with_retry(image_url)
                    processed_content.append({
                        "type": "image",
                        "image": pil_image  # TRL expects PIL image directly
                    })
                except Exception as e:
                    print(f"Error processing image {image_url}: {e}")
                    # Skip failed images
                    continue
            elif item.get("type") == "text" and "text" in item:
                # Keep text content
                processed_content.append({
                    "type": "text",
                    "text": item["text"]
                })
            else:
                # Handle other content types (convert to text)
                text_content = str(item.get("text", str(item)))
                processed_content.append({
                    "type": "text", 
                    "text": text_content
                })
        else:
            # Handle non-dict items (convert to text)
            processed_content.append({
                "type": "text",
                "text": str(item)
            })
    
    return processed_content

def convert_to_sft_format(prepared_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert prepared samples to TRL SFT format with downloaded PIL images."""
    processed_samples = []
    
    for i, sample in enumerate(prepared_samples):
        try:
            messages = sample["messages"]
            
            # Process each message to download images and convert to PIL objects
            processed_messages = []
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    # Process content to download images
                    if isinstance(message["content"], list):
                        processed_content = process_images_in_content(message["content"])
                        processed_messages.append({
                            "role": message["role"],
                            "content": processed_content
                        })
                    else:
                        # Handle non-list content (convert to list format)
                        if isinstance(message["content"], str):
                            processed_messages.append({
                                "role": message["role"],
                                "content": [{
                                    "type": "text", 
                                    "text": message["content"]
                                }]
                            })
                        else:
                            # Handle other content types
                            processed_messages.append({
                                "role": message["role"],
                                "content": [{
                                    "type": "text",
                                    "text": str(message["content"])
                                }]
                            })
                else:
                    # Handle malformed messages
                    processed_messages.append({
                        "role": message.get("role", "user"),
                        "content": [{
                            "type": "text",
                            "text": str(message)
                        }]
                    })
            
            # TRL SFT expects just "messages" column, not episode_id
            processed_samples.append({
                "messages": processed_messages
            })
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Successfully converted {len(processed_samples)} samples to SFT format")
    return processed_samples

def define_dataset_features():
    """Define the features for the TRL SFT dataset."""
    # Simplified structure that matches TRL expectations
    return None  # Let datasets infer the schema automatically

def create_sft_vlm_dataset(
    dataset_name: str = "OpenGVLab/GUI-Odyssey",
    output_dataset_path: str = "data/sft_vlm_dataset",
    max_samples: int = 10,
    max_workers: int = 4,
    config_path: str = "data/dataset_config.json",
    system_prompt_path: str = "system_prompt.md",
    hub_repo_id: Optional[str] = None
):
    """Main function to create TRL SFT-compatible VLM dataset from GUI-Odyssey."""
    
    # Load configuration and system prompt
    screenshot_paths = load_dataset_config(config_path)
    system_prompt = load_system_prompt(system_prompt_path)
    
    # Load the original dataset
    print(f"Loading dataset: {dataset_name}")
    if dataset_name.startswith("data/"):
        # Load local dataset
        dataset = load_from_disk(dataset_name)
        if hasattr(dataset, 'keys') and 'train' in dataset.keys():
            dataset = dataset['train']
    else:
        # Load from hub
        dataset = load_dataset(dataset_name, split="all", streaming=True)
    
    # Prepare interleaved dataset (convert to conversation format)
    print(f"Preparing interleaved dataset with max {max_samples} samples...")
    prepared_samples = prepare_interleaved_dataset(dataset, screenshot_paths, system_prompt, max_samples)
    
    if not prepared_samples:
        raise ValueError("No samples were successfully prepared!")
    
    # Convert to SFT format with PIL images
    print("Converting to SFT format and downloading images...")
    sft_samples = convert_to_sft_format(prepared_samples)
    
    if not sft_samples:
        raise ValueError("No samples were successfully converted to SFT format!")
    
    # Create the new dataset with automatic schema inference
    print("Creating Hugging Face dataset...")
    sft_dataset = Dataset.from_list(sft_samples)
    
    # Save the dataset
    print(f"Saving dataset to: {output_dataset_path}")
    sft_dataset.save_to_disk(output_dataset_path)
    
    # Print dataset info
    print("\nDataset created successfully!")
    print(f"Final dataset size: {len(sft_dataset)}")
    print(f"Dataset features: {sft_dataset.features}")
    
    # Push to hub if repo_id is provided
    if hub_repo_id:
        print(f"Pushing dataset to Hub: {hub_repo_id}")
        sft_dataset.push_to_hub(hub_repo_id)
        print("Dataset successfully pushed to Hub!")
    else:
        print("Skipping Hub upload (no repo_id provided)")
    


def main():
    parser = argparse.ArgumentParser(description="Create TRL SFT-compatible VLM dataset from GUI-Odyssey")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="OpenGVLab/GUI-Odyssey",
        help="Dataset name or path to local dataset"
    )
    parser.add_argument(
        "--output_dataset_path", 
        type=str,
        default="data/sft_vlm_dataset",
        help="Path to save the processed dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of worker threads for image downloading"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="data/dataset_config.json",
        help="Path to dataset configuration file"
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default="system_prompt.md",
        help="Path to system prompt file"
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID to push the dataset to (e.g. 'username/dataset-name')"
    )
    
    args = parser.parse_args()
    
    create_sft_vlm_dataset(
        dataset_name=args.dataset_name,
        output_dataset_path=args.output_dataset_path,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        config_path=args.config_path,
        system_prompt_path=args.system_prompt_path,
        hub_repo_id=args.hub_repo_id
    )

if __name__ == "__main__":
    main() 