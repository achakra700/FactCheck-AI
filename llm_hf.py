"""
Hugging Face LLM Integration using Inference API.
Improved with retry logic and better error handling.
"""

import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def call_llm(prompt, max_new_tokens=512, temperature=0.1):
    """
    Call Hugging Face Inference API with retry logic.
    
    Args:
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    if not HF_API_TOKEN:
        raise ValueError(
            "HF_API_TOKEN not set. Get your token from https://huggingface.co/settings/tokens"
        )
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }

    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                    return data[0]["generated_text"].strip()
                return str(data)
                
            elif response.status_code == 503:
                print("⏳ Model loading, retrying in 20s...")
                time.sleep(20)
                retry_count += 1
                
            else:
                raise RuntimeError(f"HF API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏳ Request timeout, retrying...")
            retry_count += 1
            time.sleep(5)
            
        except Exception as e:
            if retry_count < max_retries - 1:
                print(f"Error: {e}, retrying...")
                retry_count += 1
                time.sleep(5)
            else:
                raise

    raise RuntimeError("Max retries exceeded")


# Example usage
if __name__ == "__main__":
    response = call_llm("What is the capital of France?")
    print(response)
