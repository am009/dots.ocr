#!/usr/bin/env python3
"""
Test script for demo_hf_revised.py FastAPI server using OpenAI client
Tests with demo/demo_image1.jpg
"""

import os
import time
import base64
from openai import OpenAI


def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_non_stream():
    """Test non-streaming chat completion"""
    print("Testing non-streaming chat completion...")
    
    client = OpenAI(
        base_url="http://172.17.0.5:8000/v1",
        api_key="dummy-key"
    )
    
    image_path = "demo/demo_image1.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    base64_image = encode_image_to_base64(image_path)
    
    response = client.chat.completions.create(
        model="dots-ocr",
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from this image"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=12000,
        temperature=0.1
    )
    
    print(f"Response ID: {response.id}")
    print(f"Model: {response.model}")
    print(f"Content: {response.choices[0].message.content}")
    print(f"Finish Reason: {response.choices[0].finish_reason}")
    print("-" * 50)


def test_stream():
    """Test streaming chat completion"""
    print("Testing streaming chat completion...")
    
    client = OpenAI(
        base_url="http://localhost:8000/v1", 
        api_key="dummy-key"
    )
    
    image_path = "demo/demo_image1.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    base64_image = encode_image_to_base64(image_path)
    
    stream = client.chat.completions.create(
        model="dots-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Extract all text from this image"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=12000,
        temperature=0.1,
        stream=True
    )
    
    print("Streaming response:")
    full_content = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end='', flush=True)
            full_content += content
    
    print("\n" + "-" * 50)
    print(f"Full content length: {len(full_content)} characters")
    print("-" * 50)


def test_models_endpoint():
    """Test the models endpoint"""
    print("Testing models endpoint...")
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy-key" 
    )
    
    models = client.models.list()
    print(f"Available models: {len(models.data)}")
    for model in models.data:
        print(f"- {model.id} (owned by: {model.owned_by})")
    print("-" * 50)


def main():
    """Run all tests"""
    print("Starting tests for demo_hf_revised.py server")
    print("Make sure the server is running with: python demo/demo_hf_revised.py --server")
    print("=" * 60)
    
    try:
        test_models_endpoint()
        time.sleep(1)
        
        test_non_stream()
        time.sleep(1)
        
        test_stream()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        print("Make sure the server is running on http://localhost:8000")


if __name__ == "__main__":
    main()