import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
import base64
import io
import json
import time
import uuid
import threading
from typing import Dict, List, Optional, Union, Generator
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.image_utils import fetch_image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Union[str, Dict]]]]

class ChatCompletionRequest(BaseModel):
    model: str = "dots-ocr"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 12000
    stream: Optional[bool] = False
    
class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    
class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]

# Global variables for model and processor
model = None
processor = None

# Alternative: Use semaphore for configurable concurrency (set MAX_CONCURRENT_REQUESTS=1 for single processing)
MAX_CONCURRENT_REQUESTS = int(os.environ.get('MAX_CONCURRENT_REQUESTS', '1'))
processing_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

app = FastAPI()

def initialize_model():
    global model, processor
    if model is None or processor is None:
        model_path = "./weights/DotsOCR"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def extract_image_from_message(content) -> Optional[str]:
    if isinstance(content, str):
        return None
    
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            return item.get("image_url", {}).get("url")
        elif isinstance(item, dict) and item.get("type") == "image":
            return item.get("image")
    return None

def extract_text_from_message(content) -> str:
    if isinstance(content, str):
        return content
    
    text_parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(item.get("text", ""))
    return " ".join(text_parts)

def inference_openai_format(image_input, prompt, temperature=0.1, top_p=1.0, max_tokens=12000, stream=False):
    global model, processor
    
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Use semaphore to limit concurrent processing (supports both single and multiple concurrent requests)
    with processing_semaphore:
        # Use fetch_image to handle various image formats
        try:
            image = fetch_image(image_input)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
        
        # Save image temporarily for processing
        temp_path = f"/tmp/temp_image_{uuid.uuid4().hex}.jpg"
        image.save(temp_path)
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": temp_path
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")
            
            if stream:
                return generate_stream(inputs, max_tokens, temperature, top_p)
            else:
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=max_tokens, 
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        top_p=top_p
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    return output_text[0] if output_text else ""
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

def generate_stream(inputs, max_tokens, temperature, top_p):
    from transformers import TextIteratorStreamer
    import threading
    
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "streamer": streamer
    }
    
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for text in streamer:
        if text:
            yield text

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    initialize_model()
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    last_message = request.messages[-1]
    if last_message.role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")
    
    # Extract image and text from the message
    image_input = extract_image_from_message(last_message.content)
    text_prompt = extract_text_from_message(last_message.content)
    
    if not image_input:
        raise HTTPException(status_code=400, detail="No image found in the message")
    
    if not text_prompt:
        # Use default OCR prompt if no text is provided
        text_prompt = dict_promptmode_to_prompt.get('prompt_layout_all_en', 'Extract text from the image.')
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    if request.stream:
        def stream_generator():
            full_content = ""
            try:
                for chunk in generate_stream_response(image_input, text_prompt, request.temperature, request.top_p, request.max_tokens):
                    full_content += chunk
                    
                    stream_response = ChatCompletionStreamResponse(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": chunk},
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {stream_response.model_dump_json()}\n\n"
                
                # Send final chunk
                final_response = ChatCompletionStreamResponse(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={},
                            finish_reason="stop"
                        )
                    ]
                )
                yield f"data: {final_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "server_error"
                    }
                }
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        try:
            content = inference_openai_format(
                image_input, 
                text_prompt, 
                request.temperature, 
                request.top_p, 
                request.max_tokens, 
                False
            )
            
            response = ChatCompletionResponse(
                id=completion_id,
                object="chat.completion",
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=content),
                        finish_reason="stop"
                    )
                ],
                usage={
                    "prompt_tokens": 0,  # TODO: Calculate actual token usage
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            )
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

def generate_stream_response(image_input, prompt, temperature, top_p, max_tokens):
    global model, processor
    
    # Use semaphore to limit concurrent processing (supports both single and multiple concurrent requests)
    with processing_semaphore:
        # Use fetch_image to handle various image formats
        image = fetch_image(image_input)
        temp_path = f"/tmp/temp_image_{uuid.uuid4().hex}.jpg"
        image.save(temp_path)
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": temp_path
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")
            
            for chunk in generate_stream(inputs, max_tokens, temperature, top_p):
                yield chunk
                
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "dots-ocr",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dots-ocr"
            }
        ]
    }

def inference(image_path, prompt, model, processor, temperature=0.1, top_p=1.0, max_height=None, max_width=None, max_new_tokens=12000, stream=False):
    print(f"Processing image: {image_path}")
    image = fetch_image(image_path)

    max_width = 1200
    if max_width is not None and image.width > max_width:
        new_size = (max_width, int(image.height * (max_width / image.width)))
        print(f"Resize image from {image.size} to {new_size}")
        image = image.resize(new_size, Image.LANCZOS)
        resized_path = "/tmp/resized_image.jpg"
        image.save(resized_path)
        image_path = resized_path

    if max_height is not None and image.height > max_height:
        new_size = (int(image.width * (max_height / image.height)), max_height)
        print(f"Resize image from {image.size} to {new_size}")
        image = image.resize(new_size, Image.LANCZOS)
        resized_path = "/tmp/resized_image.jpg"
        image.save(resized_path)
        image_path = resized_path

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")
    print(inputs.keys())
    for key, tensor in inputs.items():
        if isinstance(tensor, torch.Tensor):
            memory_mb = tensor.element_size() * tensor.nelement() / (1024*1024)
            print(f"{key}: shape={tensor.shape}, memory={memory_mb:.2f}MB")

    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    
    if stream:
        from transformers import TextStreamer
        streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                temperature=temperature,
                top_p=top_p,
                streamer=streamer
            )
    else:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=temperature, top_p=top_p)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        print(output_text)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run DotsOCR inference or server')
    parser.add_argument('--server', action='store_true', help='Run as OpenAI-compatible server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--prompt_type', type=str, default='prompt_layout_all_en',
                        choices=['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_ocr', 'prompt_grounding_ocr'],
                        help='Prompt type')
    parser.add_argument('--image_path', type=str, default='demo/demo_image1.jpg',
                        help='Path to the input image')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top p for generation')
    parser.add_argument('--max_height', type=int, default=None,
                        help='Maximum height for image resizing')
    parser.add_argument('--max_width', type=int, default=None,
                        help='Maximum width for image resizing')
    parser.add_argument('--max_new_tokens', type=int, default=12000,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--stream', action='store_true',
                        help='Enable streaming output')
    
    args = parser.parse_args()
    
    if args.server:
        print(f"Starting OpenAI-compatible server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        model_path = "./weights/DotsOCR"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={ "": "cuda:0" },
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        prompt = dict_promptmode_to_prompt[args.prompt_type]
        print(f"prompt: {prompt}")
        inference(args.image_path, prompt, model, processor, 
                  temperature=args.temperature, top_p=args.top_p, 
                  max_height=args.max_height, max_width=args.max_width, 
                  max_new_tokens=args.max_new_tokens, stream=args.stream)