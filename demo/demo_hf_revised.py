import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

def inference(image_path, prompt, model, processor, temperature=0.1, top_p=1.0, max_height=None, max_width=None, max_new_tokens=12000, stream=False):
    from PIL import Image
    print(f"Processing image: {image_path}")
    original_image = Image.open(image_path)
    image = original_image

    max_width = 1200
    if max_width is not None and image.width > max_width:
        new_size = (max_width, int(image.height * (max_width / image.width)))
        print(f"Resize image from {original_image.size} to {new_size}")
        image = original_image.resize(new_size, Image.LANCZOS)
        resized_path = "/tmp/resized_image.jpg"
        image.save(resized_path)
        image_path = resized_path

    if max_height is not None and image.height > max_height:
        new_size = (int(original_image.width * (max_height / image.height)), max_height)
        print(f"Resize image from {original_image.size} to {new_size}")
        image = original_image.resize(new_size, Image.LANCZOS)
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
        # Inference: Generation of the output with streaming
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
    
    parser = argparse.ArgumentParser(description='Run DotsOCR inference')
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

    model_path = "./weights/DotsOCR"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.float32,
        device_map={ "": "cuda:0" },
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        # max_memory={0: "14GiB"}  # Limit memory per GPU
    )
    processor = AutoProcessor.from_pretrained(model_path,  trust_remote_code=True)

    # prompt type必须是下列四种之一：
    # prompt_layout_all_en prompt_layout_only_en prompt_ocr prompt_grounding_ocr
    prompt = dict_promptmode_to_prompt[args.prompt_type]
    print(f"prompt: {prompt}")
    inference(args.image_path, prompt, model, processor, 
              temperature=args.temperature, top_p=args.top_p, 
              max_height=args.max_height, max_width=args.max_width, 
              max_new_tokens=args.max_new_tokens, stream=args.stream)
