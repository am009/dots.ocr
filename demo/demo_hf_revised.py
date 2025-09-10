import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

def inference(image_path, prompt, model, processor):
    from PIL import Image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    
    max_size = 1200
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        print(f"Resized image from {Image.open(image_path).size} to {image.size}")
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
    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=12000, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    print(output_text)



if __name__ == "__main__":

    model_path = "./weights/DotsOCR"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.float32,
        device_map={ "": "cuda:0" },
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "14GiB"}  # Limit memory per GPU
    )
    processor = AutoProcessor.from_pretrained(model_path,  trust_remote_code=True)

    image_path = "demo/demo_image1.jpg"
    prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
    print(f"prompt: {prompt}")
    inference(image_path, prompt, model, processor)