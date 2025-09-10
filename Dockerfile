from rednotehilab/dots.ocr:vllm-openai-v0.9.1

RUN git clone https://github.com/rednote-hilab/dots.ocr.git /DotsOCR
cd /DotsOCR
pip install -e .

sed -i 's/bf16=True/bf16=False/' /root/.cache/huggingface/modules/transformers_modules/DotsOCR/modeling_dots_vision.py