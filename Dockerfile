from rednotehilab/dots.ocr:vllm-openai-v0.9.1

# docker run --name dots-ocr-container -it -v .:/DotsOCR --runtime=nvidia --gpus=all --privileged --entrypoint bash rednotehilab/dots.ocr:vllm-openai-v0.9.1
# docker run --name dots-ocr-container -p 51234:5000 --restart=always -it -v /sn640/ai-apps/dots.ocr:/workspace --runtime=nvidia --gpus=all --privileged --entrypoint bash rednotehilab/dots.ocr:vllm-openai-v0.9.1 /workspace/start.sh
# docker stop dots-ocr-container ; docker start dots-ocr-container
# docker logs -f dots-ocr-container
# docker stop dots-ocr-container ; docker start dots-ocr-container

RUN git clone https://github.com/rednote-hilab/dots.ocr.git /workspace && \
    cd /workspace && \
    pip install flask --ignore-installed && \
    pip install -e .

ENTRYPOINT ['/bin/bash', '/workspace/start.sh']

# sed -i 's/bf16=True/bf16=False/' weights/DotsOCR/modeling_dots_vision.py


# pip install flash_attn_triton==0.1.1
# rm -rf /usr/local/lib/python3.12/dist-packages/flash_attn
