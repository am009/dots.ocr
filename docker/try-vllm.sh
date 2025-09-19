export CUDA_VISIBLE_DEVICES=0
export VLLM_SERVER_DEV_MODE=1
export PYTHONPATH=/workspace/weights:$PYTHONPATH
sed -i '/^from vllm\.entrypoints\.cli\.main import main/a from DotsOCR import modeling_dots_ocr_vllm' $(which vllm)
pip uninstall flash-attn
vllm serve /workspace/weights/DotsOCR \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.95 \
            --chat-template-content-format string \
            --served-model-name dotsocr-model \
            --enable-sleep-mode \
            --trust-remote-code \
            --max-model-len 16000 \
            --max-num-seqs 1 \
            --model-impl transformers


