# #!/bin/bash

python -m vllm.entrypoints.openai.api_server \
    --tensor-parallel-size 2 \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 9570 \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 128000 \
    --limit-mm-per-prompt '{\"image\": 64, \"video\": 256}' \
    --mm-processor-kwargs '{\"max_pixels\": 1024, \"min_pixels\": 128}' \
    --max-num-batched-tokens 8192