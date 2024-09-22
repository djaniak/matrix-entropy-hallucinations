#!/bin/bash

# NOTE 1: Runs on A40 GPU with 48GB of memory
# NOTE 2: To run on a specific GPU index pass it exactly as follows: --gpus '"device=0"'
docker run \
    --runtime nvidia \
    --gpus <your_device> \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env-file "vllm.env" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
        --model "mistralai/Mistral-Nemo-Instruct-2407" \
        --device cuda \
        --dtype bfloat16 \
        --max-model-len 8000 \
        --max-num-seqs 64
