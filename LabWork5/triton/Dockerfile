FROM nvcr.io/nvidia/tritonserver:25.12-vllm-python-py3

RUN python3 -m pip install --no-cache-dir \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu130

RUN python3 -m pip install --no-cache-dir transformers sentencepiece accelerate

RUN python3 -m pip install --no-cache-dir safetensors tokenizers