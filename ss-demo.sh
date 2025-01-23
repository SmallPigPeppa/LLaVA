export HF_HOME=/ppio_net0/huggingface
MODEL_PATH="continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-ocr-merged"
#MODEL_PATH="liuhaotian/llava-v1.5-7b"
python -m llava.serve.cli \
    --model-path  $MODEL_PATH\
    --image-file "demo/020.jpg"
