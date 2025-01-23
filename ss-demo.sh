export HF_HOME=/ppio_net0/huggingface
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "demo/020.jpg"
