export HF_HOME=/ppio_net0/huggingface
#MODEL_PATH="continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-ocr-merged"
#MODEL_PATH="continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-gqa-merged"
MODEL_PATH="continual-ckpt/llava-c/llava-v1.5c-7b-lora-task-vg-merged"
#MODEL_PATH="liuhaotian/llava-v1.5-7b"
#MODEL_PATH="continual-ckpt/domain/llava-v1.5-7b-lora-task-coco-merged"
#MODEL_PATH="continual-ckpt/domain/llava-v1.5-7b-lora-task-ocr_vqa-merged"
#MODEL_PATH="continual-ckpt/domain/llava-v1.5-7b-lora-task-textvqa-merged"
#MODEL_PATH="continual-ckpt/domain/llava-v1.5-7b-lora-task-gqa-merged"
#MODEL_PATH="continual-ckpt/domain/llava-v1.5-7b-lora-task-vg-merged"
#MODEL_PATH="continual-ckpt/domain-incremental-mse/llava-v1.5-7b-lora-task-coco-v4-lambda1.0-merged"
#python -m llava.serve.cli \
#    --model-path  $MODEL_PATH\
#    --image-file "demo/020.jpg"


#python -m llava.serve.cli \
#    --model-path  $MODEL_PATH\
#    --image-file "demo/011341143X.jpg"

#python -m llava.serve.cli \
#    --model-path  $MODEL_PATH\
#    --image-file "demo/001.jpg"


python -m llava.serve.cli \
    --model-path  $MODEL_PATH\
    --image-file "demo/011.jpg"
