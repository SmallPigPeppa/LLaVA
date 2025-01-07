import torch
import time


# 定义一个函数来占用GPU并进行计算
def occupy_gpu_and_compute(gpu_id):
    try:
        # 指定使用的GPU
        device = torch.device(f"cuda:{gpu_id}")

        # 创建两个大Tensor来占用显存
        tensor1 = torch.randn(25, 10000, 10000, device=device)
        tensor2 = torch.randn(25, 10000, 10000, device=device)

        # 打印确认信息
        # print(f"GPU {gpu_id} is being occupied and computations are being performed.")

        result_add = tensor1 + tensor2

        # 乘法计算
        result_mul = tensor1 * tensor2

        # print(
        #     f"GPU {gpu_id}: Add result first element = {result_add[0, 0].item()}, Mul result first element = {result_mul[0, 0].item()}")

        # time.sleep(1)

    except Exception as e:
        print(f"Error on GPU {gpu_id}: {e}")


# 在0到7号GPU上运行
if __name__ == "__main__":
    while True:
        for gpu_id in range(0,4):
            occupy_gpu_and_compute(gpu_id)
