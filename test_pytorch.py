import torch
import sys

print("--- PyTorch 验证 ---")

try:
    # 1. 检查 CUDA 是否可用
    is_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_available}")

    if not is_available:
        print("❌ PyTorch 未找到 CUDA。请检查 NVIDIA 驱动和 Conda 安装。", file=sys.stderr)
        sys.exit(1)

    # 2. 获取并打印 GPU 型号
    device_name = torch.cuda.get_device_name(0)
    print(f"Device Name: {device_name}")

    # 3. 检查 PyTorch 所使用的 CUDA 版本
    torch_cuda_version = torch.version.cuda
    print(f"PyTorch built with CUDA: {torch_cuda_version}")

    # 4. 执行一个简单的 GPU 运算
    print("\nAttempting simple tensor operation on GPU...")
    x = torch.tensor([1.0, 2.0]).cuda()
    y = torch.tensor([3.0, 4.0]).cuda()
    z = x + y
    print(f"✅ GPU operation successful. Result: {z.cpu()}")

except Exception as e:
    print(f"\n❌ PyTorch GPU 验证失败: {e}", file=sys.stderr)

print("--- 验证结束 ---")