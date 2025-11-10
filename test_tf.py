import tensorflow as tf
import sys
import time

print("--- 1. 开始 TensorFlow GPU 验证 ---")

# 打印 TensorFlow 版本
print(f"TensorFlow 版本: {tf.__version__}")
print(f"Python 版本: {sys.version.split()[0]}")

# 步骤 1: 检测物理 GPU
print("\n--- 2. 检测物理 GPU 设备 ---")
gpus = tf.config.list_physical_devices('GPU')

if not gpus:
    print("❌ 验证失败: TensorFlow 未检测到任何物理 GPU。")
    print("请检查您的 NVIDIA 驱动、CUDA Toolkit 和 cuDNN 版本是否匹配。")
    print("--- 验证结束 ---")
    sys.exit()

print(f"✅ 成功: 找到 {len(gpus)} 个物理 GPU:")
for i, gpu in enumerate(gpus):
    print(f"  GPU [{i}]: {gpu.name}")

# 默认情况下，TensorFlow 会自动在第一个 GPU 上运行
# 我们可以通过 tf.device 明确指定它

try:
    # 步骤 2: 尝试在 GPU 上执行计算
    print("\n--- 3. 尝试在 GPU 上执行矩阵乘法 ---")
    
    # 开启 TensorFlow 的设备放置日志 (显示哪个设备在执行操作)
    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(True)

    # 显式在第一个 GPU (GPU:0) 上创建两个大张量并执行乘法
    with tf.device(f'/device:GPU:0'):
        print("正在 GPU:0 上创建张量...")
        start_time = time.time()
        a = tf.random.normal([2000, 2000], dtype=tf.float32)
        b = tf.random.normal([2000, 2000], dtype=tf.float32)
        
        print("正在 GPU:0 上执行矩阵乘法 (tf.matmul)...")
        c = tf.matmul(a, b)
        
        # .numpy() 会强制等待计算完成
        c.numpy() 
        end_time = time.time()

    print(f"\n✅ 成功: GPU 矩阵乘法执行完毕。")
    print(f"   耗时: {end_time - start_time:.4f} 秒")
    print("   (请检查上面的日志，您应该能看到 'device:GPU:0' 相关的输出)")


    # 步骤 3: 验证 CPU vs GPU (可选)
    print("\n--- 4. (对比) 尝试在 CPU 上执行相同操作 ---")
    with tf.device('/device:CPU:0'):
        print("正在 CPU:0 上创建张量...")
        start_time = time.time()
        a_cpu = tf.random.normal([2000, 2000], dtype=tf.float32)
        b_cpu = tf.random.normal([2000, 2000], dtype=tf.float32)
        
        print("正在 CPU:0 上执行矩阵乘法 (tf.matmul)...")
        c_cpu = tf.matmul(a_cpu, b_cpu)
        c_cpu.numpy()
        end_time = time.time()

    print(f"\n✅ 成功: CPU 矩阵乘法执行完毕。")
    print(f"   耗时: {end_time - start_time:.4f} 秒")
    print("   (通常，GPU 耗时会远少于 CPU 耗时)")


except Exception as e:
    print(f"\n❌ 验证失败: 在 GPU 上执行计算时发生运行时错误。")
    print(f"错误详情: {e}")
    print("这可能表示虽然检测到了 GPU，但 cuDNN 或 CUDA 库在运行时加载失败。")

print("\n--- 验证结束 ---")