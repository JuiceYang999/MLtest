import tensorflow as tf
import sys

print(f"--- TensorFlow 验证 ---")
try:
    print(f"TensorFlow Version: {tf.__version__}")

    # 检查 GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"找到的物理 GPU 数量: {len(gpus)}")

    if not gpus:
        print("❌ 未找到 GPU。TensorFlow 将使用 CPU。")
    else:
        for i, gpu in enumerate(gpus):
            print(f"  GPU [{i}]: {gpu.name}")

        # 尝试在 GPU 上执行一个简单操作
        tf.config.set_visible_devices(gpus[0], 'GPU')
        with tf.device(gpus[0].name):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print("\n✅ 成功在 GPU 上执行矩阵乘法。")
        print("   结果:", c.numpy())

except Exception as e:
    print(f"\n❌ TensorFlow GPU 验证失败: {e}", file=sys.stderr)

print(f"--- 验证结束 ---")