import xgboost as xgb
import numpy as np
import time
import sys

print(f"--- XGBoost GPU 验证 ---")
print(f"XGBoost Version: {xgb.__version__}")

# 1. 创建一些大型虚拟数据
# 我们需要足够的数据，以便在 GPU 上能观察到负载
N_SAMPLES = 1000000  # 100 万个样本
N_FEATURES = 50      # 50 个特征

print(f"\nCreating dummy data ({N_SAMPLES} samples, {N_FEATURES} features)...")
try:
    X = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
    y = np.random.randint(0, 2, size=N_SAMPLES).astype(np.float32)

    # 将数据转换为 XGBoost 的 DMatrix 格式
    dtrain = xgb.DMatrix(X, label=y)
    print("Data creation successful.")

except Exception as e:
    print(f"❌ 数据创建失败: {e}", file=sys.stderr)
    print("   (提示: 可能是内存不足)", file=sys.stderr)
    sys.exit(1)


# 2. (关键) 设置 GPU 训练参数
# 'device': 'cuda' 明确告诉 XGBoost 使用 CUDA (NVIDIA GPU)
# 'tree_method': 'hist' 在现代版本中会自动启用 (以前需要 'gpu_hist')
params = {
    'device': 'cuda',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 100,  # 我们训练 100 棵树
    'max_depth': 5
}

print(f"\nAttempting to train on GPU (device='cuda')...")

# 3. 运行训练并计时
try:
    start_time = time.time()

    # 执行训练
    # 'evals' 参数只是为了让训练过程输出一些日志
    bst = xgb.train(params, dtrain, evals=[(dtrain, 'train')], num_boost_round=params['n_estimators'], verbose_eval=False)

    end_time = time.time()

    print(f"\n✅✅✅ XGBoost GPU 训练成功! ✅✅✅")
    print(f"Total Training Time: {end_time - start_time:.4f} seconds.")
    print("   (如果此时间非常短，说明 GPU 运行良好)")

except xgb.core.XGBoostError as e:
    print(f"\n❌❌❌ XGBoost GPU 训练失败! ❌❌❌", file=sys.stderr)
    print(f"错误: {e}", file=sys.stderr)
    print("\n   (提示: 最常见的原因是 XGBoost 无法找到 CUDA 库。)", file=sys.stderr)
    print("   (请确保您的 Conda 环境中的 PyTorch-CUDA 捆绑包已正确安装。)", file=sys.stderr)
except Exception as e:
    print(f"\n❌ 发生意外错误: {e}", file=sys.stderr)

print(f"\n--- 验证结束 ---")