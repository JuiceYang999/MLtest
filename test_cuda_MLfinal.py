import xgboost as xgb
import numpy as np
import time
import sys
import warnings

print("--- XGBoost GPU 最终验证 ---")
print(f"XGBoost Version: {xgb.__version__}\n")

# 1. 创建中等规模的数据
N_SAMPLES = 1_000_000  # 100 万样本
N_FEATURES = 50      # 50 个特征
N_ESTIMATORS = 100   # 100 棵树

print(f"创建数据 ({N_SAMPLES} samples, {N_FEATURES} features)...")
try:
    X = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
    y = np.random.randint(0, 2, size=N_SAMPLES).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    print("数据创建成功。\n")
except Exception as e:
    print(f"❌ 数据创建失败: {e} (可能是内存不足)", file=sys.stderr)
    sys.exit(1)

# 2. 定义训练参数
params = {
    'device': 'cuda',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6
}

# 3. (关键) 运行并捕获警告
print(f"--- 正在执行 GPU 训练 (device='cuda') ---")

was_cuda_warning_found = False
time_taken = 0.0

# 我们使用 'with warnings.catch_warnings...' 来“捕获”所有弹出的警告
with warnings.catch_warnings(record=True) as w_list:
    warnings.simplefilter("always")  # <--- 这是对上一脚本 Bug 的修复
    
    start_time = time.time()
    
    try:
        # 运行训练
        xgb.train(
            params,
            dtrain,
            num_boost_round=N_ESTIMATORS,
            verbose_eval=False
        )
        
        end_time = time.time()
        time_taken = end_time - start_time
        
    except Exception as e:
        print(f"\n❌❌❌ 训练期间发生崩溃! ❌❌❌", file=sys.stderr)
        print(f"   错误: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 检查被捕获的警告中，是否有我们关心的那一个
    for w in w_list:
        if "not compiled with CUDA support" in str(w.message):
            was_cuda_warning_found = True
            break

# 4. 最终诊断
print(f"\n训练用时: {time_taken:.4f} 秒")

if was_cuda_warning_found:
    print("\n--- 诊断结果: ❌ 失败 ❌ ---")
    print("   原因: XGBoost 报告它未编译 CUDA 支持 (已回退到 CPU)。")
else:
    print("\n--- 诊断结果: ✅✅✅ 成功 ✅✅✅ ---")
    print("   原因: 未检测到 CUDA 编译警告。")
    print(f"   GPU 已参与训练 (在 {time_taken:.4f} 秒内完成)。")

print("\n--- 验证结束 ---")