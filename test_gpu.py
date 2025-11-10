import xgboost
import numpy as np
from scipy import stats

# 生成示例数据
np.random.seed(114514)
X = np.random.randn(100, 3)  # 生成100个样本，每个样本有3个特征
y = stats.bernoulli.rvs(0.5, size=100)  # 生成二分类标签，概率为0.5

# 设置参数
params = {
    "device": "cuda"
}

# 创建DMatrix对象
Xy = xgboost.DMatrix(X, y)

# 训练模型
model = xgboost.train(params, Xy)

# 测试模型
test_array = np.random.randn(1, 3)
dtest = xgboost.DMatrix(test_array)
pred = model.predict(dtest)
print(pred)