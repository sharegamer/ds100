import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 创建示例数据
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [2, 2],
    np.random.randn(100, 2) + [6, 2],
    np.random.randn(100, 2) + [4, 6]
])

# 使用不同的随机种子初始化
results = []
for seed in [0, 1, 2, 50, 100]:
    kmeans = KMeans(n_clusters=3, init='random', random_state=seed, n_init=1)
    kmeans.fit(X)
    results.append(kmeans.inertia_)  # inertia_是簇内平方和
    
print("不同初始化的簇内平方和：", results)
# 输出可能类似：[329.45, 342.18, 328.93, 355.21, 329.45]