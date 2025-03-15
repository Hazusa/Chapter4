from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN

# 1. 加载数据
dataset = np.load("dataset.npy")  # 形状 (180, 5, 15)
labels = np.load("labels.npy")    # 形状 (180,)

# 2. 检查并处理NaN
nan_mask = np.isnan(dataset).any(axis=(1, 2))  # 检查每个样本是否含NaN
if nan_mask.any():
    print(f"删除含NaN的样本数量: {nan_mask.sum()}")
    dataset = dataset[~nan_mask]  # 删除含NaN的样本
    labels = labels[~nan_mask]    # 同步删除对应的标签
nan_locations = np.argwhere(np.isnan(dataset))
print("NaN的位置（样本, 行, 列）：\n", nan_locations)

# 2. 展平为二维数据 (180, 5 * 15=75)
n_samples, n_rows, n_cols = dataset.shape
X_flattened = dataset.reshape(n_samples, -1)  # (180, 75)

# 3. 数据标准化（PCA对尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flattened)

# 4. PCA降维
pca = PCA(n_components=20)  # 保留20个主成分
X_pca = pca.fit_transform(X_scaled)
# 6. 检查类别分布
label_counts = Counter(labels)
minority_class = min(label_counts, key=label_counts.get)
minority_count = label_counts[minority_class]
print("标签分布:", label_counts)
print(f"少数类样本数量: {minority_count}")

# 5. ADASYN过采样
adasyn = ADASYN(sampling_strategy={1: 60}, n_neighbors=min(3, minority_count - 1)) #将类别标签为1的样本数量调整为60
X_resampled_pca, y_resampled = adasyn.fit_resample(X_pca, labels)

# 6. PCA逆变换重建特征
X_resampled_scaled = pca.inverse_transform(X_resampled_pca)
X_resampled = scaler.inverse_transform(X_resampled_scaled)  # 逆标准化

# 7. 恢复三维结构 (n_samples, 5, 15)
X_resampled_3d = X_resampled.reshape(-1, n_rows, n_cols)  # 形状 (n_new_samples, 5, 15)

# 8. 保存结果
np.save("new_dataset2.npy", X_resampled_3d)
np.save("new_labels2.npy", y_resampled)