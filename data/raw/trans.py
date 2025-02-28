import pandas as pd
import numpy as np

# 读取Excel文件（假设无表头）
excel_path = "raw.xlsx"
df = pd.read_excel(excel_path, header=None)

# 分割特征和标签
data = df.iloc[:, :75].values  # 前75列为特征
labels = df.iloc[:, 75].values  # 第76列为标签

# 重塑为3D数组 [样本数, 时间步, 特征]
n_samples = data.shape[0]
npy_data = data.reshape(n_samples, 5, 15)  # 转换为(样本数,5,15)

# 保存文件
np.save("dataset.npy", npy_data)
np.save("labels.npy", labels)