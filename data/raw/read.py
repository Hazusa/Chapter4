import numpy as np

# 指定 .npy 文件路径
file_path = 'labels.npy'  # 替换为你的 .npy 文件路径

# 加载 .npy 文件
data = np.load(file_path)

# 查看数据的基本信息
print("Shape of the array:", data.shape)  # 查看数组形状
print("Data type of the array:", data.dtype)  # 查看数组数据类型

# 打印数组内容
print("Array content:")
print(data)