import pandas as pd

# 读取Excel文件
input_file = 'input.xlsx'
output_file = 'output.xlsx'

# 加载Excel数据
df = pd.read_excel(input_file)

# 确保列数至少为16列
if df.shape[1] < 16:
    raise ValueError("输入文件的列数不足16列，请检查数据格式。")

# 初始化结果列表
result = []

# 按每五行处理
for i in range(0, len(df), 5):
    # 获取当前块的五行数据
    block = df.iloc[i:i + 5, :15]

    # 将五行的前15列展平为一行
    flattened = block.values.flatten()

    # 如果存在第一行的第16列，则获取其值
    if i < len(df) and len(df.columns) >= 16:
        extra_value = df.iloc[i, 15]  # 第一行的第16列
    else:
        extra_value = None  # 如果不存在第16列，设置为None或空值

    # 合并展平的数据和额外的值
    result_row = list(flattened) + [extra_value]

    # 添加到结果列表中
    result.append(result_row)

# 创建结果DataFrame
result_df = pd.DataFrame(result)

# 保存到新的Excel文件
result_df.to_excel(output_file, index=False)

print(f"处理完成，结果已保存到 {output_file}")