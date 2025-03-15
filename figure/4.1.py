import matplotlib.pyplot as plt
import numpy as np

# ===== 数据配置 =====
models = ['CNN', 'LSTM', 'BERT', 'RoBERTa', 'Informer', 'AutoFormer', 'TST', 'ADA-TST']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

# 从表格提取的数值矩阵（百分比形式）
data_percent = np.array([
    [72.22, 75.83, 66.92, 67.61],
    [75.00, 71.78, 71.79, 71.45],
    [77.78, 77.58, 75.46, 75.83],
    [80.56, 84.76, 80.46, 79.41],
    [80.56, 82.47, 76.81, 76.93],
    [83.33, 85.27, 84.85, 82.28],
    [85.22, 85.83, 84.92, 85.61],
    [89.11, 91.03, 89.81, 88.41]
])
data = data_percent / 100  # 转换为0-1范围

# ===== 可视化参数 =====
plt.rcParams["font.family"] = "SimSun"  # 中文字体
bar_width = 0.07  # 调整为更窄的宽度
index = np.arange(len(metrics)) * 0.9  # 指标间距扩大50%

model_colors = [
    '#8dd3c7',  # CNN: 薄荷绿
    '#ffffb3',  # LSTM: 浅黄
    '#bebada',  # BERT: 薰衣草紫
    '#fb8072',  # RoBERTa: 珊瑚红
    '#80b1d3',  # Informer: 天蓝
    '#fdb462',  # AutoFormer: 橙黄
    '#b3de69',  # TST: 青柠绿
    '#fccde5'  # ADA-TST: 粉红
]
# ===== 绘图引擎 =====
fig, ax = plt.subplots(figsize=(15, 6))

# 绘制每个模型的柱状图（带精确数值标签）
for i, (model, model_colors) in enumerate(zip(models, model_colors)):
    offset = bar_width * (i - len(models) / 2 + 0.5)
    bars = ax.bar(index + offset, data[i], bar_width,
                  color=model_colors, label=model, edgecolor='black', linewidth=0.5)

    # 添加百分比标签（保持原始表格精度）
    ax.bar_label(bars,
                 labels=[f"{val:.2f}%" for val in data_percent[i]],
                 fontsize=13, rotation=90, padding=5)

# ===== 样式优化 =====
ax.set_xlabel("评价指标", fontsize=18)
ax.set_xticks(index)
ax.set_xticklabels(metrics, fontsize=18)
ax.set_ylabel("模型性能", fontsize=18)
ax.set_ylim(0, 0.95)  # 对应0-95%的显示范围
ax.set_yticks(np.linspace(0.6, 0.95, 8))
ax.set_yticklabels([f"{int(y * 100)}%" for y in np.linspace(0.6, 0.95, 8)], fontsize=13)

# 网格和边框优化
ax.grid(axis='y', linestyle='--', alpha=0.7)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# 图例优化
legend = ax.legend(
    loc='upper right',           # 将位置改为右上角
    bbox_to_anchor=(1.13, 1),    # 微调位置避免遮挡曲线
    ncol=1,                      # 改为1列布局
    frameon=True,                # 显示边框
    fancybox=True,               # 圆角边框
    shadow=True,                 # 添加阴影
    fontsize=14,                 # 缩小字体
    borderpad=0.8,               # 调整边距
    title='模型',            # 添加图例标题
    title_fontsize=12
)
# 调整画布留白
plt.subplots_adjust(right=0.8)  # 为右侧图例预留空间

# 设置图例背景样式
legend.get_frame().set_facecolor('#F5F5F5')  # 浅灰色背景
legend.get_frame().set_edgecolor('#404040')  # 深灰色边框

plt.tight_layout()
plt.show()