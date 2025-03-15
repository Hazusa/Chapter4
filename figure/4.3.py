import matplotlib.pyplot as plt
import numpy as np

# 数据配置（与表格4.6严格对齐）
sample_ratios = ['原始', '6:1', '5:1', '4:1', '3:1', '2:1', '1:1']
metrics = {
    'Accuracy':   [0.8522, 0.8686, 0.8878, 0.8911, 0.8856, 0.8733, 0.8556],
    'Precision':  [0.8583, 0.8578, 0.8758, 0.9103, 0.8847, 0.8527, 0.8478],
    'Recall':     [0.8492, 0.8679, 0.8746, 0.8981, 0.8781, 0.8785, 0.8564],
    'F1':         [0.8561, 0.8645, 0.8826, 0.8841, 0.8793, 0.8628, 0.8485],
    #'UAR':        [0.8178, 0.8219, 0.8782, 0.8841, 0.8366, 0.8122],
    #'MCC':        [0.6439, 0.6702, 0.6721, 0.7106, 0.6826, 0.6580]
}

# 可视化配置
plt.rcParams["font.family"] = "SimSun"  # 中文字体
colors = ['#2F4F4F', '#DC143C', '#228B22', '#1E90FF']#, '#9370DB', '#FF8C00']
markers = ['o', 's', '^', 'D']#, 'v', 'p']
linestyles = ['-', '--', '-.', ':']#, '-', '--']

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(sample_ratios))

# 绘制各指标曲线
for idx, (metric, values) in enumerate(metrics.items()):
    ax.plot(x, values,
            label=metric,
            color=colors[idx],
            linestyle=linestyles[idx],
            linewidth=2,
            marker=markers[idx],
            markersize=8,
            markeredgecolor='w')

# 坐标轴优化
ax.set_xticks(x)
ax.set_xticklabels(sample_ratios, fontsize=12)
ax.set_ylim(0.8, 0.925)
ax.set_yticks(np.arange(0.8, 0.925, 0.025))
ax.tick_params(axis='y', labelsize=12)

# 辅助元素
ax.grid(True, linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 标签和标题
ax.set_xlabel("样本分布比例", fontsize=14, labelpad=10)
ax.set_ylabel("指标结果", fontsize=14, labelpad=10)
ax.set_title("ADASYN扩增样本后的模型的性能对比",
            fontsize=16, pad=20)

# 图例优化
legend = ax.legend(
    loc='upper right',           # 将位置改为右上角
    bbox_to_anchor=(1.02, 1),    # 微调位置避免遮挡曲线
    ncol=2,                      # 改为2列布局
    frameon=True,                # 显示边框
    fancybox=True,               # 圆角边框
    shadow=True,                 # 添加阴影
    fontsize=10,                 # 缩小字体
    borderpad=0.8,               # 调整边距
    title='性能指标',            # 添加图例标题
    title_fontsize=12
)

# 调整画布留白
plt.subplots_adjust(right=0.8)  # 为右侧图例预留空间

# 设置图例背景样式
legend.get_frame().set_facecolor('#F5F5F5')  # 浅灰色背景
legend.get_frame().set_edgecolor('#404040')  # 深灰色边框
'''
# 数值标签（可选）
for ratio in x:
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        value = metrics[metric][ratio]
        ax.text(ratio, value+0.01, f'{value:.4f}',
               ha='center', va='bottom', fontsize=9)
'''
plt.tight_layout()
plt.show()