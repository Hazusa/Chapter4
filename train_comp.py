import yaml
import torch

from train import f1, acc
from utils.data_loader import get_loaders
from utils.metrics import calculate_metrics
from models.model_factory import get_model

# 加载对比配置
with open("configs/compare.yaml") as f:
    config = yaml.safe_load(f)

# 遍历所有模型
for model_name in config["models"]:
    print(f"\n=== Training Model: {model_name} ===")

    # 初始化模型
    model_config = {**config["common_params"], ** config["model_specific_params"][model_name]}
    model = get_model(model_name, model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 数据加载（确保所有模型使用相同划分）
    train_loader, val_loader = get_loaders(..., seed=42)  # 固定随机种子

    # 训练循环（与之前train.py类似）
    best_f1 = 0
    #for epoch in range(model_config["epochs"]):
        # ... 原有训练代码 ...

    # 保存最佳模型和指标
    torch.save(model.state_dict(), f"experiments/best_{model_name}.pth")
    with open("experiments/results.csv", "a") as f:
        f.write(f"{model_name},{acc:.4f},{f1:.4f}\n")