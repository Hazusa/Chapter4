import numpy as np
import yaml
import torch
from models.light_tst import LightweightTSClassifier
from utils.data_loader import get_loaders
from utils.metrics import calculate_metrics
from models.model import InformerLite,AutoformerTiny,LiteTST,TS_CNN,TS_LSTM,TS_BiLSTM,TS_BERT,TS_RoBERTa
#from data.raw.trans import npy_data, labels

with open("configs/config.yaml", encoding='utf-8') as f:
    config = yaml.safe_load(f)

config["lr"] = float(config["lr"])

def initialize_model():
    # 初始化模型
    model = AutoformerTiny(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        d_model=config["d_model"]
    )
    return model

# 数据加载
data = np.load('dataset.npy')
labels = np.load('labels.npy')

# 训练循环
for _ in range(2):  # 假设进行两次训练
    # 初始化模型和优化器
    model = initialize_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, val_loader = get_loaders(data, labels, batch_size=16, split_ratio=0.8)

    # 训练循环
    for epoch in range(config["epochs"]):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_preds, val_labels = [], []
            for x, y in val_loader:
                outputs = model(x)
                val_preds.extend(torch.argmax(outputs, 1).tolist())
                val_labels.extend(y.tolist())

        acc, f1 = calculate_metrics(val_labels, val_preds)
        print(f"Epoch {epoch}: Val Acc={acc:.4f}, F1={f1:.4f}")