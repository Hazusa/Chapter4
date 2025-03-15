import logging
import os
from datetime import datetime
import numpy as np
import yaml
import torch
from models.light_tst import LightweightTSClassifier
from utils.data_loader import get_loaders
from utils.metrics import calculate_metrics
from models.model import InformerLite,AutoformerTiny,LiteTST,TS_CNN,TS_LSTM,TS_BiLSTM,TS_BERT,TS_RoBERTa
#from data.raw.trans import npy_data, labels


# 加载配置
with open("configs/config_bert.yaml", encoding='utf-8') as f:
    config = yaml.safe_load(f)

config["lr"] = float(config["lr"])
# 初始化]

'''
model = TS_RoBERTa(
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
)
'''

model = TS_BERT(
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
)


# 获取当前时间并格式化
current_time = datetime.now().strftime("%Y%m%d%H%M%S")
# 获取当前训练模型的名称
model_name = model.__class__.__name__
# 配置日志记录
log_filename = os.path.join('./experiments/logs', f"{current_time}_{model_name}.log")
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
model_dir = './experiments/models'


#optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
optimizer = torch.optim.AdamW(model.parameters(),
                             lr=config["lr"],
                             weight_decay=0.001)  # 明确添加权重衰减
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=3,
    factor=0.5,
    min_lr=1e-6  # 添加最小学习率限制
)
criterion = torch.nn.CrossEntropyLoss()


# 数据加载
#data = np.load('dataset.npy')
#labels = np.load('labels.npy')
data = np.load('new_dataset2.npy')
labels = np.load('new_labels2.npy')
print("Data stats - min:", np.nanmin(data), "max:", np.nanmax(data), "mean:", np.nanmean(data))
print("Labels unique values:", np.unique(labels))
data = np.nan_to_num(data, nan=0.0, posinf=1e5, neginf=-1e5)  # 处理异常值

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

indices = np.arange(len(data))
np.random.shuffle(indices)
shuffled_data = data[indices]
shuffled_labels = labels[indices]

# 修改后的数据加载（保持原有loader获取方式）
train_loader, val_loader = get_loaders(shuffled_data, shuffled_labels, batch_size=16, split_ratio=0.8)

# 初始化最佳准确率
best_acc = 0
# 训练循环
for epoch in range(config["epochs"]):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    # 添加参数初始化检查（仅第一次迭代时打印）
    if epoch == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Initialized {name} with mean {param.data.mean():.4f}, std {param.data.std():.4f}")

    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()

        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch {epoch}: Train Loss={train_loss / len(train_loader):.4f}, Train Acc={train_acc:.2f}%")

# 验证
    model.eval()
    with torch.no_grad():
        val_preds, val_labels = [], []
        for x, y in val_loader:
            outputs = model(x)
            val_preds.extend(torch.argmax(outputs, 1).tolist())
            val_labels.extend(y.tolist())

    acc, precision, recall,f1= calculate_metrics(val_labels, val_preds,1)
    print(f"Epoch {epoch}: Val Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    if acc > best_acc and acc > 0.6:
        best_acc = acc
        model_filename = os.path.join(model_dir, f"{current_time}_{model_name}.pth")
        torch.save(model.state_dict(), model_filename)
        logging.info(f"Best model saved at epoch {epoch} with accuracy {best_acc:.4f}")
    scheduler.step(acc)