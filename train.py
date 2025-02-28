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
with open("configs/config.yaml", encoding='utf-8') as f:
    config = yaml.safe_load(f)

config["lr"] = float(config["lr"])
# 初始化]
'''model = LightweightTSClassifier(
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
    d_model=config["d_model"]
)'''

model = InformerLite(
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
    d_model=config["d_model"]
)

'''
model = AutoformerTiny(
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
    d_model=config["d_model"]
)
'''
'''
model = LiteTST(
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
    d_model=config["d_model"]
)
'''
'''
model = TS_RoBERTa(
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
)
'''

# 获取当前时间并格式化
current_time = datetime.now().strftime("%Y%m%d%H%M%S")
# 获取当前训练模型的名称
model_name = model.__class__.__name__
# 配置日志记录
log_filename = os.path.join('./experiments/logs', f"{current_time}_{model_name}.log")
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
model_dir = './experiments/models'


optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
criterion = torch.nn.CrossEntropyLoss()

# 数据加载
data = np.load('dataset.npy')
labels = np.load('labels.npy')
train_loader, val_loader = get_loaders(data, labels, batch_size=16, split_ratio=0.8)

# 初始化最佳准确率
best_acc = 0
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

    acc, precision, recall,f1= calculate_metrics(val_labels, val_preds,1)
    print(f"Epoch {epoch}: Val Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    if acc > best_acc and acc > 0.6:
        best_acc = acc
        model_filename = os.path.join(model_dir, f"{current_time}_{model_name}.pth")
        torch.save(model.state_dict(), model_filename)
        logging.info(f"Best model saved at epoch {epoch} with accuracy {best_acc:.4f}")
