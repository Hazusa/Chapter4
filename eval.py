import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from models.light_tst import LightweightTSClassifier
from utils.data_loader import TSDataset
from utils.metrics import calculate_metrics


def evaluate(model_path, test_data_path, test_label_path, result_dir="results"):
    """评估流程"""
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_data = np.load(test_data_path)  # 形状 [N, 5, 15]
    test_labels = np.load(test_label_path)
    test_dataset = TSDataset(test_data, test_labels, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = LightweightTSClassifier(
        input_dim=15,
        num_classes=len(np.unique(test_labels))
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 运行预测
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    # 计算指标
    acc, f1 = calculate_metrics(all_labels, all_preds)
    print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

    # 生成分类报告
    report = classification_report(
        all_labels, all_preds,
        target_names=[f"Class {i}" for i in range(model.num_classes)],
        digits=4
    )
    print("\nClassification Report:\n", report)

    # 保存结果
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/eval_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1-Score: {f1:.4f}\n\n")
        f.write(report)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(all_labels),
                yticklabels=np.unique(all_labels))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"{result_dir}/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--test_data", type=str, default="data/processed/test_data.npy",
                        help="Path to test data numpy file")
    parser.add_argument("--test_labels", type=str, default="data/processed/test_labels.npy",
                        help="Path to test labels numpy file")
    parser.add_argument("--result_dir", type=str, default="experiments/latest",
                        help="Directory to save evaluation results")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        test_data_path=args.test_data,
        test_label_path=args.test_labels,
        result_dir=args.result_dir
    )