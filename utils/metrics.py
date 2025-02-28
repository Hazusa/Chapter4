from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    cohen_kappa_score
)
import numpy as np

def calculate_metrics(true_labels, pred_labels, zero_division):
    """计算准确率和宏平均F1"""
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro")
    precision = precision_score(true_labels, pred_labels, average="macro", zero_division=zero_division)
    recall = recall_score(true_labels, pred_labels, average="macro")
    return acc, precision, recall, f1