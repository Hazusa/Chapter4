import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TSDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = torch.FloatTensor(data)  # 原始数据形状 [N,5,15]
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        '''
        if self.augment:
            # 数据增强：随机添加噪声
            if torch.rand(1) < 0.5:
                x += torch.randn_like(x) * 0.01
        return x, y
        '''
        return x, y

def get_loaders(data, labels, batch_size=16, split_ratio=0.8):
    indices = np.random.permutation(len(data))
    split = int(len(data) * split_ratio)

    train_set = TSDataset(data[indices[:split]], labels[indices[:split]], augment=True)
    val_set = TSDataset(data[indices[split:]], labels[indices[split:]], augment=False)

    return (
        DataLoader(train_set, batch_size, shuffle=True),
        DataLoader(val_set, batch_size)
    )