# MIMIC3-BenchMark의 데이터리더 결과를 DataLoader로 옮겨주는 Script
import torch
from torch.utils.data import Dataset,DataLoader


def readerToDataLoader(data):
    dataset = MyDataset(data=data[0],labels=data[1])
    #32
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    return dataloader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return sample, label
