import os
import pickle
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def data_preprocess(file: str):
    data = pd.read_csv(f'data/{file}')
    data = data.iloc[:, 1:]
    return data

class MakeDATA(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        data = np.asarray(data, dtype= np.float32)
        data = data[::-1]
        seq_data = []
        for i in range(len(data) - seq_len + 1):
            x = data[i : i + seq_len]
            seq_data.append(x)
        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])
        self.samples = np.asarray(self.samples, dtype = np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def LoadData(file:str, seq_len:int):
    data = data_preprocess(file)
    data = MakeDATA(data, seq_len)
    train_data, test_data = train_test_split(data, train_size = 0.8, random_state = 2021)
    print(f'{file} data loaded with sequence {seq_len}')
    return train_data, test_data
