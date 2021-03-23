import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler   
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

def simulate_data():

    def func(x):
        return 1 + (1/4000)*sum((x+7)**2) - np.prod(np.cos((x+7) / range(1, len(x)+1, 1)))

    observations = 1000
    features = 10

    X = 20*(np.random.rand(observations, features).astype(np.float32)-1/2)
    y = np.zeros(observations).astype(np.float32)
    for i in range(observations):
        y[i] = func(X[i,:])
    return X, y

def split_train_test(X, y):
    # Train - Test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    # Split train into train-val
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, random_state=None) 


    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)

    return X_train, y_train, X_val, y_val, X_test, y_test


class TorchSetDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)




