# Train script


import IPython
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from numpy import random
import time



def func(x):
    return 1 + (1/4000)*sum((x+7)**2) - np.prod(np.cos((x+7) / range(1, len(x)+1, 1)))


#Create Artificial Data
obs=int(1e6)#set number of observations
vars=100#set number of features
X=20*(random.rand(obs,vars).astype(np.float32)-1/2)
y=np.zeros(obs).astype(np.float32)
for i in range(obs):
    y[i]=func(X[i,:])

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


class torch_set_dataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_dataset = torch_set_dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = torch_set_dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_dataset = torch_set_dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())


EPOCHS = 2
BATCH_SIZE = int(1e2)
LEARNING_RATE = 0.0001
NUM_FEATURES =X_train.shape[1]
N_NEURONS=int(2000)


train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


class MultipleRegression(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()
        self.layer_1 = nn.Linear(num_features, N_NEURONS)
        self.layer_2 = nn.Linear(N_NEURONS, N_NEURONS)
        self.layer_3 = nn.Linear(N_NEURONS, N_NEURONS)
        self.layer_4 = nn.Linear(N_NEURONS, N_NEURONS)
        self.layer_5 = nn.Linear(N_NEURONS, N_NEURONS)
        self.layer_out = nn.Linear(N_NEURONS, 1)
        self.relu = nn.ReLU()
    def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.relu(self.layer_2(x))
            x = self.relu(self.layer_3(x))
            x = self.relu(self.layer_4(x))
            x = self.relu(self.layer_5(x))
            x = self.layer_out(x)
            return (x)
    def predict(self, test_inputs):
            x = self.relu(self.layer_1(test_inputs))
            x = self.relu(self.layer_2(x))
            x = self.relu(self.layer_3(x))
            x = self.relu(self.layer_4(x))
            x = self.relu(self.layer_5(x))
            x = self.layer_out(x)
            return (x)





model = MultipleRegression(NUM_FEATURES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
  
  
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
gpu_name=torch.cuda.get_device_name(0)
print(gpu_name)


loss_stats = {
    'train': [],
    "val": []
}


print("Begin training.")
t1=time.time()
for e in range(1, EPOCHS+1):
    # TRAINING
    train_epoch_loss = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch)
        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()  
    # VALIDATION    
    with torch.no_grad(): 
        val_epoch_loss = 0
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device) 
            y_val_pred = model(X_val_batch)            
            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
            val_epoch_loss += val_loss.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))                              
    #print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')
    print(e,train_epoch_loss/len(train_loader),val_epoch_loss/len(val_loader))

t2=time.time()
t2_t1 = "%.1f" % (t2-t1)
print("training got ",t2_t1," seconds")


y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_pred_list.append(y_test_pred.cpu().numpy())

y_pred_list = np.concatenate(y_pred_list, axis=0).flatten()
rmse = mean_squared_error(y_train, y_pred_list)**0.5
r_square = r2_score(y_train, y_pred_list)
print("Root Mean Squared Error :",rmse)
print("R^2 :",r_square)


