
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model.model import MultipleRegression
from build_dataset import TorchSetDataset, simulate_data, split_train_test

def train(model, data_loader, optimizer):
    # put model in train mode
    model.train()

    train_epoch_loss = 0

    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch)
        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()  
        
    return train_epoch_loss

def validate(model, data_loader, optimizer):
    # this is validation - no gradient calculation
    with torch.no_grad(): 

        val_epoch_loss = 0
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device) 
            y_val_pred = model(X_val_batch)            
            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
            val_epoch_loss += val_loss.item()
    return val_epoch_loss


def test(model, data_loader, optimizer):
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
  



if __name__ == "__main__":
    epochs = 5
    batch_size = 100
    learning_rate = 0.0001

    num_neurons = 2000
    
    # Generate artificial data
    X, y = simulate_data()

    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_test(X, y)

    # Get number of features
    num_features = X_train.shape[1]

    # Get datasets
    train_dataset = TorchSetDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = TorchSetDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    test_dataset = TorchSetDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    # Load datasets
    train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size)

    # Get model we defined
    model = MultipleRegression(num_features, num_neurons)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) 

    # Move the model to device
    model.to(device)
    criterion = nn.MSELoss()

    # Use  Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_stats = {
        'train': [],
        "val": []
        }

    t1=time.time()
    for epoch in range(epochs):
        train_epoch_loss = train(model, train_loader, optimizer)
        val_epoch_loss = train(model, val_loader, optimizer)
        
        normed_train_epoch_loss = train_epoch_loss/len(train_loader)
        normed_val_epoch_loss = val_epoch_loss/len(val_loader)

        loss_stats['train'].append(normed_train_epoch_loss)
        loss_stats['val'].append(normed_val_epoch_loss)  
        
        print(f"Epoch={epoch}, Train_epoch_loss={normed_train_epoch_loss}, \ 
              Validation_epoch_loss={normed_val_epoch_loss}") 

    t2=time.time()
    t2_t1 = "%.1f" % (t2-t1)
    print("training got ",t2_t1," seconds")




    test(model, test_loader, optimizer)
