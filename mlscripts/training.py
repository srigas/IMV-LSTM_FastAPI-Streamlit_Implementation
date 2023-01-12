import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import os

from c_utils import train_model
from network import IMVLSTM

def training(batch_size,n_units,epochs,patience):

    # Get the parent directory, i.e. the one containing all subfolders
    parent_dir = os.path.dirname(os.getcwd())

    X_train, X_val = np.load(os.path.join(parent_dir,"results/"+"X_train.npy")), np.load(os.path.join(parent_dir,"results/"+"X_val.npy"))
    y_train, y_val = np.load(os.path.join(parent_dir,"results/"+"y_train.npy")), np.load(os.path.join(parent_dir,"results/"+"y_val.npy"))

    # Cast the arrays into tensors, to be fed into the Networks
    X_train_tens, X_val_tens = torch.Tensor(X_train), torch.Tensor(X_val)
    y_train_tens, y_val_tens = torch.Tensor(y_train), torch.Tensor(y_val)

    # Load the data into DataLoaders in batches
    train_loader = DataLoader(TensorDataset(X_train_tens, y_train_tens),batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tens, y_val_tens),batch_size=batch_size, shuffle=False)
                                                                                        
    # results path                       
    path = os.path.join(parent_dir,"results/model.pt")
    
    # Retrieve the value for the predict_window variable
    predict_window = 1 if len(y_train.shape) == 1 else y_train.shape[1]
                                                                                        
    # Initialize the model
    model = IMVLSTM(input_dim=X_train_tens.shape[2], output_dim=predict_window, n_units=n_units)

    # Setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0)
                                                                                        
    # Setup an epoch scheduler, to tamper with the learning rate as epochs increase
    e_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma = 0.9)

    model, optimizer, t_epochs, mse_t, mse_val = train_model(model=model,optimizer=optimizer,e_scheduler=e_scheduler,
                                                             train_loader=train_loader,val_loader=val_loader,epochs=epochs,
                                                             patience=patience,path=path)
    
    return t_epochs, round(mse_t,5), round(mse_val,5)