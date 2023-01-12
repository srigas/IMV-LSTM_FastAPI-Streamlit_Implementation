import os

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import pickle

from preprocessing import preprocess_df
from network import IMVLSTM
from c_utils import CustomMinMaxScaler, eval_model

def test_results(data,eval_batch_size):
    # results path                       
    path = os.path.join(os.path.dirname(os.getcwd()),"results/model.pt")

    # Load the data scalers parameters and the feature names
    with open(os.path.join(os.path.dirname(os.getcwd()),"results/preparams.pkl"), 'rb') as f:
        paramsdict = pickle.load(f)

    X_max, X_min, y_min, y_max = paramsdict['X_max'], paramsdict['X_min'], paramsdict['y_min'], paramsdict['y_max']

    # calculate model parameters without needing to input
    # them explicitly every time
    dummy = torch.load(path)['U_j']
    input_dim = dummy.shape[0]
    n_units = dummy.shape[2]
    predict_window = 1 if isinstance(y_min, float) else len(y_min)

    # reload the model
    model = IMVLSTM(input_dim=input_dim, output_dim=predict_window, n_units=n_units)
    model.load_state_dict(torch.load(path))

    # preprocess the data for evaluation
    window_size = X_min.shape[0]
    X_test, y_test = preprocess_df(data=data, window_size=window_size, predict_window=predict_window, val_percent=0.1, test=True)

    # Scale them using the train scalers
    X_test, y_test = CustomMinMaxScaler(X_min,X_max,y_min,y_max,X_test,y_test)

    # Tensorization
    X_test_tens = torch.Tensor(X_test)
    y_test_tens = torch.Tensor(y_test)

    # DataLoader
    test_loader = DataLoader(TensorDataset(X_test_tens, y_test_tens),batch_size=eval_batch_size,shuffle=False)
    
    print("Initiating the evaluation of the model.")
    figs = eval_model(model,test_loader,paramsdict,predict_window,window_size)
    
    return figs