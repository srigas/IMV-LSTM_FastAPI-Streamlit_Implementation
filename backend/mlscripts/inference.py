import pickle

import pandas as pd
import numpy as np

import torch

from mlscripts import CustomMinMaxScaler, infer
from mlscripts.network import IMVLSTM

def infer_result(data):

    # Load the data scalers parameters and the feature names
    with open("results/preparams.pkl", 'rb') as f:
        paramsdict = pickle.load(f)

    X_max, X_min, y_min, y_max = paramsdict['X_max'], paramsdict['X_min'], paramsdict['y_min'], paramsdict['y_max']

    window_size = X_min.shape[0]

    # LSTM model path                       
    path = "results/model.pt"

    # calculate model parameters without needing to input
    # them explicitly every time
    dummy = torch.load(path)['U_j']
    input_dim = dummy.shape[0]
    n_units = dummy.shape[2]
    predict_window = 1 if isinstance(y_min, float) else len(y_min)

    # pre-process given csv
    wind_cats = ['NE', 'NW', 'SE', 'cv']
    dummies = pd.get_dummies(data, prefix='', prefix_sep='')
    dummies = dummies.T.reindex(wind_cats).T.fillna(0)
    data = pd.concat([data, dummies], axis=1)
    data = data.rename(columns={"pm2.5": "Auto-regressive", "DEWP": "Dew Point", "TEMP": "Temperature",
                            "PRES": "Pressure", "Iws": "Wind Speed", "Is": "Snow", "Ir": "Rain",
                            "NE": "NE Wind", "NW": "NW Wind", "SE": "SE Wind", "cv": "SW Wind"})
    data.drop(['cbwd'], axis=1, inplace=True)

    # obtain the feature matrix
    X = np.zeros((1,window_size, len(data.columns)))
    X[0,:,:] = data.iloc[:,:]

    # transform the feature
    y = y_max # placeholder value for the CustomMinMaxScaler
    X_inf, _ = CustomMinMaxScaler(X_min,X_max,y_min,y_max,X,y)
    X_inf_tens = torch.Tensor(X_inf)

    # reload the model
    model = IMVLSTM(input_dim=input_dim, output_dim=predict_window, n_units=n_units)
    model.load_state_dict(torch.load(path))

    results = infer(model,X_inf_tens,paramsdict)
    
    return [round(result,5) for result in results]