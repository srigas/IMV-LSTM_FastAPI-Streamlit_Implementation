import pandas as pd
import numpy as np
import os

from c_utils import CustomMinMaxScaler

import pickle

def preprocess_df(data, window_size, predict_window, val_percent, test):
    """
    window_size: the number of time-steps used in order to predict new values
    predict_window: the number of forward timestemps to attempt to predict
    """
    print("Initiating pre-processing of dataset")
    df = data.copy()
    df.drop(['No'], axis=1, inplace=True)
    # one-hot encoding
    wind_dir_df = pd.get_dummies(df['cbwd'])
    # merge dfs
    df = pd.concat([df, wind_dir_df], axis=1)
    df = df.rename(columns={"pm2.5": "Auto-regressive", "DEWP": "Dew Point", "TEMP": "Temperature",
                            "PRES": "Pressure", "Iws": "Wind Speed", "Is": "Snow", "Ir": "Rain",
                            "NE": "NE Wind", "NW": "NW Wind", "SE": "SE Wind", "cv": "SW Wind"})
    # drop original feature for wind direction
    df.drop(['cbwd'], axis=1, inplace=True)
    
    # Fill missing values in such a way to preserve continuity with previous/next values
    df['Auto-regressive'] = df['Auto-regressive'].fillna(method='ffill').fillna(method='bfill')
    
    # a list of the features to be used, i.e. all features except for the date info
    # here we simply omit the year, month, day, hour columns, i.e. timestamps
    feats = list(df.columns[4:])
    
    # organize the data in order to respect the window_size
    X = np.zeros((len(df), window_size, len(feats)))
    for i, name in enumerate(feats):
        for j in range(window_size):
            X[:, j, i] = df[name].shift(window_size - j - 1).fillna(method='bfill')
            
    # Define target values
    if predict_window == 1:
        y = df['Auto-regressive'].shift(-1).fillna(method='ffill').values
    else:
        y = np.zeros((len(df), predict_window))
        for j in range(predict_window):
            y[:,j] = df['Auto-regressive'].shift(-1-j).fillna(method='ffill').values
    
    if not test:
        print("Proceeding with data splitting.")
        # Split the data into train-validation
        chidx = int((1-val_percent)*(len(df)))
        stats = f"The total number of entries is {len(df)}.\nOut of this total, {chidx} are used for training and {len(df)-chidx} are used for validation."
        print(stats)

        X_train, X_val = X[:chidx], X[chidx:]
        y_train, y_val = y[:chidx], y[chidx:]

        print("Data split. Proceeding with scaling.")

        # Only the training set is used for the transformation
        X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
        y_min, y_max = y_train.min(axis=0), y_train.max(axis=0)

        X_train, y_train = CustomMinMaxScaler(X_min,X_max,y_min,y_max,X_train,y_train)
        X_val, y_val = CustomMinMaxScaler(X_min,X_max,y_min,y_max,X_val,y_val)

        print(f"Data scaled. Pre-processing complete.")

        # Get the parent directory, i.e. the one containing all subfolders
        parent_dir = os.path.dirname(os.getcwd())

        names = ["X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy"]
        for name, matrix in zip(names,[X_train, X_val, y_train, y_val]):
            np.save(os.path.join(parent_dir,"results/"+name), matrix)

        # Also save the scaler's parameters, since they will need to be used in
        # the inverse transformation during inference, along with the feature names
        paramsdict = {'X_min' : X_min, 'X_max' : X_max, 'y_min' : y_min, 'y_max' : y_max, 'feats' : feats}

        with open(os.path.join(parent_dir,"results/"+'preparams.pkl'), 'wb') as f:
            pickle.dump(paramsdict, f)
            
        return stats
    else:
        print(f"Pre-processing complete.")
        return X, y