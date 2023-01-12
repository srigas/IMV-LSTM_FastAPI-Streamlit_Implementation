import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn

from sklearn.metrics import mean_squared_error, mean_absolute_error

from PIL import Image

# Scikit-learn's implementation is less straightforward for 3d data
def CustomMinMaxScaler(Xmin,Xmax,ymin,ymax,X,y):
    # 1e-10 is added in order to avoid zero divisions
    X_scaled = (X - Xmin)/(Xmax - Xmin + 1e-10)
    y_scaled = (y - ymin)/(ymax - ymin + 1e-10)
    return X_scaled, y_scaled
    
# Model training
def train_model(model,optimizer,e_scheduler,train_loader,val_loader,epochs,patience,path):
    
    loss_fn = nn.MSELoss()

    # Initialize these two parameters
    min_val_loss, stopping_ct = 9999, 0
    
    print(f'Initiating the training of the IMV-LSTM.')

    finalep = 0
    tot_samps_train, tot_samps_val = len(train_loader.dataset), len(val_loader.dataset)
    act_mse_tr, act_mse_val = 0, 0
    for epoch in range(epochs):
        finalep += 1
        train_loss = 0
        for x_batch, y_batch in train_loader:
            # Reset the gradients
            optimizer.zero_grad()
            # Predict the output, as well as the importance factors
            y_pred, VWTI, VI = model(x_batch)
            y_pred = y_pred.squeeze(1)
            # Calculate the Mean Square Error loss
            loss = loss_fn(y_pred, y_batch)
            # Backpropagation
            loss.backward()
            # Add the loss
            train_loss += loss.item()*x_batch.shape[0]
            optimizer.step()

        e_scheduler.step()
        
        print(f"Epoch: {epoch+1}/{epochs}.")
        MSE_train = train_loss/tot_samps_train
        print(f"Training Loss in terms of RMSE: {(MSE_train)**0.5:.4f}.\t\t Training Loss in terms of MSE: {MSE_train:.4f}.")
        
        # Proceed to validation
        if tot_samps_val != 0:
            with torch.no_grad():
                val_loss = 0
                for x_batch, y_batch in val_loader:
                    # Predict the output, as well as the importance factors
                    output, VWTI, VI = model(x_batch)
                    output = output.squeeze(1)
                    val_loss += loss_fn(output, y_batch).item()*x_batch.shape[0]
        
            MSE_val = val_loss/tot_samps_val
            print(f"Validation Loss in terms of RMSE: {(MSE_val)**0.5:.4f}.\t Validation Loss in terms of MSE: {MSE_val:.4f}.")
            
            if patience != -1:
                # Save the model only if val loss decreased
                if min_val_loss > val_loss**0.5:
                    min_val_loss = val_loss**0.5
                    torch.save(model.state_dict(), path)
                    act_mse_tr, act_mse_val = MSE_train, MSE_val
                    stopping_ct = 0
                # Otherwise increase the stopping counter by one
                else:
                    print(f'Validation Loss Increased. Early Stopping counter {stopping_ct+1}/{patience}.')
                    stopping_ct += 1

                if stopping_ct == patience:
                    finalep -= patience
                    print(f'Early Stopping counter reached patience limit.')
                    print('Terminating training and reverting back to last stable version.')
                    model.load_state_dict(torch.load(path))
                    break
    
    print(f'Training finished after an effective number of {finalep} epochs.')
    # Save the full model, not just the state dict
    torch.save(model.state_dict(), path)
    if patience == -1:
        return model, optimizer, finalep, MSE_train, MSE_val
    else:
        return model, optimizer, finalep, act_mse_tr, act_mse_val

def eval_model(model,test_loader,paramsdict,predict_window,window_size):
    # eval mode
    with torch.no_grad():
        preds, targets = [], []
        VWTI, VI = [], []
        for x_batch, y_batch in test_loader:
            x_batch = x_batch
            y_batch = y_batch
            output, a, b = model(x_batch)
            output = output.squeeze(1)
            preds.append(output.detach().numpy())
            targets.append(y_batch.detach().numpy())
            VWTI.append(a.detach().numpy())
            VI.append(b.detach().numpy())

    # Predicted values using the model
    preds = np.concatenate(preds)
    
    # Retrieve the y_max, y_min values
    y_min, y_max = paramsdict['y_min'], paramsdict['y_max']
    max_minus_min = y_max - y_min
    
    preds = preds*max_minus_min + y_min # inverse scaling
    
    # Actual targets
    targets = np.concatenate(targets)
    targets = targets*max_minus_min + y_min # inverse scaling
    
    # Scores
    RMSE = mean_squared_error(targets, preds)**0.5
    MAE = mean_absolute_error(targets, preds)
    
    print("Evaluation complete.")
    print(f"The RMSE for the model's performance on the test data is: {RMSE:.5f}")
    print(f"The MAE for the model's performance on the test data is: {MAE:.5f}")
    
    # Fixing the VIs and VWTIs
    VWTI, VI = np.concatenate(VWTI), np.concatenate(VI)
    VWTI, VI = VWTI.mean(axis=0), VI.mean(axis=0)
    VWTI, VI = VWTI[..., 0], VI[..., 0]
    
    # a correction for the plots in case of wide
    # prediction windows
    if predict_window == 1:
        actual_targs = targets
        actual_preds = preds
    else:
        actual_targs = targets[:,0]+[targets[-1,1]]
        actual_preds = preds[:,0]+[preds[-1,1]]
    
    # Start plotting the outputs
    sns.set(style = "darkgrid")
    
    #savepath = os.path.join(os.path.dirname(os.getcwd()), "results/")
    
    matplotlib.use('agg')
    
    figs = []
    
    # Output 1: Model performance, including scores
    sns.set(font_scale=1.3)
    fig = plt.figure(figsize=(15, 8))
    plt.plot(actual_targs, linewidth=0.8, label="Target Values", color=(0.83106498, 0.23844675, 0.30880431)) #"#d9cd8d"
    plt.plot(actual_preds, linewidth=0.8, label="Predicted Values", color=(0.19946175, 0.5289504 , 0.73910035)) # "#2a4d24"
    plt.ylabel('PM2.5 Predictions')
    plt.xlabel('Time')
    plt.xticks([])
    plt.title(f'Model evaluation - RMSE: {RMSE:.5f} MAE: {MAE:.5f}')
    plt.legend(loc='best')
    #plt.savefig(savepath+'final_score.png', bbox_inches='tight')
    fig.canvas.draw()
    figs.append(Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
    plt.close(fig)
    
    # Output 2: VWTIs heatmap diagram
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(12, 12))
    x_axis_labels = ["t-"+str(i) for i in np.arange(window_size, 0, -1)]
    y_axis_labels = paramsdict['feats']
    sns.heatmap(ax=ax, data=VWTI.transpose(1,0)[:,:window_size], linewidths=.5,
                cmap=sns.color_palette("coolwarm", as_cmap=True),
                xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    plt.yticks(rotation=0)
    ax.set_title("Variable-wise Temporal Importance values")
    ax.set_xlabel("Time step lag (interval of 1 hour)")
    #plt.savefig(savepath+'VWTIs.png', bbox_inches='tight')
    fig.canvas.draw()
    figs.append(Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
    plt.close(fig)
    
    # Output 3: Final VI values
    sns.set(font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(VI, labels=paramsdict['feats'], colors=plt.cm.Spectral(np.linspace(0, 1, 11)), autopct='%.0f%%')
    ax.set_title("Variable Importance for final results")
    #plt.savefig(savepath+'VIs.png', bbox_inches='tight')
    fig.canvas.draw()
    figs.append(Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
    plt.close(fig)
    
    return figs