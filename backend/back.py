import io
import os
import sys

import pandas as pd

# Get the parent directory, i.e. the one containing all subfolders
parent_dir = os.path.dirname(os.getcwd())

# adding the mlscripts to the system path
sys.path.insert(0, os.path.join(parent_dir,"mlscripts"))

from preprocessing import preprocess_df
from training import training
from evaluation import test_results

from starlette.responses import PlainTextResponse, JSONResponse

from fastapi import FastAPI, File
#import zipfile

from pydantic import BaseModel

# Define the app
app = FastAPI(title = "IMV-LSTM FastAPI-Streamlit implementation", version = "1.0")

# Define a class to setup the request of the first post process
class requestPreproc(BaseModel):
    window_size: int = 10
    predict_window: int = 1
    val_percent: float = 0.3

# A post process for the data preprocessing given the train dataframe
@app.post("/preprocessing")
def do_preprocess(item: requestPreproc):
    
    # Load train dataframe
    df = pd.read_csv(os.path.join(parent_dir,"data/train_data.csv"))

    stats = preprocess_df(data=df, window_size=item.window_size, predict_window=item.predict_window,
                          val_percent=item.val_percent, test=False)
    
    return PlainTextResponse(stats)

# Define a class to setup the request of the second post process
class requestTrain(BaseModel):
    batch_size: int = 32
    n_units: int = 128
    epochs: int = 100
    patience: int = 12

# A post process for the model training
@app.post("/training")
def do_train(item: requestTrain):
    
    t_epochs, mse_t, mse_val = training(batch_size=item.batch_size,n_units=item.n_units,
                                        epochs=item.epochs,patience=item.patience)
    r_dict = {"epochs" : t_epochs, "mse train" : mse_t, "mse val" : mse_val}
    
    return JSONResponse(r_dict)

# Here we do not define a class to setup the request for the
# third post process, because it involves a mix between json payload
# and a file. This will be handled differently during the request, see
# the front.py file. However, in this case we need a class to define
# the response of the third post process.
class responseEval(BaseModel):
    image: bytes
    # This is a needed workaround, due to the bug reported here:
    # https://github.com/pydantic/pydantic/issues/3756
    class Config:
        json_encoders = { bytes: lambda bs: "".join(map(chr, bs)) }

# A post process for the model evaluation
@app.post("/evaluation", response_model=list[responseEval])
def do_eval(file: bytes = File(...),
            eval_batch_size: int = 32):
    
    # First read the contents of the uploaded file
    test_df = pd.read_csv(io.BytesIO(file))
    
    # Using this df, proceed with the model evaluation
    figs = test_results(data=test_df,eval_batch_size=eval_batch_size)
    bytefigs = []
    for fig in figs:
        fig_byte_array = io.BytesIO()
        fig.save(fig_byte_array, format="PNG")
        bytefigs.append(fig_byte_array.getvalue())
        
    return [{"image" : fig} for fig in bytefigs]

# TODO: Post process for inference
# Although it is straightforward, since it's just
# slightly changing the eval routine