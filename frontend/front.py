import io

import requests
from PIL import Image

import pandas as pd

import streamlit as st

from requests_toolbelt.multipart.encoder import MultipartEncoder

import json

# interact with FastAPI endpoint
base_url = "http://127.0.0.1:8000/"
#base_url = "http://fastapi:8000/"

urls = ["preprocessing","training","evaluation"]

# ----------------------------

# How to open byte image through pil:
#bytimg = bytefigs[0]
#Image.open(io.BytesIO(bytimg)).convert("RGB")

# construct UI layout
st.title("IMV-LSTM FastAPI-Streamlit implementation")

st.write(
    """Write some explanation"""
)  # description and instructions

### !!!
## !!!
### !!!

# PART 1

if st.button("Preprocess!"):
    
    window_size = 10
    predict_window = 1
    val_percent = 0.2
    
    datadict = {"window_size" : window_size, "predict_window" : predict_window, "val_percent" : val_percent}
    
    # post request for pre-processing
    req1 = requests.post(base_url+"preprocessing", json=datadict, headers={"Content-Type": "application/json"})
    
    print(req1.content.decode())

# PART 2

if st.button("Train!"):
    
    batch_size = 32
    n_units = 128
    epochs = 1
    patience = 12
    
    datadict = {"batch_size" : batch_size, "n_units" : n_units, "epochs" : epochs, "patience" : patience}
    
    # post request for pre-processing
    req2 = requests.post(base_url+"training", json=datadict, headers={"Content-Type": "application/json"})
    
    print(req2.content.decode())
    
# PART 3

input_file = st.file_uploader("Upload a .csv file for evaluation.")

if st.button("Train model!"):

    col1, col2, col3 = st.columns(3)

    if input_file:
        
        eval_batch_size = 32 #####
        
        # Define a multipartencoder to handle large files, and also take care of the int parameter
        m = MultipartEncoder(fields={"file": ("filename", input_file, "text/csv"),
                                     "eval_batch_size" : f"{eval_batch_size}"})

        # post request for evaluation
        req3 = requests.post(base_url+"evaluation", data=m, headers={"Content-Type": m.content_type})
        
        # The response comes in the form of bytes, this is done to retrieve its list structure
        jsonresp = json.loads(req3.content)
        figs = []
        
        for item in jsonresp:
            # turn the string that was returned from the response back into a byte
            byteimg = bytes(item["image"], encoding="raw_unicode_escape")
            # Append as image items
            figs.append(Image.open(io.BytesIO(byteimg)).convert("RGB"))
        
        col1.header("results")
        col1.image(figs[0], use_column_width=True)
        col2.header("vwtis")
        col2.image(figs[1], use_column_width=True)
        col3.header("vis")
        col3.image(figs[2], use_column_width=True)

    else:
        # handle case with no image
        st.write("You did not provide a .csv file for evaluation. Please read the documentation and retry.")