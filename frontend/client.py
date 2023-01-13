import io
import json
import requests
from PIL import Image

import pandas as pd

import streamlit as st

from requests_toolbelt.multipart.encoder import MultipartEncoder

base_url = "http://backendcon:8000/" # <- use the container name for the backend in the docker-compose.yml file
#base_url = "http://localhost:8000/" # <- for local running

# ----------------------------

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title("IMV-LSTM model implementation using FastAPI & Streamlit")

st.write("""A "full-stack" implementation of the IMV-LSTM neural network presented in [this](https://arxiv.org/pdf/1905.12034.pdf) paper. This falls under the broader category of Explainable AI, where not only is the LSTM used to make predictions on time-series data, but we can also gain insights on why it makes said predictions, i.e. which features are more influential for them, as well as how this influence (importance) varies over time.""")
st.write("""The server side is built on FastAPI, which generates four endpoints: for preprocessing, training, evaluation and inference. This Streamlit app corresponds to the client side, where requests can be sent to each of these endpoints. Note that every time we wish to perform a request, we do not need to perform all previous requests anew (this is why the pipeline was split into these four components). For example, if the data is preprocessed once, then we can proceed to train the model using any number of different configurations for the training parameters, without having to preprocess the data every time.""")

# the sidebar is used for parameter setting
with st.sidebar:
    st.markdown("""# Preprocessing parameters""")
    window_size = st.slider('Set the number of previous data-points that compose the feature matrix used for prediction.', 1, 100, 10)
    predict_window = st.slider('Set the number of data-points to be predicted.', 1, 20, 1)
    val_percent = st.slider('Set the percentage of the training dataset to be used for validation.', 0.0, 0.5, 0.3)

    st.markdown("""# Training parameters""")
    batch_size = st.slider('Set the size of batches for training.', 1, 256, 32)
    n_units = st.slider('Set the number of LSTM units for the LSTM model.', 10, 512, 128)
    epochs = st.slider('Set the number of maximum epochs for model training.', 1, 1000, 100)
    patience = -1
    if val_percent > 0.0:
        earlystop = st.radio('Enable Early Stopping?',['Enable', 'Disable'])
        if earlystop == 'Enable':
            patience = st.slider('Set the patience limit for the early stopping mechanism.', 1, 100, 12)

    st.markdown("""# Evaluation parameters""")
    eval_batch_size = st.slider('Set the size of batches for evaluation.', 1, 256, 32)

# PART 1

markdown_part_1 = """### Preprocessing
The model is trained using a subset of the [PM2.5 Beijing Data](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) dataset. This first part of the pipeline corresponds to cleaning the data properly, preprocessing (for example, filling missing values) and scaling them before splitting them to be used for training and validation. The features that are eventually used for training are: `pm2.5`, `DEWP`, `TEMP`, `PRES`, `cbwd`, `Iws`, `Is`, `Ir`. Once you set the preprocessing parameters using the sidebar's sliders, click on the following button to perform the data preprocessing."""

st.markdown(markdown_part_1)

if st.button("Preprocess data"):
    
    # payload for preprocessing
    datadict = {"window_size" : window_size, "predict_window" : predict_window, "val_percent" : val_percent}
    
    # post request for preprocessing
    req1 = requests.post(base_url+"preprocessing", json=datadict, headers={"Content-Type": "application/json"})
    
    st.write(req1.content.decode())

# PART 2

markdown_part_2 = """### Training
Once the data have been preprocessed, we may proceed with the model's training. Press the following button to train the model, once having set the training parameters at the siderbar. Note that the early stopping mechanism is by default deactivated if no validation data are used (i.e. percentage = 0). Even if a validation dataset is used, there is still the option to deactivate the early stopping mechanism if required."""

st.markdown(markdown_part_2)

if st.button("Train model"):
    
    # payload for training
    datadict = {"batch_size" : batch_size, "n_units" : n_units, "epochs" : epochs, "patience" : patience}
    
    # post request for preprocessing
    req2 = requests.post(base_url+"training", json=datadict, headers={"Content-Type": "application/json"})
    
    reqresults = eval(req2.content.decode())
    epochs, mse_train, mse_val = reqresults["epochs"], reqresults["mse train"], reqresults["mse val"]
    st.write(f"Model training finished after an effective number of {epochs} epoch(s).")
    st.write(f"The final training MSE is equal to {mse_train}, while the final validation MSE is equal to {mse_val}.")
    
# PART 3

markdown_part_3 = """### Evaluation
After having trained a model with a satisfactory final MSE (preferably on validation data), we may evaluate the model on unseen data-points, using the evaluation endpoint. Note that the data need to be properly labeled, as this process is not equivalent to inference. The `backend/data` directory of the GitHub repository contains a [test_data.csv](https://github.com/srigas/IMV-LSTM_FastAPI-Streamlit_Implementation/blob/main/backend/data/test_data.csv) file which may be used for this purpose, but feel free to create your own custom dataset or use other publicly available ones, as long as they conform to the standards of the PM2.5 dataset when it comes to feature names, types, etc. Once a suitable .csv file has been uploaded, press the following button to proceed with model evaluation."""

st.markdown(markdown_part_3)

input_file = st.file_uploader("Upload a .csv file for evaluation.", type='csv')

if st.button("Evaluate model"):

    if input_file:
        
        # Define a multipartencoder to handle large files, and also take care of the int parameter
        # this corresponds to the payload for evaluation
        m = MultipartEncoder(fields={"file": ("placeholder", input_file, "text/csv"),
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
        
        st.write("The results of the evaluation are shown below. The first graph corresponds to the target values and predictions of the LSTM model.")
        
        # This is a simple hack to place the diagrams properly
        cola, colb, colc = st.columns([1,6,1])

        with cola:
            st.write("")

        with colb:
            st.image(figs[0])

        with colc:
            st.write("")

        st.write("The following graphs depict the importance of each feature for the predictions performed during evaluation. The graph on the left depicts the variable-wise temporal importance values of the features. Each row is normalized to 1 and depicts the influence of each time step within the window of each feature to the predictions. For example, if the value for feature `X` is 0.07 at time t-8 and 0.14 at time t-1, it means that the values of feature `X` are more influential on the prediction at times closer to the prediction than at earlier times. The graph on the right depicts the variable importance of each feature, i.e. the features that were overall more influential for the predictions compared to the ones that were not.")

        col1, col2 = st.columns(2)

        col1.image(figs[1], use_column_width=True)
        col2.image(figs[2], use_column_width=True)

    else:
        st.write("You did not provide a .csv file for evaluation. Please read the documentation and retry.")

# PART 4

markdown_part_4 = """### Inference
This part is perhaps the most important one when it comes to realistic applications, since the real solution to problems comes only after the model is deployed and exposed to new data. Using the following form, you can upload your own .csv file containing a header and a number of rows equal to the `window_size` parameter used during preprocessing and model training (for example, if `window_size=10`, the .csv file must contain a header and exactly 10 rows). The format must be identical to the one used in the PM2.5 dataset, excluding the features that are not used, like the form shown below:"""

st.markdown(markdown_part_4)

df = pd.DataFrame(columns=['pm2.5','DEWP','TEMP','PRES','cbwd','Iws','Is', 'Ir'])

st.dataframe(df)

markdown_part_5 = """The `backend/data` directory of the GitHub repository contains a [sample_request.csv](https://github.com/srigas/IMV-LSTM_FastAPI-Streamlit_Implementation/blob/main/backend/data/sample_request.csv) file which may be used for this purpose, for the default value of `window_size`."""

st.markdown(markdown_part_5)

input_file_inf = st.file_uploader("Upload a .csv file for inference.", type='csv')

if st.button("Infer results"):
    
    if input_file_inf:

        m = MultipartEncoder(fields={"file": ("placeholder", input_file_inf, "text/csv"),})

        # post request for evaluation
        req4 = requests.post(base_url+"inference", data=m, headers={"Content-Type": m.content_type})

        infresult = eval(req4.content.decode())["value"]
        if len(infresult) == 1:
            st.write(f"The pm2.5 prediction inferred using the input data is: {infresult[0]}")
        else:
            str_to_write = [f"‣ The prediction for t = {idx} is {prediction}.  \n " for idx, prediction in enumerate(infresult)]
            st.write("The following list contains the sequence of pm2.5 predictions inferred for the following hours using the input data:")
            st.write(''.join(str_to_write))
    else:
        st.write("You did not provide a .csv file for inference. Please read the documentation and retry.")