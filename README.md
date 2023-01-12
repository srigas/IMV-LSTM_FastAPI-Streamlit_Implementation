# IMV-LSTM FastAPI-Streamlit Implementation

## Table of Contents

+ [About](#about)
+ [Getting Started](#getting_started)
    + [Prerequisites](#prerequisites)
+ [Installing the requirements](#installing)
  + [Using the Makefile](#installing_makefile)
  + [Manual Installations](#installing_manually)
+ [Running the code](#run_locally)
    + [Execution Options](#execution_options)
        + [main.py](#src_main)
+ [Todo](#todo)
+ [License](#license)

## About <a name = "about"></a>

This repository corresponds to a "full-stack" implementation of the IMV-LSTM neural network presented in [this](https://arxiv.org/pdf/1905.12034.pdf) paper. This falls under the broader category of Explainable AI, where not only is the LSTM used to make predictions on time-series data, but we can also gain insights on why it makes said predictions, i.e. which features are more influential for them, as well as how this influence (importance) varies over time. The model is trained using a subset of the [PM2.5 Beijing Data](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) dataset and is subsequently evaluated using either the remaining data points (there is a [test_data.csv](/data/test_data.csv) file available in the [data](/data) folder of this repository), or custom-made testing datasets, as long as they are in the form of the PM2.5 dataset.

For the whole process of training and evaluating the model, three endpoints are generated using [FastAPI](https://fastapi.tiangolo.com/). The first is an endpoint for data pre-processing called `preprocessing`, with a callable function that takes care of pre-processing and transformations on the PM2.5 training dataset. The arguments are the following:

- `window_size`: integer with default value 10, corresponds to the number previous data-points that compose the feature matrix which is used for the prediction.
- `predict_window`: integer with default value 1, corresponds to the number of data-points to be predicted in the future. Obviously, higher values of this parameter lead to less accurate predictions (quantity over quality).
- `val_percent`: float between 0 and 0.5, corresponds to the percentage of the training dataset that is allocated for the validation of the model during its training. Setting this value to 0 automatically deactivates the Early Stopping mechanism that is utilized during training.

The second endpoint is called `training` and takes care of the model's training. The relevant function's arguments are:

- `batch_size`: integer with default value 32, corresponds to the size of the data batches during model training.
- `n_units`: integer with default value 128, corresponds to the number of LSTM units of the model.
- `epochs`: integer with default value 100, corresponds to the maximum number of training epochs (or exact number of training epochs if Early Stopping is deactivated).
- `patience`: integer with default value 12, corresponds to the number of patience epochs of the Early Stopping mechanism. If set to -1, it is deactivated.

Finally, the third endpoint is called `evaluation` and the relevant function call has the following two arguments:

- `eval_batch_size`: integer with default value 32, corresponds to the size of the data batches during model evaluation.
- `file`: a .csv file containing the testing data, to be uploaded so that the model can be evaluated.

All of the aforementioned arguments are input by users on the application's client side, which corresponds to a streamlit server.

TO BE CONTINUED
