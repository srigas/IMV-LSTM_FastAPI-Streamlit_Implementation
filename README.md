# IMV-LSTM FastAPI-Streamlit Implementation

## Table of Contents

+ [About](#about)
+ [FastAPI](#fastapi)
    + [Preprocessing](#preprocessing)
    + [Training](#training)
    + [Evaluation](#evaluation)
    + [Inference](#inference)
+ [Streamlit](#streamlit)
+ [Docker](#docker)
    + [Installation](#installation)

## About <a name = "about"></a>

This repository corresponds to a "full-stack" implementation of the IMV-LSTM neural network presented in [this](https://arxiv.org/pdf/1905.12034.pdf) paper. This falls under the broader category of Explainable AI, where not only is the LSTM used to make predictions on time-series data, but we can also gain insights on why it makes said predictions, i.e. which features are more influential for them, as well as how this influence (importance) varies over time. The model is trained using a subset of the [PM2.5 Beijing Data](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) dataset and is subsequently evaluated using either the remaining data-points, or custom-made evaluation datasets, as long as they are in the form of the PM2.5 dataset. The model can also be used for inference.

The structure of the project is split into two main components: the [backend](/backend), containing the FastAPI code and all ML related scripts, and the [frontend](/frontend), containing the streamlit code and configurations for the user interface of the project.

## FastAPI <a name = "fastapi"></a>

As far as the backend is concerned, for the whole process of training and evaluating the model, as well as performing inference, four endpoints are generated using [FastAPI](https://fastapi.tiangolo.com/).

### Preprocessing <a name = "preprocessing"></a>

The endpoint is called `preprocessing` and aims at cleaning the data properly, preprocessing (for example, filling missing values) and scaling them before splitting them to be used for training and validation. The subset of the PM2.5 dataset used for the training of the model can be found in [backend/data](/backend/data/train_data.csv). The features that are eventually used for training are: `pm2.5`, `DEWP`, `TEMP`, `PRES`, `cbwd`, `Iws`, `Is`, `Ir`. The arguments for the call are the following:

- `window_size`: integer with default value 10, corresponds to the number previous data-points that compose the feature matrix which is used for the prediction.
- `predict_window`: integer with default value 1, corresponds to the number of data-points to be predicted in the future. Obviously, higher values of this parameter lead to less accurate predictions (quantity over quality).
- `val_percent`: float between 0 and 0.5, corresponds to the percentage of the training dataset that is allocated for the validation of the model during its training. Setting this value to 0 automatically deactivates the Early Stopping mechanism that is utilized during training.

### Training <a name = "training"></a>

The second endpoint is called `training` and takes care of the model's training. The relevant function's arguments are:

- `batch_size`: integer with default value 32, corresponds to the size of the data batches during model training.
- `n_units`: integer with default value 128, corresponds to the number of LSTM units of the model.
- `epochs`: integer with default value 100, corresponds to the maximum number of training epochs (or exact number of training epochs if Early Stopping is deactivated).
- `patience`: integer with default value 12, corresponds to the number of patience epochs of the Early Stopping mechanism. If set to -1, it is deactivated.

Note that the training happens using the CPU of your machine, so if you intend to train your model for many epochs, you may have to wait for a while. For this reason, the `backend/results` directory contains some already preprocessed data along with a trained model, in case evaluation and inference are the only things that the user wants to try out.

### Evaluation <a name = "evaluation"></a>

After having trained a model with a satisfactory final MSE (preferably on validation data), the model can be evaluated on unseen data-points, using the `evaluation` endpoint. Note that the data need to be properly labeled, as this process is not equivalent to inference. The `backend/data` directory of the present repository contains a [test_data.csv](/backend/data/test_data.csv) file which may be used for this purpose, but feel free to create your own custom dataset or use other publicly available ones, as long as they conform to the standards of the PM2.5 dataset when it comes to feature names, types, etc. The arguments for the relevant call are:

- `eval_batch_size`: integer with default value 32, corresponds to the size of the data batches during model evaluation.
- `file`: a .csv file containing the evaluation data, to be uploaded so that the model can be evaluated.

### Inference <a name = "inference"></a>

Finally, the last endpoint called `inference` is perhaps the most important one when it comes to realistic applications, since the real solution to problems comes only after the model is deployed and exposed to new data. The data-points used for inference must correspond to feature matrices with rows equal to the `window_size` parameter used during preprocessing and training, conforming to the following format:

| | pm2.5  | DEWP | TEMP | PRES | cbwd | Iws | Is | Ir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0  | ...  | ... | ... | ... | ... | ... | ... | ... |

The directory `backend/data` contains a [sample_request.csv](/backend/data/sample_request.csv) file, which corresponds to a sample request for inference, for `window_size=10`. Feel free to construct your own.

## Streamlit <a name = "streamlit"></a>

As far as the application's frontend is concerned, it is based on [Streamlit](https://streamlit.io/https://streamlit.io/) and creates a user interface where all the endpoint call parameters can be configured through sliders. The following image shows how the UI looks like once everything is up and running. To customize the theme (for example, change by default to a darker theme) you can tinker with the [config.toml](/frontend/.streamlit/config.toml) file.

## Docker <a name = "docker"></a>

In order to ensure compatibility irregardless of OS, python versions, libraries, etc. the web application has been containerized into [Docker](https://www.docker.com/) containers. Both the frontend and backend have their own Dockerfile and their relative dependency is encoded into the docker-compose.yml file in the root directory. 

### Installation <a name = "installation"></a>

To run the web app on your machine, you need to have Docker installed. First, clone the repository to a local folder using

```
git clone https://github.com/srigas/IMV-LSTM_FastAPI-Streamlit_Implementation
```

Next, nagivate into the main directory of the repository and run `docker-compose build` followed by `docker-compose run`. This is all that is required for the web app to start running. If everything has gone smoothly, the terminal must inform you about the healthy condition of both containers, as well as where to navigate in your browser in order to interact with the Streamlit UI (see image below for reference).

![Terminal](https://github.com/srigas/temp_rep/blob/main/imvlstmstreamlit/terminal.png)
