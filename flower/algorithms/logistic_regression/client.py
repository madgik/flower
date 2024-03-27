import glob
import os.path
import sys
import warnings
from pathlib import Path

import flwr as fl
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FOLDER = Path(PROJECT_ROOT / "tests" / "test_data" / "dementia_v_0_1")

if __name__ == "__main__":
    xvars = ['lefthippocampus', 'leftamygdala']
    yvars = ['gender']

    dataframes_list = []
    for file in glob.glob(os.path.join(DATA_FOLDER / "ppmi*.csv")):
        curr_df = pd.read_csv(file)
        dataframes_list.append(curr_df)

    full_data = pd.concat(dataframes_list)

    le = preprocessing.LabelEncoder()
    le.fit(['M', 'F'])

    X_train = full_data[xvars].values
    X_test = X_train
    y_train = le.transform(full_data[yvars].values)
    y_test = y_train

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)


    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}


    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
