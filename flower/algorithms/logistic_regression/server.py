from typing import Dict

import flwr as fl
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    # _, (X_test, y_test) = utils.load_mnist()

    xvars = ['lefthippocampus', 'leftamygdala']
    yvars = ['gender']

    dataframes_list = []

    for i in range(10):
        curr_filename = '../../../tests/test_data/dementia_v_0_1/ppmi' + str(i) + '.csv'
        curr_df = pd.read_csv(curr_filename)
        dataframes_list.append(curr_df)

    full_data = pd.concat(dataframes_list)

    print(np.unique(full_data[yvars]))

    le = preprocessing.LabelEncoder()
    le.fit(['M', 'F'])
    print(list(le.classes_))

    X_test = full_data[xvars].values
    y_test = le.transform(full_data[yvars].values.ravel())

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)

        accuracy_output = {"accuracy": accuracy}

        if server_round == 5:
            import json
            with open('result.json', 'w') as f:
                json.dump(accuracy_output, f)

        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
