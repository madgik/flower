import flwr as fl
import utils
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import numpy as np
from sklearn import preprocessing

from typing import Tuple, Union, List



XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

LogRegParamsCV = List[LogRegParams]
ModelsList = List[LogisticRegression]

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(modelsList: ModelsList):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    #_, (X_test, y_test) = utils.load_mnist()

    xvars = ['lefthippocampus', 'leftamygdala']
    yvars = ['gender']

    dataframes_list=[]

    for i in range(10):
        curr_filename = '/Users/aglenis/MIP-Engine/tests/test_data/dementia_v_0_1/ppmi'+str(i)+'.csv'
        curr_df = pd.read_csv(curr_filename)
        dataframes_list.append(curr_df)

    full_data = pd.concat(dataframes_list)

    print(np.unique(full_data[yvars]))

    le = preprocessing.LabelEncoder()
    le.fit(['M','F'])
    print(list(le.classes_))

    X_test = full_data[xvars].values
    y_test = le.transform(full_data[yvars].values.ravel())


    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters_list: List[fl.common.NDArrays], config):
        # Update model with the latest parameters
        modelsList_new = utils.set_model_params(modelsList, parameters_list)
        #modelsList = modelsList_new
        #loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy_list = []
        for i,curr_model in enumerate(modelsList):
            #print('curr_model coeff is '+str(curr_model.coef_))

            if server_round ==5:
                y_pred = curr_model.predict(X_test)
                df = pd.DataFrame()
                df['y_pred'] = y_pred
                df.to_csv('y_pred_'+str(i)+'_federated.csv',index=False)

            accuracy = curr_model.score(X_test, y_test)
            accuracy_list.append(accuracy)
            print('accuracy is '+str(accuracy))
        return 0, {"accuracy": np.mean(np.array(accuracy_list))}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    n_models =5
    modelsList = []
    for i in range(n_models):
        curr_model = LogisticRegression(penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
        solver = 'saga')
        modelsList.append(curr_model)
    utils.set_initial_params(modelsList)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(modelsList),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
