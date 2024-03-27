import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression

import utils

import sys
import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import KFold

if __name__ == "__main__":
    arg_array = sys.argv
    client_nr = arg_array[1]
    # Load MNIST dataset from https://www.openml.org/d/554
    data_filename = '/Users/aglenis/MIP-Engine/tests/test_data/dementia_v_0_1/ppmi'+str(int(client_nr)+2)+'.csv'
    xvars = ['lefthippocampus', 'leftamygdala']
    yvars = ['gender']
    #(X_train, y_train), (X_test, y_test) = utils.load_mnist()
    #full_data = pd.read_csv(data_filename)
    if client_nr == 0:
        datasets = [i for i in range(5)]
    else:
        datasets = [i for i in range(5,10)]
    dataframes_list = []
    for i in datasets:
        curr_filename = '/Users/aglenis/MIP-Engine/tests/test_data/dementia_v_0_1/ppmi'+str(i)+'.csv'
        curr_df = pd.read_csv(curr_filename)
        dataframes_list.append(curr_df)

    full_data = pd.concat(dataframes_list)

    le = preprocessing.LabelEncoder()
    le.fit(['M','F'])

    X = full_data[xvars].values
    #X_test = X_train
    y = le.transform(full_data[yvars].values.ravel())
    #y_test = y_train

    n_models = 5

    kf = KFold(n_splits=n_models)

    X_train_list = []
    y_train_list = []

    X_test_list = []
    y_test_list = []
    for train, test in kf.split(X):

        X_train = X[train]
        y_train = y[train]
        X_train_list.append(X_train)
        y_train_list.append(y_train)

        X_test = X[test]
        y_test = y[test]
        X_test_list.append(X_test)
        y_test_list.append(y_test)

    print('X_train_list is '+str(X_train_list))
    print('y_train_list is '+str(y_train_list))

    # Split train set into 10 partitions and randomly use one for training.
    #partition_id = np.random.choice(10)
    #(X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # Create LogisticRegression Model
    modelsList = []
    for i in range(n_models):
        curr_model = LogisticRegression(
            penalty="l2",
            max_iter=1,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting
            solver = 'saga'
        )
        modelsList.append(curr_model)
    # Setting initial parameters, akin to model.compile for keras models
    #print('just before setting models params')
    modelsList= utils.set_initial_params(modelsList)
    print(modelsList)
    #print('setted initial params')
    #model2 = modelsList[0]
    #print(model2)
    #print(model2.coef_)
    #print(model2.fit_intercept)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(modelsList)

        def fit(self, parameters_list, config):  # type: ignore
            utils.set_model_params(modelsList, parameters_list)

            # Ignore convergence failure due to low local epochs
            accuracy_list_fit = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i in range(len(modelsList)):
                    modelsList[i].fit(X_train_list[i], y_train_list[i])
                    curr_accuracy_fit = modelsList[i].score(X_test_list[i], y_test_list[i])
                    accuracy_list_fit.append(curr_accuracy_fit)
                    print('accuracy during fit is '+str(curr_accuracy_fit))
            print(f"Training finished for round {config['server_round']}")
            model_ret = utils.get_model_parameters(modelsList)
            #print('model ret is '+str(model_ret))
            #print(model_ret.shape)
            tuple_ret = model_ret.tolist(),len(X_train),{"accuracy":np.mean(np.array(accuracy_list_fit))}
            #print(tuple_ret)
            return tuple_ret

        def evaluate(self, parameters_list, config):  # type: ignore
            utils.set_model_params(modelsList, parameters_list)
            #loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy_list = []
            for i,curr_model in enumerate(modelsList):
                curr_accuracy = curr_model.score(X_test_list[i], y_test_list[i])
                accuracy_list.append(curr_accuracy)
            return 0.0, len(X_test), {"accuracy": np.mean(np.array(accuracy_list))}

    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
