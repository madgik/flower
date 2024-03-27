from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import openml

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

LogRegParamsCV = List[LogRegParams]
ModelsList = List[LogisticRegression]

def get_model_parameters(modelsList: ModelsList) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    ret_list = []
    for model in modelsList:
        if model.fit_intercept:
            coeff = model.coef_
            intercept = model.intercept_
            curr_array = np.hstack((coeff,intercept.reshape(intercept.shape[0],1)))
            #print('params with intercept '+str(curr_array))
        else:
            #print('params without intercept '+str(params))
            curr_array = model.coef_
            #print(params)
        #print(params)
        ret_list.append(curr_array)
        #print(ret_list)
    #print(ret_list)
    #print('ret_list is '+str(ret_list))
    ret_array = np.array(ret_list)
    print('ret array from get parameters is '+str(ret_array))

    return ret_array


def set_model_params(
    modelsList: ModelsList, params: LogRegParams
) -> ModelsList:
    """Sets the parameters of a sklean LogisticRegression model."""
    print('running set params')
    #raise ValueError(params)
    print('whole params is '+str(params))
    #print('params shape is '+str(params.shape))
    for i,model in enumerate(modelsList):
        curr_params = params[i]
        print('curr params is '+str(curr_params))
        curr_params = params[i]
        modelsList[i].coef_ = curr_params[:,0:-1]
        if model.fit_intercept:
            modelsList[i].intercept_ = curr_params[:,-1]
    return modelsList


def set_initial_params(modelsList: ModelsList):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = 2  # Number of features in dataset
    for i,model in enumerate(modelsList):
        modelsList[i].classes_ = np.array([i for i in range(n_classes)])

        modelsList[i].coef_ = np.zeros((n_classes, n_features))
        if modelsList[i].fit_intercept:
            modelsList[i].intercept_ = np.zeros((n_classes,))
    return modelsList



def load_mnist() -> Dataset:
    """Loads the MNIST dataset using OpenML.

    OpenML dataset link: https://www.openml.org/d/554
    """
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    X = Xy[:, :-1]  # the last column contains labels
    y = Xy[:, -1]
    # First 60000 samples consist of the train set
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    return (x_train, y_train), (x_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
