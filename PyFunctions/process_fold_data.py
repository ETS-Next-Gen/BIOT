import typing as t

import numpy as np


def ProcessFoldData(
    X: np.ndarray, Fe: np.ndarray, testId: np.ndarray, which_dummy: np.ndarray = None
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    X: embedding matrix (response)
    Fe: external feature matrix (predictors)
    test.id: vector of integers indicating the rows of X and Fe to assign to the test set
    dummy: vector whose elements = T if the corresponding column in Fe is a dummy variable, F otherwise

    """

    if which_dummy is None:
        which_dummy = np.repeat(False, Fe.shape[1])
        # which_dummy = torch.zeros(Fe.shape[1], dtype=torch.bool)

    # Gathering test data using given train IDs
    train_id = np.setdiff1d(np.arange(X.shape[0]), testId)
    # train_id = torch.tensor(list(set(range(X.shape[0])) - set(testId)), device=device)
    Fe_test = Fe[testId, :]
    X_test = X[testId, :]

    # Iolate non dummy features
    not_dummy = np.where(~np.isin(np.arange(1, Fe.shape[1] + 1), which_dummy))[0]
    # not_dummy = torch.tensor(list(set(range(0, Fe.shape[1])) - set(which_dummy.nonzero().flatten())), device=device)
    Fe_train = Fe[train_id[:, None], not_dummy]

    # Mean and Std Dev of non-dummy features in the training data
    Fe_mean = np.sum(Fe_train, axis=0) / Fe_train.shape[0]
    Fe_sd = np.std(Fe_train, axis=0)

    # Replace std dev of trainnig data (where it equals 0)
    #  with 1 to avoid dividing by 0
    wh_zero_sd = np.where(Fe_sd == 0)[0]
    if len(wh_zero_sd) > 0:
        Fe_sd[wh_zero_sd] = 1

    # Get mean of embedding dimension in the training set
    X_mean = np.mean(X[train_id, :], axis=0)

    # Scale and center training dataset using its mean and sd
    Fe_norm = (Fe_train - Fe_mean) / Fe_sd
    Fe_norm = np.column_stack((Fe[train_id[:, None], which_dummy], Fe_norm))
    # Fe_norm = torch.cat((Fe[train_id[:, None], which_dummy], Fe_norm))
    X_norm = X[train_id, :] - X_mean

    # Scale and center test set using means and sds from the training set
    Fe_test = Fe_test[:, not_dummy] - Fe_mean / Fe_sd
    Fe_test = np.column_stack((Fe_test[:, which_dummy], Fe_test))
    # Fe_test = torch.cat((Fe_test[:, which_dummy], Fe_test))
    X_test = X_test - X_mean

    return (Fe_norm, X_norm, Fe_test, X_test)
