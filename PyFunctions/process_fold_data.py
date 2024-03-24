import numpy as np
from sklearn.preprocessing import StandardScaler

def ProcessFoldData(X, Fe, testId, which_dummy = None) -> list:
  
  """
  X: embedding matrix (response)
  Fe: external feature matrix (predictors)
  test.id: vector of integers indicating the rows of X and Fe to assign to the test set
  dummy: vector whose elements = T if the corresponding column in Fe is a dummy variable, F otherwise
  
  """
  if which_dummy is None:
    which_dummy = np.repeat(False, Fe.shape[1])
  
  # Gathering test data using given train IDs
  train_id = np.setdiff1d(np.arange(X.shape[0]), testId)
  Fe_test = Fe[testId, :]
  X_test = X[testId, :]

  # Iolate non dummy features
  not_dummy = np.where(~np.isin(np.arange(1, Fe.shape[1] + 1), which_dummy))[0]
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
  # scaler = StandardScaler()
  # Fe_norm = scaler.fit_transform(Fe_train)
  # Fe_norm = np.column_stack((Fe[train_id[:, None], which_dummy], Fe_norm))
  # X_norm = StandardScaler(with_std=False).fit_transform(X[train_id, :])

  Fe_norm = (Fe_train - Fe_mean) / Fe_sd
  Fe_norm = np.column_stack((Fe[train_id[:, None], which_dummy], Fe_norm))
  X_norm = X[train_id, :] - X_mean

  # Scale and center test set using means and sds from the training set
  # Fe_test = scaler.fit_transform(Fe_test[:, not_dummy])
  # Fe_test = np.column_stack((Fe_test[:, which_dummy], Fe_test))
  # X_test = StandardScaler(with_std=False).fit_transform(X_test)

  Fe_test = Fe_test[:, not_dummy] - Fe_mean / Fe_sd
  Fe_test = np.column_stack((Fe_test[:, which_dummy], Fe_test))
  X_test = X[train_id, :] - X_mean

  return ( Fe_norm, X_norm, Fe_test, X_test ) 
