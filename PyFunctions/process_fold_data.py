import numpy as np
import torch

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

def ProcessFoldData(X, Fe, testId, which_dummy = None):
  
  """
  X: embedding matrix (response)
  Fe: external feature matrix (predictors)
  test.id: vector of integers indicating the rows of X and Fe to assign to the test set
  dummy: vector whose elements = T if the corresponding column in Fe is a dummy variable, F otherwise
  
  """

  # Define dummy and non dummy columns
  if which_dummy is None:
    # which_dummy = np.repeat(False, Fe.shape[1])
    which_dummy = torch.zeros(Fe.shape[1], dtype=torch.bool)
  not_dummy = torch.tensor(list(set(range(0, Fe.shape[1])) - set(which_dummy.nonzero().flatten())))
  
  # Gathering train IDs
  # train_id = np.setdiff1d(np.arange(X.shape[0]), testId)
  uniques, counts = torch.cat((torch.arange(Fe.shape[0]), testId)).unique(return_counts=True)
  train_id = uniques[counts == 1]

  # Gathering test data
  Fe_test = Fe[testId, :]
  X_test = X[testId, :]

  # Gather train data, Iolate non dummy features
  # not_dummy = np.where(~np.isin(np.arange(1, Fe.shape[1] + 1), which_dummy))[0]
  Fe_train = Fe[train_id[:, None], not_dummy]

  # Mean and Std Dev of non-dummy features in the training data
  # Fe_mean = np.sum(Fe_train, axis=0) / Fe_train.shape[0]
  # Fe_sd = np.std(Fe_train, axis=0)
  Fe_mean = torch.mean(Fe_train, dim=0)
  Fe_sd = torch.std(Fe_train, dim=0)
  
  # Replace std dev of trainnig data (where it equals 0)
  #  with 1 to avoid dividing by 0
  # wh_zero_sd = np.where(Fe_sd == 0)[0]
  wh_zero_sd = torch.where(Fe_sd == 0)[0]
  if len(wh_zero_sd) > 0:
    Fe_sd[wh_zero_sd] = 1
  
  # Get mean of embedding dimension in the training set
  # X_mean = np.mean(X[train_id, :], axis=0)
  X_mean = torch.mean(X[train_id], dim=0)

  # Scale and center features using its mean and sd
  Fe_norm = (Fe_train - Fe_mean) / Fe_sd
  Fe_test = Fe_test[:, not_dummy] - Fe_mean / Fe_sd

  # Center embeddings using mean
  X_norm = X[train_id, :] - X_mean
  X_test = X_test - X_mean

  # Concat with dummy columns... may need fixing
  if True in which_dummy:
    # Fe_norm = np.column_stack((Fe[train_id[:, None], which_dummy], Fe_norm))
    Fe_norm = torch.cat((Fe[train_id, which_dummy], Fe_norm))
    # Fe_test = np.column_stack((Fe_test[:, which_dummy], Fe_test))
    Fe_test = torch.cat((Fe_test[:, which_dummy], Fe_test))
  

  return ( Fe_norm, X_norm, Fe_test, X_test ) 


# TESTING
import time
X = torch.tensor(np.genfromtxt("Datasets/embedding.csv", delimiter=',', dtype='float64'))
Fe =  torch.tensor(np.genfromtxt("Datasets/dataset.csv", delimiter=',', skip_header=1, dtype='float64'))
foldIds = torch.split(torch.randperm(Fe.size(0)), Fe.size(0) // 10)

# .0587 sec

# Average: 0.009849357604980468
K = 10
tot = 0
for i in range(K):
  s = time.time()
  Fe_norm, X_norm, Fe_test, X_test = ProcessFoldData(X = X, Fe = Fe, testId = foldIds[0])
  elapsed = time.time() - s
  print(f"Time: {elapsed}")
  tot += elapsed

print(f"Average: {tot / K}")
