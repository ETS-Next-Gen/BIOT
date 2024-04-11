import numpy as np
import torch

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

def ProcessFoldData(X, Fe, testId, which_dummy = None, device='cpu'):
  
  """
  X: embedding matrix (response)
  Fe: external feature matrix (predictors)
  test.id: vector of integers indicating the rows of X and Fe to assign to the test set
  dummy: vector whose elements = T if the corresponding column in Fe is a dummy variable, F otherwise
  
  """

  # Define dummy and non dummy columns
  if which_dummy is None:
    which_dummy = torch.zeros(Fe.shape[1], dtype=torch.bool, device=device)
  not_dummy = torch.tensor(list(set(range(0, Fe.shape[1])) - set(which_dummy.nonzero().flatten())), device=device)
  
  # Gathering train IDs
  uniques, counts = torch.cat((torch.arange(Fe.shape[0]), testId)).unique(return_counts=True)
  train_id = uniques[counts == 1]

  # Gathering test data
  Fe_test = Fe[testId, :]
  X_test = X[testId, :]

  # Gather train data, Iolate non dummy features
  Fe_train = Fe[train_id[:, None], not_dummy]

  # Mean and Std Dev of non-dummy features in the training data
  Fe_mean = torch.mean(Fe_train, dim=0)
  Fe_sd = torch.std(Fe_train, dim=0)
  
  # Replace std dev of trainnig data (where it equals 0)
  #  with 1 to avoid dividing by 0
  wh_zero_sd = torch.where(Fe_sd == 0)[0]
  if len(wh_zero_sd) > 0:
    Fe_sd[wh_zero_sd] = 1
  
  # Get mean of embedding dimension in the training set
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
import numpy as np
from guppy import hpy; h=hpy()

def testing():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  X = torch.tensor(np.genfromtxt("Datasets/embedding.csv", delimiter=',', dtype='float64'), device=device)
  Fe =  torch.tensor(np.genfromtxt("Datasets/dataset.csv", delimiter=',', skip_header=1, dtype='float64'), device=device)
  # foldIds = torch.split(torch.randperm(Fe.size(0)), Fe.size(0) // 10)
  foldIds = torch.tensor([28, 12, 20], device=device)

  heap_status1 = h.heap()

  Fe_norm, X_norm, Fe_test, X_test = ProcessFoldData(X = X, Fe = Fe, testId = foldIds)

  heap_status2 = h.heap()
  print(f"Mem: {heap_status2.size - heap_status1.size}")

  K = 30
  times = []
  for i in range(0, K):
    s = time.time()

    Fe_norm, X_norm, Fe_test, X_test = ProcessFoldData(X = X, Fe = Fe, testId = foldIds)

    elapsed = time.time() - s

    times.append(elapsed)
    print(elapsed)

testing()