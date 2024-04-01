import numpy as np

def GetMSEPred(Fe, X, R, W):
  
  """
  X: embedding matrix (response)
  Fe: external feature matrix (predictors)
  R: orthogonal transformation matrix
  W: regression weights
  
  Returns the MSE
  """
  nrow, ncol = X.shape

  return (1 / (2 * nrow * ncol)) * np.sum((X @ R - Fe @ W)**2)
