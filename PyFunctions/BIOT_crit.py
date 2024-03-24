import torch

def BIOTCrit(Fe, X, R, W, lam):
  """  
  Fe: external feature matrix (predictors)
  X: embedding matrix (response)
  R: orthogonal transformation matrix
  W: regression weights
  lambda: sparsity hyperparameter
  """
  n = 1 / ( 2 * Fe.shape[0] )
  mx = X - Fe @ W @ R.T
  loss = lam * torch.sum( abs(W) )

  return ( n * torch.sum( torch.diag( mx.T @ mx)) + loss )
