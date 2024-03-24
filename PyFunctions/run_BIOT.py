from get_W_Lasso import GetWLasso
from BIOT_crit import BIOTCrit
import torch
import numpy as np

def RunBIOT(X, Fe, lam, maxIter = 2, eps = 1e-6, rotation = False):
  """
  X: embedding matrix (response)
  Fe: external feature matrix (predictors)
  lambda: sparsity hyperparameter
  max.iter: maximum number of iterations
  eps: convergence threshold
  rotation: should the orthogonal matrix be a rotation matrix? Yes = T, No = F
  """

  # Store all variables in the appropriate device memory, ideally, GPU if available.
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  X = torch.tensor(X).to(device)
  Fe = torch.tensor(Fe).to(device)
  W, r2 = torch.tensor(GetWLasso(X=Fe, Y=X, lam=lam)).to(device)

  diff = 1
  iter = 0
  crit = []
  crit.append(1)

  while( iter + 1 < maxIter and diff > eps ):  
    # Perform singular value decomposition
    u, d, v = torch.linalg.svd( (1 / (2 * Fe.shape[0])) * X.t() @ (Fe @ W))

    if rotation:
      # Set the smallest singular value to the sign of the determinant of U and V^T.
      # Set everything else to equal 1.
      smallest = torch.argmin(d)
      d[smallest] = torch.sign(torch.det(u @ v.T))
      d[d != d[smallest]] = 1

      R = u @ torch.diag(d) @ v.T

    else:
      # If rotation is not preferred, do not multiply R times the diagonal of d.
      R = u @ v.t()
    
    # Find the new weights and r^2, 
    W, r2 = GetWLasso(X=Fe, Y=X@R, lam=lam)

    # Find the difference between the critical values of the current and previous iteration.
    # We will stop iterating once this difference is less than epsilon.
    crit.insert(iter + 1, BIOTCrit(Fe, X, R, W, lam))
    diff = abs(crit[iter] - crit[iter + 1])

    # On to the next iteration. 
    iter += 1
  

  return ( R, W, iter, crit, r2 ) 
