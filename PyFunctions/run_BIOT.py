from get_W_Lasso import Lasso
import torch

def RunBIOT(X, Fe, lam, maxIter = 2, eps = 1e-6, rotation = False, device='cpu'):
  '''
  X: embedding matrix (response)
  Fe: external feature matrix (predictors)
  lambda: sparsity hyperparameter
  max.iter: maximum number of iterations
  eps: convergence threshold
  rotation: should the orthogonal matrix be a rotation matrix? Yes = T, No = F
  '''

  # Store all variables in the appropriate device memory, ideally, GPU if available.
  W, r2 = Lasso(Fe, X, lam, device=device)

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
      d[smallest] = torch.sign(torch.linalg.det(u @ v.T))
      d[d != d[smallest]] = 1

      R = u @ torch.diag(d) @ v.T

    else:
      # If rotation is not preferred, do not multiply R times the diagonal of d.
      R = u @ v.t()
    
    # Find the new weights and r^2 
    W, r2 = Lasso(Fe, X@R, lam, device=device)

    # Find the difference between the critical values of the current and previous iteration.
    # We will stop iterating once this difference is less than epsilon.
    crit.insert(iter + 1, BIOTCrit(Fe, X, R, W, lam))
    diff = abs(crit[iter] - crit[iter + 1])

    # On to the next iteration. 
    iter += 1
  

  return ( R, W, iter, crit, r2 ) 


def BIOTCrit(Fe, X, R, W, lam):
  '''  
  Fe: external feature matrix (predictors)
  X: embedding matrix (response)
  R: orthogonal transformation matrix
  W: regression weights
  lambda: sparsity hyperparameter
  '''

  n = 1 / ( 2 * Fe.shape[0] )
  mx = X - Fe @ W @ R.T
  loss = lam * torch.sum( abs(W) )

  return ( n * torch.sum( torch.diag( mx.T @ mx)) + loss )
