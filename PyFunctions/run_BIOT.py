from PyFunctions.get_W_Lasso import GetWLasso
from PyFunctions.BIOT_crit import BIOTCrit
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
  # W = torch.ones((Fe.shape[1], X.shape[1]), dtype=torch.float64).to(device)

  diff = 1
  iter = 0
  crit = []
  crit.append(1)

  while( iter + 1 < maxIter and diff > eps ):  
    # Perform singular value decomposition
    u, d, v = torch.linalg.svd( (1 / (2 * Fe.shape[0])) * X.t() @ (Fe @ W))
    np.savetxt('Py_u.csv', u.numpy(), delimiter=',')
    np.savetxt('Py_d.csv', d.numpy(), delimiter=',')
    np.savetxt('Py_v.csv', v.numpy(), delimiter=',')
    # 383
    print(torch.argmin(d))

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
    
    # Find the new weights and r^2, 
    W, r2 = GetWLasso(X=Fe, Y=X@R, lam=lam)

    # Find the difference between the critical values of the current and previous iteration.
    # We will stop iterating once this difference is less than epsilon.
    crit.insert(iter + 1, BIOTCrit(Fe, X, R, W, lam))
    diff = abs(crit[iter] - crit[iter + 1])

    # On to the next iteration. 
    iter += 1
  

  return ( R, W, iter, crit, r2 ) 


# # TEST
X = torch.tensor(np.genfromtxt("X_train_norm_py.csv", delimiter=',', dtype='float64'))
Fe = torch.tensor(np.genfromtxt("Fe_train_norm_py.csv", delimiter=',', dtype='float64'))
R, W, crit, iter, r2 = RunBIOT(X, Fe,  0.0001, rotation=True)

Pu = np.genfromtxt("Py_svd.csv", delimiter=',', dtype='float64')
Ru = np.genfromtxt("R_svd.csv", delimiter=',', skip_header=1, dtype='float64')
# diff = abs(Ru - Pu)
# print(diff[diff > 1e-2])
np.savetxt('Rsvd_Psvd.csv', Ru - Pu, delimiter=',')

Pu = np.genfromtxt("Py_u.csv", delimiter=',', dtype='float64')
Ru = np.genfromtxt("R_u.csv", delimiter=',', skip_header=1, dtype='float64')
np.savetxt('Ru_Pu.csv', Ru - Pu, delimiter=',')

# Pu = np.genfromtxt("Py_d.csv", delimiter=',', dtype='float64')
# Ru = np.genfromtxt("R_d.csv", delimiter=',', skip_header=1, dtype='float64')
# print( Ru - Pu )

# Pu = np.genfromtxt("Py_v.csv", delimiter=',', dtype='float64')
# Ru = np.genfromtxt("R_v.csv", delimiter=',', skip_header=1, dtype='float64')
# print( Ru - Pu )

# print(f"X:{W.shape}\nR:{R.shape}\ncrit:{crit}\niter:{iter}")