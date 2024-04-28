import PyFunctions.get_W_Lasso as getW
import torch

def RunBIOT(X, Fe, lam, maxIter = 200, eps = 1e-6, rotation = False, device='cpu'):
  '''
  X: embedding matrix (response)
  Fe: external feature matrix (predictors)
  lambda: sparsity hyperparameter
  max.iter: maximum number of iterations
  eps: convergence threshold
  rotation: should the orthogonal matrix be a rotation matrix? Yes = T, No = F
  '''

  # Store all variables in the appropriate device memory, ideally, GPU if available.
  W, r2 = getW.Lasso(Fe, X, lam, device=device)
  n = Fe.shape[0]

  diff = 1
  iter = 0
  crit = []
  crit.append(1)

  while( iter + 1 < maxIter and diff > eps ):  
    # Perform singular value decomposition
    u, d, v = torch.linalg.svd( (1 / (2 * n)) * X.t() @ (Fe @ W))

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
    W, r2 = getW.Lasso(Fe, X@R, lam, device=device)

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


# TEST
# X = torch.tensor(np.genfromtxt("X_train_norm_py.csv", delimiter=',', dtype='float64'))
# Fe = torch.tensor(np.genfromtxt("Fe_train_norm_py.csv", delimiter=',', dtype='float64'))
# R, W, crit, iter, r2 = RunBIOT(X, Fe,  0.0001, rotation=True)

# Pu = np.genfromtxt("Py_svd.csv", delimiter=',', dtype='float64')
# Ru = np.genfromtxt("R_svd.csv", delimiter=',', skip_header=1, dtype='float64')
# # diff = abs(Ru - Pu)
# # print(diff[diff > 1e-2])
# np.savetxt('Rsvd_Psvd.csv', Ru - Pu, delimiter=',')

# Pu = np.genfromtxt("Py_u.csv", delimiter=',', dtype='float64')
# Ru = np.genfromtxt("R_u.csv", delimiter=',', skip_header=1, dtype='float64')
# np.savetxt('Ru_Pu.csv', Ru - Pu, delimiter=',')

# Pu = np.genfromtxt("Py_d.csv", delimiter=',', dtype='float64')
# Ru = np.genfromtxt("R_d.csv", delimiter=',', skip_header=1, dtype='float64')
# print( Ru - Pu )

# Pu = np.genfromtxt("Py_v.csv", delimiter=',', dtype='float64')
# Ru = np.genfromtxt("R_v.csv", delimiter=',', skip_header=1, dtype='float64')
# print( Ru - Pu )

if __name__ == "__main__":
    import numpy as np
    import time
    import tracemalloc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    home = ""
    X = torch.tensor(np.genfromtxt(home + "X_norm_r.csv", delimiter=',', dtype='float64'))
    Fe = torch.tensor(np.genfromtxt(home + "Fe_norm_r.csv", delimiter=',', dtype='float64'))        
    lam = torch.tensor([0.0001], dtype=torch.float64, device=device)
    
    def testing():
        K = 30
        times = []
        for i in range(K):

            s = time.time()

            R, W, crit, iter, r2 = RunBIOT(X, Fe,  lam, rotation=True, device=device)

            times.append(time.time() - s)
        np.savetxt(f"run_BIOT_times_py.csv", times, delimiter=",")
        print(f"Iter: {iter}\n")

        np.savetxt("R_py.csv", R, delimiter=",")
        np.savetxt("W_py.csv", W, delimiter=",")
        np.savetxt("crit_py.csv", crit, delimiter=",")
        np.savetxt("r2_py.csv", r2, delimiter=",")


    def memtrace():
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        R, W, crit, iter, r2 = RunBIOT(X, Fe,  lam, rotation=True, device=device)

        snapshot_after = tracemalloc.take_snapshot()

        difference = snapshot_after.compare_to(snapshot_before, 'lineno')
        mems = []
        for stat in difference:
            mems.append(stat)
        print(difference)
        np.savetxt(f"run_BIOT_mems_py.csv", mems, delimiter=",")

    testing()
    memtrace()