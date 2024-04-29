import torch

def GetRSquared(X, Y, W):
  '''
  X: predictor matrix
  W: regression weights
  Y: response matrix

  Returns the R^2. A value close to 0 indicates little correlation, while a value close to 1 indicates the opposite.
  '''
  
  return 1 - (torch.sum((X @ W - Y)**2, dim=0) / torch.sum((Y - torch.mean(Y))**2, dim=0))


def Lasso(X, Y, lam, maxIter=100000, device='cpu', alpha=0.1, intercept=False, thresh=1e-7):
  '''
  X: predictor matrix
  Y: response matrix
  lambdas: sequence of lambdas for regularization
  maxIter: maximum number of iterations allowed
  device: gpu or cpu?
  alpha: learning rate
  intercept: should the bias be calculated too?
  thresh: convergence criteria

  KEY POINT: Use lambdas of decreasing value. A higher first lambda returns higher r2, 
    although Lasso runs more iterations and takes 10x more time.
  Returns w, the regression weights, and the corresponding r^2 value.
  '''

  m, n = X.shape
  l = Y.shape[1]
  w = torch.zeros((n, l), dtype=torch.float64).to(device)
  b = 0
  prev_loss = 0
  # nLambdas = lambdas.size(0) - 1

  # Implementing Gradient Descent algorithm for Optimization
  for i in range(maxIter):
    # print(f"Iter {i}")

    # Make prediction
    Y_pred = X @ w 

    # Gradient for bias + updating bias
    if intercept:
      Y_pred += b
      db = -(2 / m) * torch.sum(Y - Y_pred)
      b = b - alpha * db

    # Set lambda index
    # idx = nLambdas if i > nLambdas else i

    # Update gradients for weight
    dw = -(2 / m) * X.T @ (Y - Y_pred)
    dw = torch.where( w > 0, dw - ((2 / m) * lam), dw + ((2 / m) * lam))

    # Update the weights
    w = w - alpha * dw

    # Compute loss.... should it be torch.sum(.. , dim=0) ?
    loss = (1 / (2 * m)) * torch.sum((Y - Y_pred) ** 2) + lam * torch.sum(torch.abs(w))

    # Check convergence criterion
    # print(f"Iter : {i}, diff : {prev_loss - loss}, lambda: {lambdas[idx]}")
    if abs(prev_loss - loss) < thresh:  # You can adjust this threshold as needed
      return w, GetRSquared(X, Y, w)

    # Update loss
    prev_loss = loss
  
  print("WARNING: Convergence criteria not met. Returning regression weights anyway.")
  return w, GetRSquared(X, Y, w)



# TESTS + PROFILING
if __name__ == "__main__":
  def runtime():
    import numpy as np
    import time
    from guppy import hpy; h=hpy()

    home = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X1 = np.genfromtxt(home + "X_norm_r.csv", delimiter=',', dtype='float64', skip_header=1)
    Fe1 = np.genfromtxt(home + "Fe_norm_r.csv", delimiter=',', dtype='float64', skip_header=1)
    X = torch.tensor(X1, device=device)
    Fe = torch.tensor(Fe1, device=device)
    # lambdas = torch.tensor([10000, 1000, 500, 300, 200, 100, 50, 10, 1, 0.0001], dtype=torch.float64)
    lam = torch.tensor([0.0001], dtype=torch.float64, device=device)

    # heap_status1 = h.heap()
    W, r2 = Lasso(Fe, X, lam, device=device)
    # np.savetxt('r2_py.csv', r2, delimiter=',')

    # np.savetxt('W_py.csv', W, delimiter=',')
    W_r = np.loadtxt('W_r.csv', delimiter=',', skiprows=1)
    R2_r = np.loadtxt('R2_r.csv', delimiter=',')

    W_diff = abs(np.mean(W_r, 0) - np.mean(W.numpy(), 0))
    R2_diff = abs(R2_r - r2.numpy())

    np.savetxt('W_mean_py.csv', np.mean(W.numpy(), 0), delimiter=',')  
    np.savetxt('W_mean_R.csv', np.mean(W_r, 0), delimiter=',')
    np.savetxt('W_mean_diff.csv', W_diff, delimiter=',')

    print(np.mean(W_diff))
    print(np.min(W_diff))
    print(np.max(W_diff))

    # heap_status2 = h.heap()
    # print(f"Mem: {heap_status2.size - heap_status1.size}")
    print(f"Avg r2: {torch.sum(r2) / (384)}")

    K = 30
    for i in range(K):

      s = time.time()

      W, r2 = Lasso(Fe, X, lam, device=device)

      elapsed = time.time() - s
      print(elapsed)
    
  runtime()