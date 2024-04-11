import torch

def GetRSquared(X, Y, W):
  '''
  X: predictor matrix
  W: regression weights
  Y: response matrix

  Returns the R^2. A value close to 0 indicates little correlation, while a value close to 1 indicates the opposite.
  '''
  
  return 1 - (torch.sum((X @ W - Y)**2, dim=0) / torch.sum((Y - torch.mean(Y))**2, dim=0))


def Lasso(X, Y, lambdas, maxIter=100000, device='cpu', alpha=0.1, intercept=False, thresh=1e-7):
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
  nLambdas = lambdas.size(0) - 1

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
    idx = nLambdas if i > nLambdas else i

    # Update gradients for weight
    dw = -(2 / m) * X.T @ (Y - Y_pred)
    dw = torch.where( w > 0, dw - ((2 / m) * lambdas[idx]), dw + ((2 / m) * lambdas[idx]))

    # Update the weights
    w = w - alpha * dw

    # Compute loss.... should it be torch.sum(.. , dim=0) ?
    loss = (1 / (2 * m)) * torch.sum((Y - Y_pred) ** 2) + lambdas[idx] * torch.sum(torch.abs(w))

    # Check convergence criterion
    # print(f"Iter : {i}, diff : {prev_loss - loss}, lambda: {lambdas[idx]}")
    if abs(prev_loss - loss) < thresh:  # You can adjust this threshold as needed
      return w, GetRSquared(X, Y, w)

    # Update loss
    prev_loss = loss
  
  print("WARNING: Convergence criteria not met. Returning regression weights anyway.")
  return w, GetRSquared(X, Y, w)



# TESTS + PROFILING
def runtime():
  import numpy as np
  import time
  from sklearn.preprocessing import StandardScaler
  import matplotlib.pyplot as plt
  from guppy import hpy; h=hpy()

  DatasetPath = "Datasets/dataset.csv"
  EmbeddingPath = "Datasets/embedding.csv"

  # heap_status1 = h.heap()

  X1 = StandardScaler(with_std=False).fit_transform(np.genfromtxt(EmbeddingPath, delimiter=',', dtype='float64'))
  Fe1 = StandardScaler().fit_transform(np.genfromtxt(DatasetPath, delimiter=',', skip_header=1, dtype='float64'))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")
  X = torch.tensor(X1, device=device)
  Fe = torch.tensor(Fe1, device=device)
  # lambdas = torch.tensor([10000, 1000, 500, 300, 200, 100, 50, 10, 1, 0.0001], dtype=torch.float64)
  lambdas = torch.tensor([100, 0.0001], dtype=torch.float64, device=device)

  # W = Lasso(Fe, X, lambdas)

  # heap_status2 = h.heap()
  # print(heap_status2.size - heap_status1.size)

  K = 5
  tot = 0
  times = []
  for i in range(K):

    s = time.time()

    W, r2 = Lasso(Fe, X, lambdas, device=device)

    elapsed = time.time() - s
    times.append(elapsed)
    print(elapsed)
    tot += elapsed


  r2 = GetRSquared(Fe, X, W)
  avg = torch.sum(r2) / (384)
  print(f"Average r2: {avg}")
  
runtime()

