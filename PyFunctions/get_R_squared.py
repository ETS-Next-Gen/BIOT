import numpy as np

def GetRSquared(X, W, Y):
  '''
  X: predictor matrix
  W: regression weights
  Y: response matrix
  '''
  
  Y = np.asmatrix(Y)
  return 1 - (np.sum((Y - X.dot(W))**2) / np.sum((Y - np.mean(Y))**2))
