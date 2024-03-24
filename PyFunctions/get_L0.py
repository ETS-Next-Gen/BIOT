import numpy as np

def GetL0(W):
  """
  W: regression weights
  """
  return np.count_nonzero(W)
