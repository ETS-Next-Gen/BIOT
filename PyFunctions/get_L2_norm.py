import numpy as np

def GetL2Norm(vec):
  """
  vec: vector
  """
  return np.sqrt(np.dot(vec, vec))
