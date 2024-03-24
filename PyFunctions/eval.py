import numpy as np
from PyFunctions.get_MSE_pred import GetMSEPred
from PyFunctions.get_L0 import GetL0

def Eval(R, W, Fe_test, X_test):
  '''  
  R: orthogonal transformation matrix
  W: regression weights
  Fe.test: external feature matrix (predictors), test set only
  X.test: embedding matrix (response), test set only
  '''

  MSE = GetMSEPred(Fe = Fe_test, 
                    X = X_test,
                    R = R, 
                    W = W)
  
  percent_nonzero = GetL0(W = W) / np.prod(W.shape)

  return (MSE, percent_nonzero)
