Eval <- function(R, W, Fe.test, X.test){
  # R: orthogonal transformation matrix
  # W: regression weights
  # Fe.test: external feature matrix (predictors), test set only
  # X.test: embedding matrix (response), test set only
  
  MSE <- GetMSEPred(Fe = Fe.test, 
                    X = X.test,
                    R = R, 
                    W = W)
  
  perc.nonzero <- GetL0(W = W)/prod(dim(W))
  
  return(cbind.data.frame(MSE = MSE, perc.nonzero = perc.nonzero))
}