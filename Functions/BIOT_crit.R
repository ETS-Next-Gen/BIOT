BIOTCrit <- function(Fe, X, R, W, lambda){
  # Fe: external feature matrix (predictors)
  # X: embedding matrix (response)
  # R: orthogonal transformation matrix
  # W: regression weights
  # lambda: sparsity hyperparameter
  
  (1/(2*nrow(Fe)))*sum(diag(t(X - Fe%*%W%*%t(R))%*%((X - Fe%*%W%*%t(R))))) + lambda*sum(abs(W))
}