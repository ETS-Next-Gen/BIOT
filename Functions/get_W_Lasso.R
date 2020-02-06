GetWLasso = function(X, Y, lambda){
  # X: predictor matrix
  # Y: response matrix
  # lambda: Lasso hyperparameter
  
  require(glmnet)
  
  W <- sapply(1:(dim(Y)[2]), function(k) {
    glmnet(x = as.matrix(X), 
           y = Y[, k], 
           intercept = F, 
           lambda = c(100, lambda), 
           family = "gaussian", 
           standardize = F)$beta[, 2]})

  R_squared <- GetRSquared(X, W, Y)
  
  return(list(W=W, R_squared=R_squared))
}