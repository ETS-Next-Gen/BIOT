GetMSEPred <- function(Fe, X, R, W) {
  # X: embedding matrix (response)
  # Fe: external feature matrix (predictors)
  # R: orthogonal transformation matrix
  # W: regression weights
  
  (1/(2*nrow(X)*ncol(X)))*sum((X%*%R - as.matrix(Fe)%*%W)^2)
  
}
 