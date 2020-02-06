GetRSquared <- function(X, W, Y){
  # X: predictor matrix
  # W: regression weights
  # Y: response matrix
  
  Y <- as.matrix(Y)
  Rsq <- sapply(1:ncol(Y), function(i){ 
    var.pred <- sum((as.matrix(X)%*%W[, i] - Y[, i])^2)
    var.y <- sum((Y[,i] - mean(Y[,i]))^2)
    val <- 1 - (var.pred/var.y)
    return(val)
  })
  return(Rsq)
}