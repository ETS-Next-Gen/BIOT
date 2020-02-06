RunBIOT <- function(X, Fe, lambda, max.iter = 200, eps = 1e-6, rotation = F){
  # X: embedding matrix (response)
  # Fe: external feature matrix (predictors)
  # lambda: sparsity hyperparameter
  # max.iter: maximum number of iterations
  # eps: convergence threshold
  # rotation: should the orthogonal matrix be a rotation matrix? Yes = T, No = F

  Fe <- as.matrix(Fe)
  Lasso.sol <- GetWLasso(X = Fe, Y = X, lambda = lambda)
  W <- Lasso.sol$W
  diff <- Inf
  iter <- 1
  crit <- list(Inf)
 
  while(iter < max.iter && diff > eps){
    
    decomp <- svd((1/(2*nrow(Fe)))*t(X)%*%Fe%*%W)
    
    if (rotation == T){ # If rotation matrix is desired
      
      sv <- decomp$d
      smallest <- which.min(sv)
      sv[smallest] <- sign(det(decomp$u%*%t(decomp$v)))
      sv[-smallest] <- 1
      R <- (decomp$u%*%diag(sv)%*%t(decomp$v))
      
    } else { # If orthogonal matrix is desired (more general)
      
      R <- (decomp$u%*%t(decomp$v))
      
    }
    
    Lasso.sol <- GetWLasso(X = Fe, Y = X%*%R, lambda = lambda)
    W <- Lasso.sol$W
    
    crit[[iter + 1]] <- BIOTCrit(Fe, X, R, W, lambda)
    diff <- abs(crit[[iter]] - crit[[iter + 1]])
    
    iter <- iter + 1
    
  }
  
  return(list(R = R, W = Lasso.sol$W, iter = iter, crit = unlist(crit), R_squared = Lasso.sol$R_squared))
  
}