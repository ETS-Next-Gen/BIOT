RunBIOT <- function(X, Fe, lambda, max.iter = 2, eps = 1e-6, rotation = F){
  # X: embedding matrix (response)
  # Fe: external feature matrix (predictors)
  # lambda: sparsity hyperparameter
  # max.iter: maximum number of iterations
  # eps: convergence threshold
  # rotation: should the orthogonal matrix be a rotation matrix? Yes = T, No = F

  Fe <- as.matrix(Fe)
  W <- matrix(1, nrow = ncol(Fe), ncol = ncol(X))
  R_squared <- matrix(1, nrow = ncol(X), ncol = 1)
  diff <- Inf
  iter <- 1
  crit <- list(Inf)
 
  while(iter < max.iter && diff > eps){
    # print(paste("Iter: ", iter))
    
    decomp <- svd( (1/(2*nrow(Fe))) * t(X) %*% Fe %*% W)
    # write.csv(decomp$u, file = "R_u.csv", row.names = FALSE)
    # write.csv(decomp$d, file = "R_d.csv", row.names = FALSE)
    # write.csv(decomp$v, file = "R_v.csv", row.names = FALSE)
    
    if (rotation == T){ # If rotation matrix is desired
      
      sv <- decomp$d
      smallest <- which.min(sv)
      sv[smallest] <- sign(det(decomp$u%*%t(decomp$v)))
      sv[-smallest] <- 1
      R <- (decomp$u%*%diag(sv)%*%t(decomp$v))
      
    } else { # If orthogonal matrix is desired (more general)
      
      R <- (decomp$u%*%t(decomp$v))
      
    }
    
    # W <- matrix(1, nrow = ncol(Fe), ncol = ncol(X))
    # R_squared <- matrix(1, nrow = ncol(X), ncol = 1)

    
    crit[[iter + 1]] <- BIOTCrit(Fe, X, R, W, lambda)
    diff <- abs(crit[[iter]] - crit[[iter + 1]])
    
    iter <- iter + 1
    
  }
  
  return(list(R = R, W = W, iter = iter, crit = unlist(crit), R_squared = R_squared))
  
}

BIOTCrit <- function(Fe, X, R, W, lambda){
  # Fe: external feature matrix (predictors)
  # X: embedding matrix (response)
  # R: orthogonal transformation matrix
  # W: regression weights
  # lambda: sparsity hyperparameter
  
  (1/(2*nrow(Fe)))*sum(diag(t(X - Fe%*%W%*%t(R))%*%((X - Fe%*%W%*%t(R))))) + lambda*sum(abs(W))
}

# # TESTING

# Read data from CSV file
X <- as.matrix(read.csv("X_norm_r.csv"))
Fe <- as.matrix(read.csv("Fe_norm_r.csv"))


K <- 30
for (i in 1:K) {
  s <- Sys.time()

  res <- RunBIOT(X, Fe, 0.0001, rotation = TRUE)

  elapsed <- Sys.time() - s
  print(elapsed)
}

# Print the iteration number
cat(sprintf("Iter: %d\n", res$iter))

# Save the matrices to CSV files
write.csv(res$R, "runBIOT/Rmx_R.csv", row.names = FALSE)
write.csv(res$W, "runBIOT/W_R.csv", row.names = FALSE)
write.csv(res$crit, "runBIOT/crit_R.csv", row.names = FALSE)
write.csv(res$r2, "runBIOT/r2_R.csv", row.names = FALSE)
