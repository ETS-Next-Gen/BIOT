RunBIOT <- function(X, Fe, lambda, max.iter = 2, eps = 1e-6, rotation = F){
  # X: embedding matrix (response)
  # Fe: external feature matrix (predictors)
  # lambda: sparsity hyperparameter
  # max.iter: maximum number of iterations
  # eps: convergence threshold
  # rotation: should the orthogonal matrix be a rotation matrix? Yes = T, No = F

  Fe <- as.matrix(Fe)
  # Lasso.sol <- GetWLasso(X = Fe, Y = X, lambda = lambda)
  # W <- Lasso.sol$W
  W <- matrix(1, nrow = ncol(Fe), ncol = ncol(X))
  diff <- Inf
  iter <- 1
  crit <- list(Inf)
 
  while(iter < max.iter && diff > eps){
    # print(paste("Iter: ", iter))
    
    decomp <- svd( (1/(2*nrow(Fe))) * t(X) %*% Fe %*% W)
    # write.csv(decomp$u, file = "R_u.csv", row.names = FALSE)
    # write.csv(decomp$d, file = "R_d.csv", row.names = FALSE)
    # write.csv(decomp$v, file = "R_v.csv", row.names = FALSE)
    # 35
    # print(which.min(decomp$d))
    
    if (rotation == T){ # If rotation matrix is desired
      
      sv <- decomp$d
      smallest <- which.min(sv)
      sv[smallest] <- sign(det(decomp$u%*%t(decomp$v)))
      sv[-smallest] <- 1
      R <- (decomp$u%*%diag(sv)%*%t(decomp$v))
      
    } else { # If orthogonal matrix is desired (more general)
      
      R <- (decomp$u%*%t(decomp$v))
      
    }
    
    # Lasso.sol <- GetWLasso(X = Fe, Y = X%*%R, lambda = lambda)
    # W <- Lasso.sol$W
    # R_squared <- Lasso.sol$R_squared
    W <- matrix(1, nrow = ncol(Fe), ncol = ncol(X))
    R_squared <- matrix(1, nrow = ncol(X), ncol = 1)

    
    crit[[iter + 1]] <- BIOTCrit(Fe, X, R, W, lambda)
    diff <- abs(crit[[iter]] - crit[[iter + 1]])
    
    iter <- iter + 1
    
  }
  
  return(list(R = R, W = W, iter = iter, crit = unlist(crit), R_squared = R_squared))
  
}

# BIOTCrit <- function(Fe, X, R, W, lambda){
#   # Fe: external feature matrix (predictors)
#   # X: embedding matrix (response)
#   # R: orthogonal transformation matrix
#   # W: regression weights
#   # lambda: sparsity hyperparameter
  
#   (1/(2*nrow(Fe)))*sum(diag(t(X - Fe%*%W%*%t(R))%*%((X - Fe%*%W%*%t(R))))) + lambda*sum(abs(W))
# }

# GetWLasso = function(X, Y, lambda){
#   # X: predictor matrix
#   # Y: response matrix
#   # lambda: Lasso hyperparameter
  
#   require(glmnet)
  
#   W <- sapply(1:(dim(Y)[2]), function(k) {
#     glmnet(x = as.matrix(X), 
#            y = Y[, k], 
#            intercept = F, 
#            lambda = c(100, lambda), 
#            family = "gaussian", 
#            standardize = F,
#            maxit = 100000)$beta[, 2]})

#   R_squared <- GetRSquared(X, W, Y)
  
#   return(list(W=W, R_squared=R_squared))
# }

# GetRSquared <- function(X, W, Y){
#   # X: predictor matrix
#   # W: regression weights
#   # Y: response matrix
  
#   Y <- as.matrix(Y)
#   Rsq <- sapply(1:ncol(Y), function(i){ 
#     var.pred <- sum((as.matrix(X)%*%W[, i] - Y[, i])^2)
#     var.y <- sum((Y[,i] - mean(Y[,i]))^2)
#     val <- 1 - (var.pred/var.y)
#     return(val)
#   })
#   return(Rsq)
# }

# # TESTING

# # Read data from CSV file
# X <- as.matrix(read.csv("Datasets/embedding.csv", header=F))
# Fe <- as.matrix(read.csv("Datasets/dataset.csv"))
# X <- scale(X, center = apply(X, 2, mean))
# Fe <- scale(Fe, center = apply(Fe, 2, mean) , scale = apply(Fe, 2, sd))

# # res <- RunBIOT(X, Fe, 0.0001, rotation = TRUE)
# # print(res$R_squared)

# K <- 30
# tot <- 0
# times <- list()
# for (i in 1:K) {
#   s <- Sys.time()

#   res <- RunBIOT(X, Fe, 0.0001, rotation = TRUE)

#   elapsed <- Sys.time() - s
#   print(elapsed)
#   times[[i]] <- elapsed
#   tot <- tot + elapsed
# }

# avg_value_R <- sum(res$R_squared) / 384
# print(avg_value_R)

# # for (i in 1:3) {
# #   p <- profmem({
# #     res <- RunBIOT(X, Fe, 0.0001, rotation = TRUE)
# #   })
# #   print(total(p))
# }