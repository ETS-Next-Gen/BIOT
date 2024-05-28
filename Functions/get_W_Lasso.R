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
           standardize = F,
           maxit = 100000)$beta[, 2]})

  R_squared <- GetRSquared(X, W, Y)
  
  return(list(W=W, R_squared=R_squared))
}

# GetWLasso = function(X, Y, lambda){
#   # X: predictor matrix
#   # Y: response matrix
#   # lambda: Lasso hyperparameter
  
#   require(glmnet)
  
#   res <- sapply(1:(dim(Y)[2]), function(k) {
#     glmnet(x = as.matrix(X), 
#            y = Y[, k], 
#            intercept = F, 
#            lambda = c(100, lambda), 
#            family = "gaussian", 
#            standardize = F,
#            maxit = 100000)$npasses})
      
#   return(list(res=res))
# }

# TESTING
# X <- as.matrix(read.csv("X_norm_r.csv"))
# Fe <- as.matrix(read.csv("Fe_norm_r.csv"))

# res <- GetWLasso(X = Fe, Y = X, lambda = 0.0001)
# write.csv(res$W, file = "W_r.csv", row.names = FALSE)
# write.csv(res$R_squared, file = "R2_r.csv", row.names = FALSE)
# avg_value_R <- sum(res$R_squared) / (384)
# # 0.8037402
# print(avg_value_R)
# print("\n")

# K <- 30
# times <- list()
# for (i in 1:K) {
#   s <- Sys.time()

#   res <- GetWLasso(X = Fe, Y = X, lambda = 0.0001)

#   elapsed <- cat(Sys.time() - s,"\n")
#   times[[i]] <- elapsed

# }
# print("\n")

# p <- profmem({
#   res <- GetWLasso(X = Fe, Y = X, lambda = 0.0001)
# })
# print(cat(total(p), "\n"))
