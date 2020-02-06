ProcessFoldData <- function(X, Fe, test.id, which.dummy = rep(c(F), ncol(Fe))){
  # X: embedding matrix (response)
  # Fe: external feature matrix (predictors)
  # test.id: vector of integers indicating the rows of X and Fe to assign to the test set
  # which.dummy: vector whose elements = T if the corresponding column in Fe is a dummy variable, F otherwise
  
  train.id <- (1:nrow(X))[-test.id]
  Fe.test <- Fe[test.id, ]
  X.test <- X[test.id, ]
  
  # Get mean and sd of external features in training set
  
  wh.not.dummy <- which((c(1:ncol(Fe) %in% which.dummy) == F))
  Fe.mean <- apply(Fe[train.id, wh.not.dummy], 2, mean) 
  Fe.sd <- apply(Fe[train.id, wh.not.dummy], 2, sd)
  wh.zero.sd <- which(Fe.sd == 0)
  
  if (length(wh.zero.sd) > 0){
    Fe.sd[wh.zero.sd] <- 1 # replace with 1 to avoid dividing by 0
  }
  
  # Get mean of embedding dimension in training set
  X.mean <- apply(X[train.id, ], 2, mean)
  
  # Scale and center training set
  Fe.norm.dum <- Fe[train.id, which.dummy]
  Fe.norm <- scale(Fe[train.id, wh.not.dummy], center = Fe.mean, scale = Fe.sd)
  Fe.norm <- cbind(Fe.norm.dum, Fe.norm)
  X.norm <- scale(X[train.id, ], center = X.mean, scale = F)
  
  # Scale and center test set using means and sds from training set
  Fe.test.dum <- Fe.test[, which.dummy]
  Fe.test <- scale(Fe.test[, wh.not.dummy], center = Fe.mean, scale = Fe.sd)
  Fe.test <- cbind(Fe.test.dum, Fe.test)
  X.test <- scale(X.test, center = X.mean, scale = F)
  
  return(list(Fe.norm = Fe.norm, X.norm = X.norm, 
              Fe.test = Fe.test, X.test = X.test))
}