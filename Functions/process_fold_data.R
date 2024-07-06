ProcessFoldData <- function(X, Fe, test.id, which.dummy = rep(c(F), ncol(Fe))){
  # X: embedding matrix (response)
  # Fe: external feature matrix (predictors)
  # test.id: vector of integers indicating the rows of X and Fe to assign to the test set
  # which.dummy: vector whose elements = T if the corresponding column in Fe is a dummy variable, F otherwise
  
  # Gathering test data using given train IDs
  train.id <- (1:nrow(X))[-test.id]
  Fe.test <- Fe[test.id, ]
  X.test <- X[test.id, ]
  
  # Get mean and sd of external features in training set
  wh.not.dummy <- which((c(1:ncol(Fe) %in% which.dummy) == F))
  
  # Mean and Std Dev of non-dummy features in the training data
  Fe.mean <- apply(Fe[train.id, wh.not.dummy], 2, mean) 
  Fe.sd <- apply(Fe[train.id, wh.not.dummy], 2, sd)

  # Replace std dev of trainnig data (where it equals 0)
  #  with 1 to avoid dividing by 0
  wh.zero.sd <- which(Fe.sd == 0)
  if (length(wh.zero.sd) > 0){
    Fe.sd[wh.zero.sd] <- 1
  }
  
  # Get mean of embedding dimension in training set
  X.mean <- apply(X[train.id, ], 2, mean)
  
  # Scale and center training set using its sd and mean
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


# TESTING
# install.packages("https://cran.r-project.org/bin/macosx/big-sur-x86_64/contrib/4.3/profmem_0.6.0.tgz", repos=NULL)
# library("profmem")
# options(profmem.threshold = 2000)

X <- read.csv(paste0("Datasets/embedding.csv"))
Fe <- read.csv(paste0("Datasets/dataset.csv"))
fold.ids <- c(29, 13, 21)

K <- 30
times <- list()
for (i in 1:K) {
  s <- Sys.time()

  fold.data <- ProcessFoldData(X = X, Fe = Fe, test.id = fold.ids[[1]])

  elapsed <- Sys.time() - s

  times[[i]] <- cat(elapsed,"\n")
}

write.csv(fold.data$Fe.norm, file = "Fe_norm_r.csv", row.names = FALSE)
write.csv(fold.data$X.norm, file = "X_norm_r.csv", row.names = FALSE)
write.csv(fold.data$Fe.test, file = "Fe_test_r.csv", row.names = FALSE)
write.csv(fold.data$X.test, file = "X_test_r.csv", row.names = FALSE)
