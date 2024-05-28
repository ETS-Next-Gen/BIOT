###############
### BIOT ####
##############

rm(list = ls())

# Download the required libraries
if (system.file(package='glmnet') == "") {
  install.packages('glmnet')
}
if (system.file(package='exactRankTests') == "") {
  install.packages('exactRankTests')
}
library(glmnet) # Lasso
library(exactRankTests) # Wilcoxon test

# File paths 
data.path <- "Datasets/"
out.path <- "Results/result.RData"
function.path <- "Functions/"

args <- commandArgs(TRUE)

# Load functions
source.files <- list.files(path = function.path, recursive = TRUE)
invisible(sapply(source.files, function(x) source(file = paste0(function.path, x))))

nlambdas = 10
nfolds = 10 
maxLambda = 3.5 
sigThresh = .05

if (length(args) >= 4) {
  nlambdas = as.numeric(args[4])
}
# lambda.vals <- seq(0.0001, 1, length = nlambdas) # non-log scale
lambda.vals <- exp(seq(log(0.0001), log(maxLambda), length.out = nlambdas)) # log scale

if (length(args) == 0) {
  # Default files for Fe and X
  X <- read.csv(paste0(data.path, "embedding.csv"), header=F)
  Fe <- read.csv(paste0(data.path, "dataset.csv"))
} else if (length(args) == 1) {
  stop("The number of inputs you provided is not enough. You should provide (in order) the path to the embedding, the path to the dataset for explaining the embedding, and then the path to the output file.")
} else if (length(args) >= 2) {
  # The order should be: embedding first and then the dataset used to explain the embedding
  X <- read.csv(args[1], header=F)
  Fe <- read.csv(args[2])
  if (length(args) >= 3) {
    out.path <- args[3]
    if (length(args) >= 5) {
        lambda.vals <- exp(seq(log(as.numeric(args[5])), log(maxLambda), length = nlambdas))
        if (length(args) >= 6) {
          lambda.vals <- exp(seq(log(as.numeric(args[5])), log(as.numeric(args[6])), length = nlambdas))
          if (length(args) >= 7) {
            lambda.vals <- exp(seq(log(as.numeric(args[5])), log(as.numeric(args[6])), length = as.numeric(args[7])))
          } else {
            stop("You provided too many inputs.")
          }
        }
    }
  }
}

##############################################
#### Run BIOT for different lambda values ####
##############################################

print("Selection of lambda in progress...")

# Prepare fold ids
seed <- 155000
set.seed(seed)
# Eval results for each lambda in lambda.vals stored in df eval.res.lambda
 eval.res.lambda <- do.call(rbind, 
                           lapply(1:length(lambda.vals), function(lambda.vals.index) {
                             
  lambda <- lambda.vals[lambda.vals.index]
  print(paste('Processing lambda index ', lambda.vals.index))
  # Split data row indexes randomly into K folds
  K <- nfolds # number of folds for K-fold cross-validation
  fold.ids <- suppressWarnings(split(sample(nrow(Fe)), seq(1, nrow(Fe), length = K)))
  
  eval.res.K <- c()
  for (index.fold in 1:K){
    print(paste('    Processing fold index ', index.fold))
    # Process fold data
    fold.data <- ProcessFoldData(X = X, Fe = Fe, test.id = fold.ids[[index.fold]])
    
    # normalize lambda
    lambda.norm <- lambda/sqrt(ncol(fold.data$Fe.norm))
    # Get rotation matrix and weights
    res <- RunBIOT(X = fold.data$X.norm, 
                   Fe = fold.data$Fe.norm,
                   lambda = lambda.norm, rotation=T)
    R <- res$R
    W <- res$W
    print(res$R_squared)
    #print(res$W)

    # Eval
    tmp <- Eval(R = R, 
                W = W, 
                Fe.test = fold.data$Fe.test,
                X.test = fold.data$X.test)
    
    if (!is.na(tmp$MSE)) eval.res.K[[index.fold]] <- cbind.data.frame(lambda = lambda,lambda.norm = lambda.norm,tmp)
  }
  return(do.call(rbind.data.frame, eval.res.K))
}))


####################################
#### Now choose the best lambda ####
####################################

# Consider the lambda with the min avg_MSE
# lambda.avg.MSE: average test MSE for each lambda value in lambda.val
lambda.avg.MSE <- sapply(1:length(lambda.vals), function(lambda.val) 
   mean(eval.res.lambda[, "MSE"][which(eval.res.lambda[, "lambda"] == lambda.vals[lambda.val])], na.rm = T))
which.lambda.min.MSE <- which.min(lambda.avg.MSE)

print(rbind(lambda.vals, lambda.avg.MSE))

print(paste0("The lambda with the min avg MSE is ", lambda.vals[which.lambda.min.MSE], " at index ", which.lambda.min.MSE))

results <- eval.res.lambda
lambda.index <- which.lambda.min.MSE
not.finished <- T
if (which.lambda.min.MSE == nlambdas) {
    test.index <- lambda.index
} else {
    test.index <- lambda.index + 1
}
while(not.finished & test.index <= nlambdas){
  MSE1 <- results[which(results[, "lambda"] == lambda.vals[lambda.index]), 
                  "MSE"]
  MSE2 <- results[which(results[, "lambda"] == lambda.vals[test.index]),
                  "MSE"]
  if (sum(abs(MSE1) - abs(MSE2)) == 0){
    pval <- 1
  } else {
    pval <- exactRankTests::wilcox.exact(MSE1,
                                         MSE2,
                                         paired = T)$p.val
  }
  
  if (pval <= sigThresh){
    best.lambda.index <- test.index - 1
    not.finished <- F
  }
  if (pval > sigThresh & test.index == length(lambda.vals)){
    best.lambda.index <- test.index
    not.finished <- F
  }
  test.index <- test.index + 1
}

print(paste0("The most sparse lambda that is not significantly different from the best lambda is ", lambda.vals[best.lambda.index], " at index ", best.lambda.index))

################################################################
#### Now run BIOT with the best lambda on the whole dataset ####
################################################################
# Some elements of Fe can have a standard deviation (sd) equal to 0, which is an issue when scaling.
# In order to avoid this problem, the sd for these columns is set to 1.
Fe.sd <- apply(Fe, 2, sd)
wh.zero.sd <- which(Fe.sd == 0)
if (length(wh.zero.sd) > 0){
  Fe.sd[wh.zero.sd] <- 1 # replace with 1 to avoid dividing by 0
}

res <- RunBIOT(X = scale(X, center=T, scale=F),
               Fe = scale(Fe, center=T, scale=Fe.sd),
               lambda = lambda.vals[best.lambda.index]/sqrt(ncol(Fe)), rotation=T)

# Put regression weights and Rsq in a list and save to .RData file
W_R_squared <- list(W = res$W, R_squared = res$R_squared)
save(W_R_squared, file = paste(out.path, 'RData.csv'))
write.csv(res$R_squared,file=paste(out.path, 'RSquared.csv'))
print(paste0("Final weights and Rsq stored in ", paste(out.path, 'RData.csv'), ". R object is called W_R_squared."))
show(W_R_squared)

# Output the rotated matrix
scaledX = scale(X, center=T, scale=F)
write.csv(scaledX, file=paste(out.path, 'scaledX.csv'), row.names=FALSE)
RMatrix = data.matrix(scaledX) %*% data.matrix(res$R)
write.csv(RMatrix, file = paste(out.path, 'rMatrix.csv'), row.names=FALSE)
write.csv(res$R, file=paste(out.path, 'rotation.csv'), row.names=FALSE)
FeX = scale(Fe, center=T, scale=Fe.sd)
write.csv(FeX, file=paste(out.path, 'features.csv'), row.names=FALSE)
cors = cor(RMatrix, FeX)
write.csv(cors, file = paste(out.path, 'cors.csv'), row.names=TRUE, col.names=TRUE)
combined = cbind(RMatrix, FeX)
write.csv(combined, file=paste(out.path, 'combined.csv'), row.names=TRUE)
WMatrix = data.matrix(FeX) %*% data.matrix(res$W)
write.csv(WMatrix, file=paste(out.path, 'pMatrix.csv'), row.names=FALSE)
write.csv(res$W, file=paste(out.path, 'weights.csv'))
