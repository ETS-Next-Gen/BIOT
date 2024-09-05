import torch
from sklearn import linear_model
import os
import numpy as np
from utils import ProcessFoldData
from scipy.stats import wilcoxon
from utils import SVD
from utils import ProcessFoldData, MSE, scale, SVD, L1
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings 
warnings.filterwarnings("ignore")

# DEFAULT FILE PATHS
datasets = "../datasets/"
output = "../output/"
datasets = "../datasets/layers10_big"
try: os.mkdir(output)
except: pass

print("Default file paths:-------------")
print(f"datasets: {datasets}")
print(f"output: {output}")
print("--------------------------------")



# DEFAULT PARAMETERS
nLambdas = 10
minLambda = 0.0001
maxLambda = 3.5
K = 10            # no of folds used for cross validation
sigThresh = .05   # sigma threshold
maxiter = 200

print("Default parameters:-------------")
print(f"nLambdas: {nLambdas}")
print(f"minLambda: {minLambda}")
print(f"maxLambda: {maxLambda}")
print(f"Number of folds: {K}")
print(f"sigThresh: {sigThresh}")
print(f"maxiter: {maxiter}")
print("--------------------------------")



# PYTORCH ENVIRONMENT VARIABLES
# device = torch.device("cuda")
device = torch.device("cuda")

print(f"Device: {device}")



# DATASET PARAMETERS FOR RANDOM DATA
Bsz = 20000
Edim = 384
Fdim = 21

# Loading from a file
file = True
# file = False
if file:
  Embeddings = torch.tensor(np.genfromtxt(f"{datasets}/embeddings.csv", delimiter=',', dtype='float64'), device=device)
  Features =  torch.tensor(np.genfromtxt(f"{datasets}/features.csv", delimiter=',', skip_header=1, dtype='float64'), device=device)
else:
  Embeddings = torch.rand(Bsz,Edim).to(torch.float64).to(device)
  Features = torch.rand(Bsz,Fdim).to(torch.float64).to(device)


Edim = Embeddings.shape[1]
Fdim = Features.shape[1] 


##############################################
#### Run BIOT for different lambda values ####
##############################################

print("Selection of lambda in progress...")

# Define lambda vector, feature vector, and embedding vector
lambdaVals = torch.exp(torch.linspace(np.log(minLambda), np.log(maxLambda), nLambdas)).to(device) / np.sqrt(Features.shape[1])
print(f"Lambda values: {lambdaVals}")

# Split data into K folds such that each foldid has the indexes to use
foldIds = torch.split(torch.randperm(Features.size(0)), Features.size(0) // K)


# preprocess embeddings and features
Features_norm, Embeddings_norm = ProcessFoldData(X = Embeddings, Fe = Features, testId = foldIds[0], only_standardize=True)
# Features_norm, Embeddings_norm, Features_test, Embeddings_test = ProcessFoldData(X = Embeddings, Fe = Features, testId = foldIds[1], CV=True)

print(F"Training data size {Features_norm.size()}")

@torch.jit.script
def lasso_coordinate_descent(X: torch.Tensor, Y: torch.Tensor, lam: torch.Tensor, W: torch.Tensor, max_iter: int = 1000, tol: float = 1e-5) -> torch.Tensor:
    _, a = X.shape
    _, b = Y.shape
    # W shape = (a, b)
    
    summation_ = torch.sum(X ** 2, dim=0) # shape = (a,)
    residual = Y - torch.matmul(X, W) # shape = (n, b)
    
    for _ in range(max_iter):
        W_old = W.clone()
        
        for k in range(a):
            
            if summation_[k] == 0:
                continue  
            residual = Y - torch.matmul(X, W) # shape = (n, b)
            residual += torch.matmul(X[:, k].unsqueeze(-1), W[k, :].unsqueeze(0)) 
                
            rho = torch.mv(residual.T, X[:, k]).unsqueeze(-1) # shape = (b,1)
            W[k,:] = (torch.sign(rho) * torch.max( torch.abs(rho) - lam, torch.zeros_like(rho)) ).squeeze() / summation_[k]
            
            # residual -= torch.matmul(X[:, k].unsqueeze(-1), W[k, :].unsqueeze(0))
            
        # stopping criteria
        if ( torch.max(torch.abs(W - W_old)) < tol * torch.max(torch.abs(W)) ):
            break
        
    return W



import time 

s = time.time()
for iter, alpha in enumerate(lambdaVals):
    print(f"Alpha: {alpha}")
    alpha *= Embeddings_norm.size(0)


    W = torch.zeros(Edim, Fdim).to(torch.float32).to(Embeddings_norm.device).T
    dummymse_error = 0
    
    start = time.time()
    for iter in range(100):
        # Rotation
        Rotation = SVD(Features_norm, W, Embeddings_norm)
        
        # Lasso regression
        Y = torch.matmul(Embeddings_norm,Rotation)
        W = lasso_coordinate_descent(Features_norm, Y, alpha, W)

        mse, reg = MSE(Embeddings_norm, Features_norm, Rotation, W,  alpha / Embeddings_norm.size(0))
        mse_error = mse + reg
        
        if abs(mse_error - dummymse_error) < 1e-6: 
            break
        else: 
            dummymse_error = mse_error
            
    print(f"Iteration {iter} : Total error {mse_error} MSE {mse} Reg {reg}")
    print(f"Finished {alpha} in {(time.time() - start) / 60} minutes")

    W = W.cpu().numpy()
    np.savetxt(f"{output}/W_{alpha}.csv", W, delimiter=",")
    
    
print(f"Total time taken: {(time.time() - s) / 60} minutes")