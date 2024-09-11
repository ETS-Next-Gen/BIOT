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
def lasso_coordinate_descent(X: torch.Tensor, Y: torch.Tensor, lam: torch.Tensor, W: torch.Tensor, max_iter: int = 1000, tol: float = 1e-10) -> torch.Tensor:
    _, a = X.shape
    _, b = Y.shape
    # W shape = (a, b)
    
    summation_ = torch.sum(X ** 2, dim=0) # shape = (a,)
    
    for _ in range(max_iter):
        W_old = W.clone()
        
        for k in range(a):
            
            if summation_[k] == 0:
                continue  
            
            residual = Y - torch.matmul(X, W) + torch.matmul(X[:, k].unsqueeze(-1), W[k, :].unsqueeze(0)) # shape = (n, b)
                
            rho = torch.mv(residual.T, X[:, k]).unsqueeze(-1) # shape = (b,1)
            W[k,:] = (torch.sign(rho) * torch.maximum( torch.abs(rho) - lam, torch.zeros_like(rho)) ).squeeze() / summation_[k]
            
        # stopping criteria
        if ( torch.max(torch.abs(W - W_old)) < tol * torch.max(torch.abs(W)) ):
            # 
            break
        
    return W


import time 

s = time.time()
for  val, alpha in enumerate(lambdaVals):
    if val != 5: 
        continue    
    
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
    np.savetxt(f"{output}/W_{val}.csv", W, delimiter=",")
    
    
print(f"Total time taken: {(time.time() - s) / 60} minutes")


Rotation = Rotation.cpu().numpy()
# Save regression weights to a CSV file
np.savetxt(f"{output}/Weights.csv", W, delimiter=",")

# ScaledX
scaledX = ( Embeddings - Embeddings.mean() ).cpu().numpy()
np.savetxt(f"{output}/ScaledX.csv", scaledX, delimiter=",")

# Output the rotated mx
RMatrix = np.dot( scaledX , Rotation )
np.savetxt(f"{output}/rMatrix.csv", RMatrix, delimiter=",")
np.savetxt(f"{output}/Rotation.csv", Rotation, delimiter=",")

# ScaledFe
scaledFe = scale(Features.cpu().numpy())
np.savetxt(f"{output}/Features.csv", scaledFe, delimiter=",")

# correlation matrix
# Initialize a result matrix
cor_result = np.zeros((RMatrix.shape[1], scaledFe.shape[1]))

# Compute correlation
for i in range(RMatrix.shape[1]):
    for j in range(scaledFe.shape[1]):
        cor_result[i, j] = np.corrcoef(RMatrix[:, i], scaledFe[:, j])[0, 1]
np.savetxt(f"{output}/Cors.csv", cor_result, delimiter=",")




# Device: cuda
# Selection of lambda in progress...
# Lambda values: tensor([2.2942e-05, 7.3370e-05, 2.3465e-04, 7.5043e-04, 2.4000e-03, 7.6755e-03,
#         2.4547e-02, 7.8505e-02, 2.5107e-01, 8.0295e-01], device='cuda:0')
# Training data size torch.Size([20471, 19])
# Alpha: 2.294157093274407e-05
# Iteration 99 : Total error 0.004017333383671939 MSE 0.002543011214584112 Reg 0.0014743221690878272
# Finished 0.4696368873119354 in 6.773028576374054 minutes
# Alpha: 7.337018905673176e-05
# Iteration 2 : Total error 0.0072622946463525295 MSE 0.002548670396208763 Reg 0.004713624250143766
# Finished 1.5019611120224 in 0.24535430669784547 minutes
# Alpha: 0.00023464761034119874
# Iteration 99 : Total error 0.007380361668765545 MSE 0.002549191005527973 Reg 0.004831170663237572
# Finished 4.803471088409424 in 2.005840214093526 minutes
# Alpha: 0.000750433886423707
# Iteration 99 : Total error 0.011232739081606269 MSE 0.002549911616370082 Reg 0.008682827465236187
# Finished 15.36213207244873 in 1.9193068504333497 minutes
# Alpha: 0.0023999870754778385
# Iteration 99 : Total error 0.020122064743191004 MSE 0.002556732390075922 Reg 0.017565332353115082
# Finished 49.13013458251953 in 1.3079829335212707 minutes
# Alpha: 0.0076754773035645485
# Iteration 99 : Total error 0.039712556172162294 MSE 0.0025744778104126453 Reg 0.03713807836174965
# Finished 157.12469482421875 in 0.7229491949081421 minutes
# Alpha: 0.024547196924686432
# Iteration 99 : Total error 0.07625015918165445 MSE 0.0026328274980187416 Reg 0.07361733168363571
# Finished 502.50567626953125 in 0.31625725428263346 minutes
# Alpha: 0.07850518077611923
# Iteration 27 : Total error 0.13342827884480357 MSE 0.002808499615639448 Reg 0.13061977922916412
# Finished 1607.07958984375 in 0.1135671059290568 minutes
# Alpha: 0.25106996297836304
# Iteration 1 : Total error 0.00382909644395113 MSE 0.00382909644395113 Reg 0.0
# Finished 5139.6533203125 in 0.16058769226074218 minutes
# Alpha: 0.8029549717903137
# Iteration 1 : Total error 0.00382909644395113 MSE 0.00382909644395113 Reg 0.0
# Finished 16437.291015625 in 0.1613401134808858 minutes
# Total time taken: 13.727245326836904 minutes
