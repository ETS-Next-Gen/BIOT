import torch
import os
import numpy as np
from utils import ProcessFoldData, MSE, scale, SVD
from scipy.stats import wilcoxon
from gpu import TorchL1
import warnings 
warnings.filterwarnings("ignore")




############################################
#### DEFAULT FILE PATHS ####
############################################
datasets = "../datasets/"
output = "../output/"
# datasets = "../datasets/layers10_big"
try: os.mkdir(output)
except: pass

print("\nDefault file paths:-------------")
print(f"datasets: {datasets}")
print(f"output: {output}")
print("--------------------------------\n")


############################################
#### BIOT HYPERPARAMETERS ####
############################################
nLambdas = 10
minLambda = 0.0001
maxLambda = 3.5
K = 10 # no of folds used for cross validation
sigThresh = .05   # sigma threshold
maxiter = 100 # Maximum number of iterations for the training step
num = 1000 # Number of training samples to use during the cross validation and testing
CV = True # To Randomly sample {num} training data during cross-validation step only
# CV =  False 


print("\nDefault parameters:-------------")
print(f"nLambdas: {nLambdas}")
print(f"minLambda: {minLambda}")
print(f"maxLambda: {maxLambda}")
print(f"Number of folds: {K}")
print(f"sigThresh: {sigThresh}")
print(f"maxiter: {maxiter}")
print(f"num of training samples for CV: {num}")
print(f"Randomly sample training data during cross-validation: {CV}")
print("--------------------------------\n")

############################################
#### PYTORCH ENVIRONMENT VARIABLES SETUP ####
############################################
device = torch.device("cuda")
# device = torch.device("cpu")
print(f"\nDevice: {device}\n")



############################################
#### LOAD DATA FROM FILE OR RANDOM DATA ####
############################################
file = True 
if file:
  Embeddings = torch.tensor(np.genfromtxt(f"{datasets}/embeddings.csv", delimiter=',', dtype='float64'), device=device)
  Features =  torch.tensor(np.genfromtxt(f"{datasets}/features.csv", delimiter=',', skip_header=1, dtype='float64'), device=device)
else:
  # DATASET PARAMETERS FOR RANDOM DATA
  Bsz = 100
  Edim = 384
  Fdim = 21
  # Loading from a file

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


run_CV = True
# run_CV = False
print(f" Do not use cross validation if the dataset is large enough, Simply train the model for different values of lambda and pick the one which gives you similar MSE error and higher regualrizaiton. Set run_CV as False.")

if run_CV:

  results = []
  # Perform cross validation for each lambda
  for li, lam in enumerate(lambdaVals):
    # Normalize lambda
    lam_norm = lam.item()
    print('Processing lambda: ', lam_norm, ' at index: ', li)

    # Cross validation!
    fold_results = []
    for foldIdx in range(0, K):
      
      clf = TorchL1(alpha=lam_norm, max_iter=maxiter, tol=1e-6)
          
      # preprocess embeddings and features
      Features_norm, Embeddings_norm, Features_test, Embeddings_test = ProcessFoldData(X = Embeddings, Fe = Features, testId = foldIds[foldIdx], CV=CV, num=num)
      
      clf.coef_ = torch.zeros(Edim, Fdim).to(torch.float32).to(device).T
      clf.alpha *= Features_norm.size(0)
      
      dummymse_error = 0
      for iter in range(maxiter):
    
        # Rotation
        Rotation = SVD(Features_norm, clf.coef_, Embeddings_norm)
        
        # Lasso regression
        Y = torch.mm(Embeddings_norm,Rotation)
        clf.fit(Features_norm,Y)
        
        mse, reg = MSE(Embeddings_norm, Features_norm, Rotation, clf, lam_norm)
        mse_error = mse + reg
        if abs(mse_error - dummymse_error) < 1e-2: 
          break
        else: 
          dummymse_error = mse_error
        
    
      # Testing
      mse, reg = MSE(Embeddings_test, Features_test, Rotation, clf, lam_norm)
      mse_errortest = mse
      print(f"Processing : {foldIdx}, Iteration: {iter}, MSE: {mse}, Reg: {reg}, Total: {mse_errortest}, mse_error_Train: {mse_error}")

      # Make sure MSE is valid
      if MSE is not None:
        fold_results.append([lam_norm, mse_error])  
    # Add output to final results
    results.append(fold_results)

  print("\nFinished running BIOT on fold data with different lambda values!")


  ####################################
  #### Now choose the best lambda ####
  ####################################

  print("\nNow calculating the best lambda...")

  # Calculate all of the average MSEs for each lambda across all folds of data
  lam_avg_mse = torch.zeros(len(results), device=device)
  for i, fold_results in enumerate(results):
    mse = torch.tensor([fold[-1] for fold in fold_results], device=device)
    fold_avg = mse.mean() # removes the scalar value from the calculated vector ?
    lam_avg_mse[i] = fold_avg

  for i, lam in enumerate(lambdaVals):
    print(f"Lambda: {lam.item()}, Avg MSE: {lam_avg_mse[i].item()}")
  print("Use this info and what is recommended by the algorithm to choose the best lambda")


  # Find the lambda with the smallest average MSE
  lam_best = lam_min_mse_idx = torch.argmin(lam_avg_mse).item()
  lam_min_mse = lambdaVals[lam_min_mse_idx]
  print(f"\nLAMBDA WITH SMALLEST AVG MSE: {lam_min_mse} and mse: {lam_avg_mse[lam_min_mse_idx]}")


  mse_min = torch.tensor([fold[1] for fold in results[lam_min_mse_idx]], device=device)
  test_idx = lam_min_mse_idx + 1
  while test_idx < len(results):
    mse_new = torch.tensor([fold[1] for fold in results[test_idx]], device=device)

    if torch.sum(torch.abs(mse_min) - torch.abs(mse_new)) == 0:
      pval = 1
    else:
      pval = wilcoxon(mse_min.cpu().numpy(), mse_new.cpu().numpy(), alternative='two-sided')[1]

    if pval <= sigThresh:
      lam_best = test_idx - 1
      break
    if pval > sigThresh and test_idx + 1 == nLambdas:
      lam_best = test_idx
      break

    test_idx += 1

  lam_best_norm = lambdaVals[lam_best].item()
  print(f"The most sparse lambda that is not significantly different from the best lambda is {lam_best_norm} at index {lam_best}")
  with open(f"{output}/{num}_best_lambda.txt", "w") as f:
    f.write(f"value:-{lam_best_norm}_index:-{lam_best}")


################################################################
#### Now run BIOT with the best lambda on the whole dataset ####
################################################################

# lam_best = 5
lam_best_norm = lambdaVals[lam_best].item()
print(f"The most sparse lambda that is not significantly different from the best lambda is {lam_best_norm} at index 5")

clf = TorchL1(alpha=lam_best_norm, max_iter=maxiter, tol=1e-6)

# preprocess embeddings and features
Features_norm, Embeddings_norm = ProcessFoldData(X = Embeddings, Fe = Features, testId = foldIds[0], only_standardize=True)
print(f"Features_norm: {Features_norm.shape}, Embeddings_norm: {Embeddings_norm.shape}")

dummymse_error = 0
maxiter = 500

clf.coef_ = torch.zeros(Edim, Fdim).to(torch.float32).to(device).T
clf.alpha *= Features_norm.size(0)

for iter in range(maxiter):
  
  # Rotation
  Rotation = SVD(Features_norm, clf.coef_, Embeddings_norm)
  
  # Lasso regression
  Y = torch.mm(Embeddings_norm,Rotation)
  clf.fit(Features_norm,Y)
  
  mse, reg = MSE(Embeddings_norm, Features_norm, Rotation, clf, lam_best_norm)
  mse_error = mse + reg
  print(f"Iter : {iter}, MSE: {mse}, Reg: {reg}, Total: {mse_error}")
        
  if abs(mse_error - dummymse_error) < 1e-6 : 
    break
  else: 
    dummymse_error = mse_error
  
print(f"\n-----Main training step, Iteration: {iter}, MSE: {mse}, Reg: {reg}, Total: {mse_error}-----\n")


############################################
#### Now Save the results to a CSV file ####
############################################
W = clf.coef_.cpu().numpy()
Rotation = Rotation.cpu().numpy()
print(W, W.max(), W.min())
print(Rotation, Rotation.max(), Rotation.min())



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
