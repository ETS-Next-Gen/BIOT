import torch
from sklearn import linear_model
import os
import numpy as np
from utils import ProcessFoldData, MSE, BIOT, SVD
from scipy.stats import wilcoxon

import warnings 
warnings.filterwarnings("ignore")

# DEFAULT FILE PATHS
datasets = "../datasets/"
output = "../output/"
try: os.mkdir(output)
except: pass

print("Default file paths:-------------")
print(f"datasets: {datasets}")
print(f"output: {output}")

# DEFAULT PARAMETERS
nLambdas = 10
minLambda = .0001
maxLambda = 3.5
K = 10            # no of folds used for cross validation
sigThresh = .05   # sigma threshold
maxiter = 1000

print("Default parameters:-------------")
print(f"nLambdas: {nLambdas}")
print(f"minLambda: {minLambda}")
print(f"maxLambda: {maxLambda}")
print(f"Number of folds: {K}")
print(f"sigThresh: {sigThresh}")
print(f"maxiter: {maxiter}")


# PYTORCH ENVIRONMENT VARIABLES
device = torch.device("cuda")
# device = torch.device("cpu")

print(f"Device: {device}")



# DATASET PARAMETERS FOR RANDOM DATA
Bsz = 100
Edim = 384
Fdim = 21

# Loading from a file
file = True
if file:
  Embeddings = torch.tensor(np.genfromtxt(f"{datasets}/embeddings.csv", delimiter=',', dtype='float64'), device=device)
  Features =  torch.tensor(np.genfromtxt(f"{datasets}/features.csv", delimiter=',', skip_header=1, dtype='float64'), device=device)

else:
  Embeddings = torch.rand(Bsz,Edim).to(torch.float64).to(device)
  Features = torch.rand(Bsz,Fdim).to(torch.float64).to(device)

assert Embeddings.shape[1] == Edim
assert Features.shape[1] == Fdim





##############################################
#### Run BIOT for different lambda values ####
##############################################

print("Selection of lambda in progress...")

# Define lambda vector, feature vector, and embedding vector
lambdaVals = torch.exp(torch.linspace(np.log(minLambda), np.log(maxLambda), nLambdas)).to(device)
print(f"Lambda values: {lambdaVals}")

# Split data into K folds such that each foldid has the indexes to use
foldIds = torch.split(torch.randperm(Features.size(0)), Features.size(0) // K)


results = []
# Perform cross validation for each lambda
for lam in lambdaVals:
  print('Processing lambda: ', lam)
  # Normalize lambda
  lam_norm = lam.item() / np.sqrt(Features.shape[1])

  # Cross validation!
  fold_results = []
  for foldIdx in range(0, K):
    print("Processing fold: ", foldIdx)

    clf = linear_model.Lasso(alpha=lam_norm)
    # intializing the rotation matrix
    Rotation = torch.randn(Edim, Edim, dtype=torch.float64, device=device)


    # preprocess embeddings and features
    Features_norm, Embeddings_norm, Features_test, Embeddings_test = ProcessFoldData(X = Embeddings, Fe = Features, testId = foldIds[foldIdx])

    dummymse_error = 0
    for iter in range(1):
      W, Rotation = BIOT(Embeddings, Features, Rotation, clf)
      mse_error = MSE(Embeddings, Features, Rotation, clf)
      if mse_error - dummymse_error < 0.000001:
        break
      dummymse_error = mse_error


    # Make sure MSE is valid
    if MSE is not None:
      fold_results.append([lam, lam_norm, mse_error])
      

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
  mse = torch.tensor([fold[2] for fold in fold_results], device=device)
  fold_avg = mse.mean() # removes the scalar value from the calculated vector ?
  lam_avg_mse[i] = fold_avg

# Find the lambda with the smallest average MSE
lam_best = lam_min_mse_idx = torch.argmin(lam_avg_mse).item()
lam_min_mse = lambdaVals[lam_min_mse_idx]
print(f"\nLAMBDA WITH SMALLEST AVG MSE: {lam_min_mse}")

mse_min = torch.tensor([fold[2] for fold in results[lam_min_mse_idx]], device=device)
test_idx = lam_min_mse_idx + 1
while test_idx < len(results):
  mse_new = torch.tensor([fold[2] for fold in results[test_idx]], device=device)

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

print(f"The most sparse lambda that is not significantly different from the best lambda is {lambdaVals[lam_best]} at index {lam_best}")


################################################################
#### Now run BIOT with the best lambda on the whole dataset ####
################################################################

clf = linear_model.Lasso(alpha=lambdaVals[lam_best].item())
# intializing the rotation matrix
Rotation = torch.randn(Edim, Edim, dtype=torch.float64, device=device)


# preprocess embeddings and features
Features_norm, Embeddings_norm, Features_test, Embeddings_test = ProcessFoldData(X = Embeddings, Fe = Features, testId = foldIds[foldIdx])

dummymse_error = 0
for iter in range(maxiter):
  W, Rotation = BIOT(Embeddings, Features, Rotation, clf)
  mse_error = MSE(Embeddings, Features, Rotation, clf)
  if mse_error - dummymse_error < 0.000001:
    break
  dummymse_error = mse_error



############################################
#### Now Save the results to a CSV file ####
############################################

W = W.cpu().numpy()
Rotation = Rotation.cpu().numpy()

# Save regression weights to a CSV file
np.savetxt(f"{output}/Weights.csv", W, delimiter=",")

# Output centered and scaled X
scaledX = ( Embeddings - torch.mean(Embeddings, dim=0) ).cpu()
np.savetxt(f"{output}/ScaledX.csv", scaledX, delimiter=",")

# Output the rotated mx
RMatrix = ( scaledX @ Rotation ).cpu()
np.savetxt(f"{output}/rMatrix.csv", RMatrix, delimiter=",")
np.savetxt(f"{output}/Rotation.csv", Rotation, delimiter=",")

# Output centered and scaled Features
scaledFe = ( (Features - torch.mean(Features, dim=0)) / torch.std(Features, dim=0) ).cpu()
np.savetxt(f"{output}/scaledFeatures.csv", scaledFe, delimiter=",")

# Calculate and save correlations
cors = np.corrcoef(RMatrix, scaledFe, rowvar=False)
np.savetxt(f"{output}/Cors.csv", cors, delimiter=",")

# Combine matrices and save
combined = np.column_stack((RMatrix, scaledFe))
np.savetxt(f"{output}/combined.csv", combined, delimiter=",")

# Calculate and save projection matrix
WMatrix = ( scaledFe @ W ).cpu()
np.savetxt(f"{output}/pMatrix.csv", WMatrix, delimiter=",")