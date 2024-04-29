##############################################
####### Welcome to speedy BIOT, folks! #######
##############################################

import argparse
import numpy as np
import torch
from scipy.stats import wilcoxon
from PyFunctions.process_fold_data import ProcessFoldData
from PyFunctions.run_BIOT import RunBIOT
from PyFunctions.get_MSE_pred import GetMSEPred


# DEFAULT FILE PATHS
# NOTE: this code assumes that the first line in the DatasetPath file is a header to skip.
FunctionPath = "PyFunctions/"
DatasetPath = "Datasets/dataset.csv"
EmbeddingPath = "Datasets/embedding.csv"
OutPath = "Results/"

# DEFAULT VARIABLE VALUES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     # checks for gpu
Nlambdas = 5
MinLambda = .0001
MaxLambda = 3.5 
K = 3            # no of folds used for cross validation
sigThresh = .05   # sigma threshold ?

# HELPER FUNCTION
def Eval(R, W, Fe_test, X_test):
  '''  
  R: orthogonal transformation matrix
  W: regression weights
  Fe.test: external feature matrix (predictors), test set only
  X.test: embedding matrix (response), test set only

  Returns MSE between XR and FeW and percent nonzero of W
  '''

  MSE = GetMSEPred(Fe = Fe_test, 
                    X = X_test,
                    R = R, 
                    W = W)
  
  percent_nonzero = torch.count_nonzero(W) / torch.prod(torch.tensor(W.shape))

  return (MSE, percent_nonzero)


# MAIN FUNCTION
def main(
  dataPath: str,
  embeddingPath: str,
  outPath: str,
  nLambdas: int,
  minLambda: int,
  maxLambda: int
):
  
  # Define lambda vector, feature vector, and embedding vector
  lambdaVals = torch.exp(torch.linspace(np.log(minLambda), np.log(maxLambda), nLambdas)).to(device)
  # print(lambdaVals)
  X = torch.tensor(np.genfromtxt(embeddingPath, delimiter=',', dtype='float64'), device=device)
  Fe =  torch.tensor(np.genfromtxt(dataPath, delimiter=',', skip_header=1, dtype='float64'), device=device)
  nFoldCols = torch.tensor(Fe.shape[1], device=device)

  ##############################################
  #### Run BIOT for different lambda values ####
  ##############################################

  print("Selection of lambda in progress...")

  # Set seed for random generation of data folds, then generate fold IDs
  foldIds = torch.split(torch.randperm(Fe.size(0)), Fe.size(0) // K)
  results = []

  # Perform cross validation for each lambda
  for lam in lambdaVals:
  
    # Select lambda value 
    print('Processing lambda: ', lam)

    # Cross validation!
    fold_results = []
    for foldIdx in range(0, K):
      print('\tProcessing fold index ', foldIdx)

      # Data pre-processing
      Fe_norm, X_norm, Fe_test, X_test = ProcessFoldData(X = X, Fe = Fe, testId = foldIds[foldIdx])

      # Normalize lambda
      lam_norm = lam / torch.sqrt(nFoldCols)

      # Get rotation matrix and weights
      R, W, crit, iter, r2 = RunBIOT(X = X_norm, 
                   Fe = Fe_norm, device=device,
                   lam = lam_norm, maxIter=2, rotation=True)
      
      # Evaluate results
      MSE, perc_nonzero = Eval(R = R, 
                W = W, 
                Fe_test = Fe_test,
                X_test = X_test)
      
      # Make sure MSE is valid
      if MSE is not None:
        fold_results.append([lam, lam_norm, MSE, perc_nonzero])

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
  print(lam_avg_mse)
  
  # Find the lambda with the smallest average MSE
  lam_best = lam_min_mse_idx = torch.argmin(lam_avg_mse).item()
  print(f"\nLAMBDA WITH SMALLEST AVG MSE: {lambdaVals[lam_min_mse_idx]}")

  mse_min = torch.tensor([fold[2] for fold in results[lam_min_mse_idx]], device=device)
  test_idx = lam_min_mse_idx + 1
  while test_idx < nLambdas:
    mse_new = torch.tensor([fold[2] for fold in results[test_idx]], device=device)

    if torch.sum(torch.abs(mse_min) - torch.abs(mse_new)) == 0:
      pval = 1
    else:
      pval = wilcoxon(mse_min.numpy(), mse_new.numpy(), alternative='two-sided')[1]  

    if pval <= sigThresh:
      lam_best = test_idx - 1
      break
    if pval > sigThresh and test_idx + 1 == Nlambdas:
      lam_best = test_idx
      break

    test_idx += 1

  print(f"The most sparse lambda that is not significantly different from the best lambda is {lambdaVals[lam_best]} at index {lam_best}")


  ################################################################
  #### Now run BIOT with the best lambda on the whole dataset ####
  ################################################################
  
  # Normalize entire dataset
  Fe_mean = torch.mean(Fe, dim=0)
  Fe_sd = torch.std(Fe, dim=0)
  X_mean = torch.mean(X, dim=0)
  Fe_norm = (Fe - Fe_mean) / Fe_sd
  X_norm = X - X_mean

  R, W, crit, iter, r2 = RunBIOT(X = X_norm, 
                   Fe = Fe_norm, device=device,
                   lam = lambdaVals[lam_best], rotation=True)

  # Save regression weights to a CSV file
  np.savetxt(f"{outPath}/Weights.csv", W.cpu.numpy(), delimiter=",")

  # Save R-squared to a separate CSV file
  np.savetxt(f"{outPath}/R2.csv", r2.cpu().numpy(), delimiter=",")

  # Output centered and scaled X
  scaledX = X - torch.mean(X.cpu().numpy(), dim=0)
  np.savetxt(f"{outPath}/ScaledX.csv", scaledX, delimiter=",")

  # Output the rotated mx
  RMatrix = scaledX @ R
  np.savetxt(f"{outPath}/rMatrix.csv", RMatrix.cpu().numpy(), delimiter=",")
  np.savetxt(f"{outPath}/Rotation.csv", R, delimiter=",")

  # Output centered and scaled Features
  scaledFe = (Fe - torch.mean(Fe, dim=0)) / torch.std(Fe, dim=0)
  np.savetxt(f"{outPath}/scaledFeatures.csv", scaledFe.cpu().numpy(), delimiter=",")

  # Calculate and save correlations
  cors = np.corrcoef(RMatrix.cpu().numpy(), scaledFe.cpu().numpy(), rowvar=False)
  np.savetxt(f"{outPath}/Cors.csv", cors, delimiter=",")

  # Combine matrices and save
  combined = np.column_stack((RMatrix.cpu().numpy(), scaledFe.cpu().numpy()))
  np.savetxt(f"{outPath}/combined.csv", combined, delimiter=",")

  # Calculate and save projection matrix
  WMatrix = scaledFe @ W
  np.savetxt(f"{outPath}/pMatrix.csv", WMatrix.cpu().numpy(), delimiter=",")


####################################
####          Run MAIN          ####
####################################
  
if __name__ == "__main__":

  # parse user arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--dataPath",
                      default=DatasetPath)
  parser.add_argument("-e", "--embeddingPath",
                      default=EmbeddingPath)
  parser.add_argument("-o", "--outPath", 
                      default=OutPath)
  parser.add_argument("-n", "--nlambdas",
                      default=Nlambdas)
  parser.add_argument("-m", "--minLambda",
                      default=MinLambda)
  parser.add_argument("-x", "--maxLambda",
                      default=MaxLambda)
  args = parser.parse_args()

  # run main!
  import time
  iter = 30
  times = []
  for i in range(iter):

    s = time.time()

    main(
      args.dataPath,
      args.embeddingPath,
      args.outPath,
      args.nlambdas,
      args.minLambda,
      args.maxLambda
    )

    print(time.time() - s)
    times.append(time.time() - s)

  np.savetxt(f"BIOT_times.csv", times, delimiter=",")

  
  

