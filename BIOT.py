##############################################
####### Welcome to speedy BIOT, folks! #######
##############################################

import argparse
import numpy as np
import torch
from PyFunctions.process_fold_data import ProcessFoldData
from PyFunctions.run_BIOT import RunBIOT
from PyFunctions.get_W_Lasso import GetWLasso
from PyFunctions.eval import Eval

# DEFAULT FILE PATHS
FunctionPath = "PyFunctions/"
DatasetPath = "Datasets/dataset.csv"
EmbeddingPath = "Datasets/embedding.csv"
OutPath = "Results/result.RData"

# DEFAULT VARIABLE VALUES
Nlambdas = 10
MinLambda = 0.0001
MaxLambda = 3.5 
K = 10            # no of folds used for cross validation
sigThresh = .05   # sigma threshold ?

# MAIN FUNCTION
def main(
  dataPath: str,
  embeddingPath: str,
  outPath: str,
  nlambdas: int,
  minLambda: int,
  maxLambda: int
):

  # Define lambda vector, feature vector, and embedding vector
  lambdaVals = np.exp(np.linspace(np.log(minLambda), np.log(maxLambda), nlambdas))
  X = np.genfromtxt(embeddingPath, delimiter=',', dtype='float128')
  Fe = np.genfromtxt(dataPath, delimiter=',', skip_header=1, dtype='float128')

  ##############################################
  #### Run BIOT for different lambda values ####
  ##############################################

  print("Selection of lambda in progress...")

  # Set seed for random generation of data folds
  np.random.seed(155000)
  results = []

  # Perform cross validation for each lambda
  for lambdaIdx in range(0, nlambdas):
  
    # Select lambda value 
    lam = lambdaVals[lambdaIdx]
    print('Processing lambda index ', lambdaIdx)

    # Split data into folds
    foldIds = np.array_split(np.random.permutation(Fe.shape[0]), K)

    # Cross validation!
    fold_results = []
    for foldIdx in range(0, K):
      print('\t\tProcessing fold index ', foldIdx)

      # Process fold data
      Fe_norm, X_norm, Fe_test, X_test = ProcessFoldData(X = X, Fe = Fe, testId = foldIds[foldIdx])

      # Normalize lambda
      lam_norm = lam / np.sqrt(Fe_norm.shape[1])

      # Get rotation matrix and weights
      R, W, crit, iter, r2 = RunBIOT(X = X_norm, 
                   Fe = Fe_norm,
                   lam = lam_norm, rotation=True)
      print(r2)
      
      # Evaluate results
      MSE, perc_nonzero = Eval(R = R, 
                W = W, 
                Fe_test = Fe_test,
                X_test = X_test)
      
      if MSE is not None:
        fold_results[foldIdx] = [lam, lam_norm, (MSE, perc_nonzero)]

    results[lambdaIdx] = fold_results


####################################
#### Now choose the best lambda ####
####################################
    
# TODO

################################################################
#### Now run BIOT with the best lambda on the whole dataset ####
################################################################
    
# TODO


####################################
#### Run MAIN                   ####
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
  main(
    args.dataPath,
    args.embeddingPath,
    args.outPath,
    args.nlambdas,
    args.minLambda,
    args.maxLambda
  )
