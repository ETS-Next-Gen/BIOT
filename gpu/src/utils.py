import torch
import typing as t
import numpy as np

# normalize the data with zero mean and 1 std 
def normalize(X, mean, std):
  return (X - mean) / std

def scale(X):
  return (X - np.min(X)) / np.std(X)


def ProcessFoldData(X: torch.Tensor, Fe: torch.Tensor, testId: torch.Tensor, CV=False, num=10000000000, only_standardize=False) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    X: embedding matrix (response)
    Fe: external feature matrix (predictors)
    test.id: vector of integers indicating the rows of X and Fe to assign to the test set
    dummy: vector whose elements = T if the corresponding column in Fe is a dummy variable, F otherwise
    """
    if only_standardize:
      
      Fe = normalize(Fe, Fe.mean(dim=0, keepdim=True), Fe.std(dim=0, keepdim=True))
      X = normalize(X, X.mean(dim=0, keepdim=True), 1)
      
      return ( Fe.to(torch.float32), X.to(torch.float32) )
      
      
    # Gathering train IDs
    uniques, counts = torch.cat((torch.arange(Fe.shape[0], device="cpu"), testId)).unique(return_counts=True)
    train_id = uniques[counts == 1]

    # Gathering test data
    Fe_test = Fe[testId, :]
    X_test = X[testId, :]

    # Gather train data
    Fe_train = Fe[train_id]
    X_train = X[train_id]
    
    
    mean = Fe_train.mean(dim=0, keepdim=True)
    std = Fe_train.std(dim=0, keepdim=True)
    
    Fe_train = normalize(Fe_train, mean, std)
    Fe_test = normalize(Fe_test, mean, std)
    
    mean = X_train.mean(dim=0, keepdim=True)
    std = 1
    
    X_train = normalize(X_train, mean, std)
    X_test = normalize(X_test, mean, std)

    if CV:
      count = min(num, Fe_train.shape[0])
      count = torch.randperm(Fe_train.shape[0])[:count] # Get me num random samples from the training set 
      Fe_train = Fe_train[count,:]
      X_train = X_train[count,:]
      
      # Fe_test = Fe_test[:count,:]
      # X_test = X_test[:count,:]
      
    return ( Fe_train.to(torch.float32), X_train.to(torch.float32), Fe_test.to(torch.float32), X_test.to(torch.float32) ) 


def MSE(Embeddings, Features, Rotation, W, lam_norm):
  Yp= torch.matmul(Features,W)
  Y = torch.matmul(Embeddings,Rotation)
  mse = torch.mean( torch.mean((Y - Yp)**2 , dim=1) ) / 2
  
  reg = torch.sum(torch.abs(W))*lam_norm
  
  return mse.item(), reg.item() 


def L1(Embeddings, Features, Rotation, W, lam_norm):
  Yp= torch.matmul(Features,W)
  Y = torch.matmul(Embeddings,Rotation)
  l1 = torch.mean( torch.mean(torch.abs((Y - Yp)) , dim=1) ) / 2
  reg = torch.sum(torch.abs(W))*lam_norm

  return l1.item(), reg.item() 


def SVD(Features, W, Embeddings):
  # Creating the decomposed matrix usng Fe, W and X
  n = Features.size(0)
  De = (1 / (2 * n)) * torch.mm(Embeddings.T, torch.mm(Features, W))
  
  # Get the SVD
  U, S, Vh = torch.svd(De)
  
  sv = S
  smallest = torch.argmin(sv)
  sv = sv.clone()  # Clone to modify
  sv[smallest] = torch.sign(torch.det(torch.mm(U, Vh.T)))
  sv[sv != sv[smallest]] = 1
  
  # Construct the rotation matrix
  R = torch.mm(U, torch.mm(torch.diag(sv), Vh.T))
  
  return R