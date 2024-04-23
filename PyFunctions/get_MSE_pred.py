import torch

def GetMSEPred(
    Fe: torch.Tensor, X: torch.Tensor, R: torch.Tensor, W: torch.Tensor
) -> float:  
    """
    X: embedding matrix (response)
    Fe: external feature matrix (predictors)
    R: orthogonal transformation matrix
    W: regression weights

    Returns the MSE
    """
    nrow, ncol = X.shape

    return (1 / (2 * nrow * ncol)) * torch.sum((X @ R - Fe @ W)**2)
