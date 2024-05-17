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
    return (1 / (2 * torch.numel(X))) * torch.sum((X @ R - Fe @ W)**2)
