import typing as t
import torch

def GetRSquared(X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    X: predictor matrix
    W: regression weights
    Y: response matrix

    Returns the R^2. A value close to 0 indicates little correlation, while a value close to 1 indicates the opposite.
    """

    return 1 - (torch.sum((X @ W - Y)**2, dim=0) / torch.sum((Y - torch.mean(Y))**2, dim=0))


def Lasso(
        X: torch.Tensor, 
        Y: torch.Tensor, 
        lambdas: torch.Tensor, 
        maxIter: int=100000, 
        device: t.Literal['cpu', 'gpu'] = 'cpu', 
        alpha: float=0.1, 
        intercept: bool=False, 
        thresh: float=1e-7
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """
    X: predictor matrix
    Y: response matrix
    lambdas: sequence of lambdas for regularization
    maxIter: maximum number of iterations allowed
    device: gpu or cpu?
    alpha: learning rate
    intercept: should the bias be calculated too?
    thresh: convergence criteria

    KEY POINT: Use lambdas of decreasing value. A higher first lambda returns higher r2, 
      although Lasso runs more iterations and takes 10x more time.
    Returns w, the regression weights, and the corresponding r^2 value.
    """

    m, n = X.shape
    l = Y.shape[1]
    w = torch.zeros((n, l), dtype=torch.float64).to(device)
    b = 0
    prev_loss = 0
    nLambdas = lambdas.size(0) - 1

    # Implementing Gradient Descent algorithm for Optimization
    for i in range(maxIter):
        # print(f"Iter {i}")

        # Make prediction
        Y_pred = X @ w 

        # Gradient for bias + updating bias
        if intercept:
            Y_pred += b
            db = -(2 / m) * torch.sum(Y - Y_pred)
            b = b - alpha * db

        # Set lambda index
        idx = nLambdas if i > nLambdas else i

        # Update gradients for weight
        dw = -(2 / m) * X.T @ (Y - Y_pred)
        dw = torch.where( w > 0, dw - ((2 / m) * lambdas[idx]), dw + ((2 / m) * lambdas[idx]))

        # Update the weights
        w = w - alpha * dw

        # Compute loss.... should it be torch.sum(.. , dim=0) ?
        loss = (1 / (2 * m)) * torch.sum((Y - Y_pred) ** 2) + lambdas[idx] * torch.sum(torch.abs(w))

        # Check convergence criterion
        # print(f"Iter : {i}, diff : {prev_loss - loss}, lambda: {lambdas[idx]}")
        if abs(prev_loss - loss) < thresh:  # You can adjust this threshold as needed
            return w, GetRSquared(X, Y, w)

        # Update loss
        prev_loss = loss

    print("WARNING: Convergence criteria not met. Returning regression weights anyway.")
    return w, GetRSquared(X, Y, w)
