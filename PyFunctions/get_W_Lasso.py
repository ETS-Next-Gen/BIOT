import typing as t
import torch


def GetWLasso(
    X: torch.Tensor, Y: torch.Tensor, lam: float, device: str = "cpu"
) -> t.Tuple[torch.Tensor, float]:
    """
    X: predictor matrix
    Y: response matrix
    lambda: Lasso hyperparameter
    """
    model = Lasso_Regression(
        learning_rate=0.02, no_of_iterations=10000, lam=lam, device=device
    )

    W = (torch.stack([model.fit(X, col) for col in Y.T]).to(device)).T
    # W = torch.zeros((X.shape[1], Y.shape[1]), dtype=torch.float64)
    # for k in range(0, Y.shape[1]):
    #   print(f"Processing weight {k}")
    #   W[:, k] = model.fit(X, Y[:, k].unsqueeze(1))

    return (W, GetRSquared(X, Y, W))


class Lasso_Regression:

    # initiating the hyperparameters
    def __init__(
        self,
        learning_rate: float,
        no_of_iterations: int,
        lam: float,
        device: str,
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lam = lam

    # fitting the dataset to the Lasso Regression model
    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        self.m, self.n = X.shape
        self.w = torch.zeros((self.n, 1), dtype=torch.float64).to(self.device)
        self.b = 0
        self.X = X
        self.Y = Y

        # TODO: compare difference to epsilon to test for convergence ?
        # implementing Gradient Descent algorithm for Optimization
        for i in range(self.no_of_iterations):

            # Make prediction
            Y_pred = self.X @ self.w + self.b

            # gradient for weight
            # for i in range(self.n):
            #   if self.w[i]>0:
            #     dw[i] = (-(2*(self.X[:,i].t()) @ (self.Y - Y_pred)) + self.lam) / self.m
            #   else :
            #     dw[i] = (-(2*(self.X[:,i].t()) @ (self.Y - Y_pred)) - self.lam) / self.m
            dw = (-(2 * (self.X.T @ (self.Y - Y_pred))) - self.lam) / self.m
            dw = torch.where(self.w > 0, dw + (2 * self.lam) / self.m, dw)

            # gradient for bias
            db = -2 * torch.sum(self.Y - Y_pred) / self.m

            # updating the weights & bias
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

        return self.w[:, 0]


def GetRSquared(X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor) -> float:
    """
    X: predictor matrix
    W: regression weights
    Y: response matrix
    """

    return 1 - (torch.sum((X @ W - Y) ** 2) / torch.sum((Y - torch.mean(Y)) ** 2))
