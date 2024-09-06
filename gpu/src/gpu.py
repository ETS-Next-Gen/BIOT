import torch




class TorchL1:
    def __init__(self, alpha, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    
    @torch.jit.script
    def fit(X: torch.Tensor, Y: torch.Tensor, lam: torch.Tensor, W: torch.Tensor, max_iter: int = 1000, tol: float = 1e-10) -> torch.Tensor:
        # lasso_coordinate_descent
        
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
    
    def predict(self, X):
        pass
    
    
@torch.jit.script
class TorchL1:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def fit(self, X, y):
        """
        Fit model using coordinate descent for L1-regularized linear regression.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert input to torch tensors
        X_torch = torch.tensor(X, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) if y.ndim == 1 else torch.tensor(y, dtype=torch.float32)

        # Initialize weights
        n_features = X_torch.shape[1]
        n_targets = y_torch.shape[1] if y.ndim == 2 else 1
        
        assert self.coef_ != None, "Weights not initialized."

        # Coordinate descent for L1 regularization
        self.W = self._coordinate_descent(X_torch, y_torch, torch.tensor(self.alpha, dtype=torch.float32), self.W, self.max_iter, self.tol)

        return self

    def _coordinate_descent(self, X, Y, lam, W, max_iter, tol):
        """
        Lasso coordinate descent algorithm.
        
        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training data.
        Y : torch.Tensor of shape (n_samples, n_targets)
            Target values.
        lam : torch.Tensor
            Regularization parameter (lambda).
        W : torch.Tensor of shape (n_features, n_targets)
            Coefficients/weights.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance for stopping criteria.
        
        Returns
        -------
        W : torch.Tensor
            Updated weights.
        """
        _, a = X.shape
        _, b = Y.shape
        summation_ = torch.sum(X ** 2, dim=0)  # (a,)

        for _ in range(max_iter):
            W_old = W.clone()

            for k in range(a):
                if summation_[k] == 0:
                    continue

                residual = Y - torch.matmul(X, W) + torch.matmul(X[:, k].unsqueeze(-1), W[k, :].unsqueeze(0))  # (n, b)
                rho = torch.mv(residual.T, X[:, k]).unsqueeze(-1)  # (b, 1)
                W[k, :] = (torch.sign(rho) * torch.maximum(torch.abs(rho) - lam, torch.zeros_like(rho))).squeeze() / summation_[k]

            # Stopping criteria
            if torch.max(torch.abs(W - W_old)) < tol * torch.max(torch.abs(W)):
                break

        return W

    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        X_torch = torch.tensor(X, dtype=torch.float32)
        y_pred = torch.matmul(X_torch, self.W).numpy()
        return y_pred.squeeze()

    def get_params(self, deep=True):
        """
        Get parameters for
