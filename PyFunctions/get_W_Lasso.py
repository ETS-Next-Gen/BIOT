import torch

def GetWLasso (X, Y, lam):
  '''
  X: predictor matrix
  Y: response matrix
  lambda: Lasso hyperparameter
  '''
  model = Lasso_Regression(learning_rate = 0.02, 
                          no_of_iterations=10000,
                         lambda_parameter=lam)
  
  W = torch.zeros((X.shape[1], Y.shape[1]), dtype=torch.float64)
  for k in range(0, Y.shape[1]):
    print(f"Processing weight {k}")
    model.fit(X, Y[:, k].unsqueeze(1))
    W[:, k] = model.getWeights()

  return ( W, GetRSquared(X, Y, W) )

  
class Lasso_Regression():

  #initiating the hyperparameters
  def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
    self.lambda_parameter = lambda_parameter


  # fitting the dataset to the Lasso Regression model
  def fit(self, X, Y):
    self.m, self.n = X.shape
    self.w = torch.zeros((self.n, 1), dtype=torch.float64).to(self.device)
    self.b = 0
    self.X = X
    self.Y = Y

    # implementing Gradient Descent algorithm for Optimization
    for i in range(self.no_of_iterations):
      self.update_weights()


  # function for updating the weight & bias value
  def update_weights(self):

    # linear equation of the model
    Y_prediction = self.predict(self.X)

    # gradient for weight
    dw = torch.zeros((self.n, 1), dtype=torch.float64).to(self.device)

    for i in range(self.n):
      if self.w[i]>0:
        dw[i] = (-(2*(self.X[:,i].t()) @ (self.Y - Y_prediction)) + self.lambda_parameter) / self.m
      else :
        dw[i] = (-(2*(self.X[:,i].t()) @ (self.Y - Y_prediction)) - self.lambda_parameter) / self.m

    # gradient for bias
    db = - 2 * torch.sum(self.Y - Y_prediction) / self.m

    # updating the weights & bias
    self.w = self.w - self.learning_rate*dw
    self.b = self.b - self.learning_rate*db

  # Predicting the Target variable
  def predict(self,X):
    return X @ self.w + self.b
  
  def getWeights(self):
    return self.w[:,0]
  

def GetRSquared(X, Y, W):
  '''
  X: predictor matrix
  W: regression weights
  Y: response matrix
  '''
  
  return 1 - (torch.sum((X @ W - Y)**2) / torch.sum((Y - torch.mean(Y))**2))