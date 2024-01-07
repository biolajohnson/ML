class Linear_Regression:
  def __init__(self, x, y):
    self._fit(x, y)

  def _fit(self, x, y):
    '''
    Fits the data to the class.
    Args:
      - x: input
      - y: target
    '''
    self.x = x
    self.y = y

  def compute_gradient(self, w, b):
    '''
    This computes the gradient of the cost fucntion and returns the dJ_dw and dJ_db.
      Args:
        - w: weights
        - b: bias
      Return:
      dJ_dw: derivative of the cost function w.r.t to w
      dJ_db: derivative of the cost fucntion w.r.t to b
    '''
    m = len(self.x)
    dJ_dw = 0
    dJ_db = 0

    for i in range(m):
      f_wb = w * self.x[i] + b
      dJ_dw += (f_wb - self.y[i]) * self.x[i]
      dJ_db += f_wb - self.y[i]

    return dJ_dw, dJ_db

  def compute_cost(self, w, b):
    '''
    The cost function used is mean squared error: MSE
      - The sum of the difference between the predicted value (y_hat) and the target value (y)
       divided by 2 times the training set. (2 for easier calculations with the gradient)
      Args:
        - w: weights
        - b: bias 
      Return: total_cost
    '''
    m = len(self.x)
    total_cost = 0

    for i in range(m):
      f_wb = w * self.x[i] + b
      total_cost += (f_wb - self.y[i]) ** 2

    total_cost /= m
    total_cost /= 2
    return total_cost
    

  def gradient_descent(self, w, b, alpha, iter=1000):
    '''
    Runs the gradient descnet algorithm.
    This updates the weights and bias of the model per time slice
    Args:
      w: weights
      b: bias
      alpha: learning rate
      iter: number of iterations
    '''
    cost = self.compute_cost(self.x, self.y, w, b)
    dJ_dw, dJ_db = self.compute_gradient(self.x, self.y, w, b)
    
    for i in range(iter):
      w -= alpha * dJ_dw
      b -= alpha * dJ_db
      if i % 10 == 0:
        print(f"iter: {i}, cost: {cost}")
    return w, b
  
  
  def plot_learning_curve():
    pass

  def plot_model_with_data():
    pass

