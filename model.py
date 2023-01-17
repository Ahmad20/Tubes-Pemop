import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        _, n_features = X.shape

        y_ = np.where(y<=0, -1, 1)

        #init weight
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(np.array(X)):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * 2 * self.lambda_param * self.w - np.dot(y_[idx], x_i)
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

class Perceptron:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.w = None
        self.b = None
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_function = self._unit_step_function
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        #init weight
        self.w = np.zeros(n_features)
        self.b = 0

        y_ = np.where(y<=0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.w) + self.b
                y_predicted = self.activation_function(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.w += update * x_i
                self.b += update

    def predict(self, X):
        linear_output = np.dot(X, self.w)
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    def _unit_step_function(self, x):
        return np.where(x>0, 1, -1)