import numpy as np

class Logistic_Regression():

    # initiating the object with the Hyperparameters
    def __init__(self, learning_rate, no_of_iterrations):
        self.learning_rate = learning_rate
        self.no_of_iterrations = no_of_iterrations

    def fit(self, X, Y):
        # m => total number of data points (rows)
        # n => total number of input features
        self.m, self.n = X.shape

        # initializing weight & bias values
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        #implementing Gradient Descent
        for i in range(self.no_of_iterrations):
            self.update_weights()

    def update_weights(self):
        #Y_hat formula (sigmoid function)
        Y_hat = 1 / (1 + np.exp( - (self.X.dot(self.w) + self.b) )) # w * X + b

        # derivatives
        dw = (1 / self.m ) * np.dot(self.X.T, (Y_hat - self.Y))
        db = (1 / self.m) * np.sum(Y_hat - self.Y)

        # updating the weight and bias using Gradient Descent
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
        

    def predict(self, X):
        Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b) )) # w * X + b
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred
