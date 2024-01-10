import numpy as np

class LinearRegression:
    def __init__ (self):
        self.b_0 = 0
        self.b_1 = 0

    def fit (self, X, y):
        X_mean = np.mean (X)
        y_mean = np.mean (y)
        ssxy, ssx = 0, 0
        for _ in range (len (X)):
            ssxy += (X[_]-X_mean)*(y[_]-y_mean)
            ssx += (X[_]-X_mean)**2

        self.b_1 = ssxy / ssx
        self.b_0 = y_mean - (self.b_1*X_mean)
        return self.b_0, self.b_1
    
    def predict (self, X):
        y_hat = self.b_0 + (X * self.b_1)
        return y_hat

if __name__ == '__main__':
    X = np.array ([173, 182, 165, 154, 170], ndmin=2)
    X = X.reshape(5, 1)
    y = np.array ([68, 79, 65, 57, 64])
    model = LinearRegression ()
    model.fit (X, y)
    y_pred = model.predict ([161])
    print (y_pred)
    
