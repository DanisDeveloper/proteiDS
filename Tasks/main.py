import numpy as np
from src import linear_regression
from src import plot
import sys


if __name__ == "__main__":
    y = np.loadtxt(sys.argv[1], delimiter=',', usecols=[1])
    x = np.expand_dims(np.linspace(0,1,len(y)),1)

    model = linear_regression.get_model()
    linear_regression.model_fit(model, x, y)
    y_pred = linear_regression.model_predict(model, x)
    
    plot.plot(x, y, y_pred)
    
    mse = linear_regression.MSE(y, y_pred)
    print('MSE = ', mse)