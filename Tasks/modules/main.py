import numpy as np
from LinearRegression import *
from Plot import Plot

def main(filename = "../data/time_messagees.txt"):
    y = np.loadtxt(filename,delimiter=',',usecols=[1])
    X = np.expand_dims(np.linspace(0,1,len(y)),1)

    model = get_model()
    model_fit(model,X,y)
    y_pred = model_predict(model,X)
    MSE = get_MSE(y,y_pred)
    print(MSE)
