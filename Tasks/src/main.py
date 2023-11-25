import numpy as np

def main(filename = "../data/time_messagees.txt"):
    y = np.loadtxt(filename,delimiter=',',usecols=[1])
    X = np.expand_dims(np.linspace(0,1,len(y)),1)

    model = LinearRegression.get_model()
    LinearRegression.model_fit(model,X,y)
    y_pred = LinearRegression.model_predict(model,X)
    MSE = LinearRegression.MSE(y,y_pred)
    print(MSE)
