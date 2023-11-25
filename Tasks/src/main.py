import numpy as np
from src import LinearRegression
from src import Plot

def main(filename):
    '''Обучение модели линейной регрессии, вывод графика и расчет средней квадратичной ошибки'''
    y = np.loadtxt(filename,delimiter=',',usecols=[1])
    X = np.expand_dims(np.linspace(0,1,len(y)),1)

    model = LinearRegression.get_model()
    LinearRegression.model_fit(model,X,y)
    y_pred = LinearRegression.model_predict(model,X)
    
    Plot.plot(X,y,y_pred)
    
    MSE = LinearRegression.MSE(y,y_pred)
    print('MSE = ',MSE)