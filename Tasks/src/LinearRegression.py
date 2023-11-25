from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_model():
    '''Получение необученной модели машинного обучения'''
    return LinearRegression()

def model_fit(model,X,y):
    '''Обучение модели линейной регрессии'''
    model.fit(X,y)

def model_predict(model,X):
    '''Предсказание модели регрессии'''
    return model.predict(X)

def MSE(y,y_pred):
    '''Средняя квадратичная ошибка'''
    return mean_squared_error(y,y_pred)