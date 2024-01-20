from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_model():
    """Получение необученной модели машинного обучения"""
    return LinearRegression()

def model_fit(model, x, y):
    """Обучение модели линейной регрессии
    model - модель линейной регрессии
    x - исходные значения
    y - целевые значения
    """
    model.fit(x, y)

def model_predict(model, x):
    """Предсказание модели регрессии
    model - модель линейной регрессии
    x - исходные значения
    """
    return model.predict(x)

def MSE(y, y_pred):
    """Средняя квадратичная ошибка
    y - реальные данные
    y_pred - предсказанные моделью данные
    """
    return mean_squared_error(y, y_pred)