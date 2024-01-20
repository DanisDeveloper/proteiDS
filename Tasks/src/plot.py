import matplotlib.pyplot as plt
import numpy as np


def plot(x,y,y_pred):
    """Вывод точечного графика и графика функции линейной регрессии
    x - исходные значения
    y - целевые значения
    y_pred - предсказанные моделью значения
    """
    plt.plot(x, y_pred, c='r')
    plt.scatter(x, y, s=1)
    plt.xlabel('Время')
    plt.ylabel('Количество сообщений')
    labels = ['00:00', '06:00', '12:00', '18:00', '24:00']
    plt.xticks(np.linspace(0, 1, len(labels), dtype=float), labels=labels)
    plt.legend(['Линейная регрессия','Количество сообщений'])
    plt.show();