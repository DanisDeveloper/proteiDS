from matplotlib import pyplot as plt

def plot(X,y,y_pred):
    '''Вывод точечного графика и графика функции линейной регрессии'''
    plt.figure(figsize=(12,8),dpi=200)
    plt.plot(X,y_pred,c='r')
    plt.scatter(X,y,s=1)
    plt.xlabel('Время')
    plt.ylabel('Количество сообщений')
    labels = ['00:00','06:00','12:00','18:00','24:00']
    plt.xticks(np.linspace(0,1,len(labels),dtype=float),labels=labels)
    plt.legend(['Линейная регрессия','Количество сообщений'])