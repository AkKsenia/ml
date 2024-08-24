import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1.1 визуализация данных

path = '/Users/kseniaaksuk/PycharmProjects/линейная регрессия/ex1data1.txt'
df = pd.read_csv(path, header=None, names=['Население', 'Прибыль'])

plt.scatter(df['Население'], df['Прибыль'], c='black', s=10)
plt.xlabel('Население города в 10000 чел')
plt.ylabel('Прибыль от кафе в 10000$')

plt.show()


# 1.2 реализация функции потерь

# задание матрицы X0 (первый столбец содержит единицы, второй - признаки)
X0 = np.ones((df['Население'].size, 2))
X0[:, 1] = df['Население'].values

# задание вектора параметров (начальные значения параметров выбираем нулевыми)
θ = np.array([0, 0]).T

# вектор выходных данных
y = df['Прибыль']

# объем выборки
m = y.size


# задание гипотезы


def h_θ(θ, X0):
    return np.dot(X0, θ)


# функция потерь

def L(θ):
    return np.sum((h_θ(θ, X0) - y) ** 2) / (2 * m)


# 1.3 реализация градиентного спуска

# задание скорости обучения
α = 0.01

# задание числа итераций
iterations_num = 5000

# градиентный спуск


def grad(θ):
    loss = []

    for i in range(iterations_num):
        loss.append(L(θ))
        θ = θ - (α / m) * np.dot((h_θ(θ, X0) - y).T, X0).T

    return θ, loss


θ_updated, loss = grad(θ)


# 1.4 проверка сходимости алгоритма обучения

plt.plot(loss, c='black')
plt.xlabel('Число итераций')
plt.ylabel('Значения функции потерь')
plt.show()


# 1.5 предсказание дохода от кафе в городе с населением 50000чел., 100000чел.

print('С помощью алгоритма градиентного спуска:')
print('')
print('параметры: ', θ_updated)
print('прибыль в городах:')
print(h_θ(θ_updated, np.array((1, 5))))
print(h_θ(θ_updated, np.array((1, 10))))
print('')


# 1.6 нахождение параметров линейной регрессии с помощью нормального уравнения,
# предсказание прибыли для городов из пункта 1.5,
# сравнение с результатом, полученным с помощью алгоритма градиентного спуска

θ_new = np.dot(np.dot(np.linalg.inv(np.dot(X0.T, X0)), X0.T), y)

print('С помощью нормального уравнения:')
print('')
print('параметры: ', θ_new)
print('прибыль в городах:')
print(h_θ(θ_new, np.array((1, 5))))
print(h_θ(θ_new, np.array((1, 10))))
print('')