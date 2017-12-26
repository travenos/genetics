import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# функция для оптимизации
def func(x1, x2, x3):
    a1 = 1
    a2 = 2
    a3 = 3
    b1 = 2
    b2 = 4
    b3 = 6
    return (x1+a1)**b1 * (x2+a2)**b2 * (x3+a3)**b3 * np.sin(np.pi*(np.fabs(x1)+np.fabs(x2)+np.fabs(x3)))**2


class GeneticEvolution(object):
    """Генетическая оптимизация функции трёх переменных"""

    def __init__(self, func, p_mut=0.8, sigma=1, k_surv=0.2, n_pairs=1000, n_steps=250):
        self.func = func
        self.H = []  # Популяция
        self.sigma = sigma  # СКО при мутации
        self.p_mut = p_mut  # Вероятность мутации
        self.k_surv = k_surv  # Мат. ожидание доли выживших
        self.n_pairs = n_pairs  # Максимальное количество пар
        self.n_steps = n_steps  # Максимальное количество итераций

    # начальная случайная популяция
    def generate_random_population(self, size=100):
        self.H = np.random.rand(size, 3)
        self.H = sorted(self.H, key=lambda X: self.func(*X))
        return self.H

    # инициализация поиска
    def initialize(self):
        self.generate_random_population()

    # основная функция "генетического" поиска
    def evolute(self):
        results = np.zeros(self.n_steps)
        coords = np.zeros((self.n_steps, 3))
        n = 0
        while n < self.n_steps:
            print('Шаг:', n)
            f = self.func(*self.H[0])
            x1 = self.H[0][0]
            x2 = self.H[0][1]
            x3 = self.H[0][2]
            results[n] = f
            coords[n] = x1, x2, x3
            print("x1 =", x1, "; x2 =", x2, "; x3 =", x3, "F(x) =", f)
            n += 1
            ind = 0
            H_new = []  # Популяция следующего поколения
            combs = combinations(range(len(self.H)), 2)
            combs = list(combs)
            np.random.shuffle(combs)
            for comb in combs:
                ind += 1
                if ind > self.n_pairs:
                    break
                a = self.mutate(self.H[comb[0]])  # Первый родитель
                b = self.mutate(self.H[comb[1]])  # Второй родитель
                X_new = self.crossover(a, b)    # Новая особь
                H_new.append(X_new)
            H_new += self.H
            self.H = self.killing(H_new)

        # Построение графиков
        plt.plot(np.arange(0, self.n_steps), results)
        plt.grid(True)
        plt.xlabel("Iteration number")
        plt.ylabel("F(X)")
        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], label='Coordinates of minimum curve')
        ax.legend()
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        plt.show()

        return self.func(*self.H[0])

    def killing(self, H):
        H = sorted(H, key=lambda X: self.func(*X))
        L_new = np.random.poisson(int(len(H) * self.k_surv))
        return H[:L_new]

    # обмен значениями (кроссинговер)
    def crossover(self, a, b):
        combs = [[a[0], a[1], b[2]],
                 [a[0], b[1], a[2]],
                 [b[0], a[1], a[2]],
                 [b[0], b[1], a[2]],
                 [b[0], a[1], b[2]],
                 [a[0], b[1], b[2]]]
        return combs[np.random.randint(0, len(combs))]

    # мутация значений происходит с определенной вероятностью
    def mutate(self, a):
        if np.random.rand() < self.p_mut:
            new_a = np.random.normal(loc=a, scale=self.sigma)
        else:
            new_a = a
        return new_a


g = GeneticEvolution(func=func)
g.initialize()
res = g.evolute()
print('Результат оптимизации:', res)
