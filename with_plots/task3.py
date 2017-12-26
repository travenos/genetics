import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from collections import namedtuple


class GeneticEvolution(object):
    """Генетический алгоритм решения задачи о рюкзаке"""

    def __init__(self, G, w_max, k_mut=0.2, k_surv=0.15, l_min=5, n_pairs=500, n_steps=500):
        if self._get_weight(G) <= w_max:
            raise ValueError("Суммарная масса вещей должна быть больше грузоподъёмности")
        self.G = G  # Множество доступных предметов
        self.w_max = w_max  # Максимальный вес рюкзака
        self.H = []  # Текущая популяция
        self.k_mut = k_mut  # Доля рюкзака, наполняемая случайными предметами (мутация)
        self.k_surv = k_surv  # Мат. ожидание доли выживших
        self.l_min = l_min  # Минимальный размер популяции
        self.n_pairs = n_pairs  # Максимальное количество пар
        self.n_steps = n_steps  # Максимальное количество итераций

    @staticmethod
    def _get_profit(bag):
        """Узнать полезность рюкзака"""
        return sum([x[0] for x in bag])

    @staticmethod
    def _get_weight(bag):
        """Узнать вес рюкзака"""
        return sum([x[1] for x in bag])

    # начальная случайная популяция
    def generate_random_population(self):
        size = self.l_min
        self.H = []
        for _ in range(size):
            bag = set()
            while True:
                good = self.G[np.random.randint(0, len(self.G))]
                bag.add(good)
                if self._get_weight(bag) > self.w_max:
                    bag.remove(good)
                    break
            self.H.append(bag)
        self.H = sorted(self.H, key=self._get_profit, reverse=True)

    # инициализация поиска
    def initialize(self):
        self.generate_random_population()

    def evolute(self):
        """Основная функция "генетического" поиска"""
        results = np.zeros(self.n_steps)
        n = 0
        while n < self.n_steps:
            print('Шаг:', n)
            profit = self._get_profit(self.H[0])
            results[n] = profit
            print("Набор вещей:", self.H[0], "\nПолезность:", profit)
            n += 1
            ind = 0
            H_new = []
            combs = combinations(range(len(self.H)), 2)
            combs = list(combs)
            np.random.shuffle(combs)
            for comb in combs:
                ind += 1
                if ind > self.n_pairs:
                    break
                a = self.H[comb[0]]
                b = self.H[comb[1]]
                new_item = self.crossover(a, b)
                new_item = self.mutate(new_item)
                H_new.append(new_item)
            H_new += self.H
            self.H = self.killing(H_new)

        # Построение графика
        plt.plot(np.arange(0, self.n_steps), results)
        plt.grid(True)
        plt.xlabel("Iteration number")
        plt.ylabel("Backpack profit")
        plt.show()

        return np.max([self._get_profit(bag) for bag in self.H])

    def killing(self, population):
        """Селекция"""
        population = sorted(population, key=self._get_profit, reverse=True)
        end = np.random.poisson(int(len(population) * self.k_surv))
        if end < self.l_min:
            end = self.l_min
        survived_population = population[0: end]
        return survived_population

    def crossover(self, a, b):
        """Скрещивание (кроссинговер) - выбор случайных предметов из родителей"""
        bag = set()
        while True:
            if a.issubset(bag):
                break
            good = list(a)[np.random.randint(0, len(a))]
            bag.add(good)
            if self._get_weight(bag) > self.w_max * (1 - self.k_mut) / 2:
                bag.remove(good)
                break
        while True:
            if b.issubset(bag):
                break
            good = list(b)[np.random.randint(0, len(b))]
            bag.add(good)
            if self._get_weight(bag) > self.w_max * (1 - self.k_mut):
                bag.remove(good)
                break
        return bag

    def mutate(self, bag):
        """Мутация - добавление случайных предметов"""
        while True:
            good = self.G[np.random.randint(0, len(self.G))]
            bag.add(good)
            if self._get_weight(bag) > self.w_max:
                bag.remove(good)
                break
        return bag


Good = namedtuple('Good', ['profit', 'weight', 'name'])

GOODS = [Good(10.1, 3, 'Ноутбук'), Good(8.2, 0.6, 'Вода'), Good(15.6, 0.1, 'Пенал'), Good(12.9, 1.5, 'Конспекты'),
         Good(6.7, 1.5, 'Комплект формы'), Good(15.15, 1, 'Спортивная форма'), Good(10.1, 0.3, 'Полотенца'),
         Good(18.9, 2, 'Туалетные принадлежности'), Good(12, 0.5, "Щётка и крем для обуви"),
         Good(18.3, 0.4, "Шлёпанцы"), Good(14.4, 0.1, "Подворотнички"), Good(15.56, 0.5, "Нательное бельё"),
         Good(10.9, 0.2, "Медикаменты"), Good(5.12, 0.8, "Свитер"), Good(4.3, 1, "Радиоприёмник"),
         Good(2.2, 3, "Гитара"), Good(8.5, 0.7, "Книга"), Good(7.67, 0.5, "Фотоаппарат"), Good(15, 0.4, "Мобильник"),
         Good(2.7, 1, "Видеокамера"), Good(6.56, 4, "Утюг"), Good(5.1, 1.5, "Чайник"), Good(1, 0.5, "Будильник"),
         Good(20.4, 8, "Гантели"), Good(3.9, 0.6, "Мультиметр"), Good(7.4, 1, "Светильник"),
         Good(2.8, 0.8, "Коллекция карточек"), Good(40, 3, "Дрель")]

MAX_WEIGHT = 8

g = GeneticEvolution(G=GOODS, w_max=MAX_WEIGHT)
g.initialize()
result = g.evolute()
print('Результат оптимизации:', result)
