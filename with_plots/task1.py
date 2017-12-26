import numpy as np
import matplotlib.pyplot as plt


class Population(object):
    """
    Класс эволюционных вычислений
    """
    def __init__(self, fit_func, decode_func, population_size=10, gen_length=30, k_pareto=4,
                 p_mut=0.8, a_select=0.9):
        self.gen_length = gen_length  # Длина вектора генов
        self.k_pareto = k_pareto  # Параметр функции распределения Парето при выборе родителей
        self.H = []  # Массив, хранящий текущую популяцию
        self.population_size = population_size  # Фиксированный размер популяции
        self.p_mut_gene = 1 - (1 - p_mut) ** (1 / gen_length)  # Вероятность мутации каждого гена
        self.fit_func = fit_func  # Функция приспособленности
        self.decode_func = decode_func  # Функция декодирования кода особи
        # Формирование первой популяции
        for _ in range(population_size):
            self.H.append(np.random.randint(0, 2, gen_length).tolist())
        self.H = sorted(self.H, key=self.get_fitness, reverse=True)  # Сортировка массива по функции приспособленности
        # Формирование массива вероятности выбора особи
        N = population_size * 2
        self.probabilities = [a_select * (N-i)/N + 1 - a_select for i in range(N)]

    def get_fitness(self, x):
        """Получить значение приспособленности особи"""
        return self.fit_func(x)

    def print_info(self):
        """Напечатать информацию о текущей популяции"""
        for x in self.H:
            for g in x:
                print(g, end="")
            print("=", end="")
            print(self.decode_func(x), end="")
            print("; F=", end="")
            print(self.get_fitness(x))

    def evolute(self, n_steps):
        """Основная функция генетического поиска"""
        self.print_info()
        result = [self.get_fitness(self.H[0])]
        for i in range(n_steps):
            H_new = []  # Массив новой популяции
            for _ in range(self.population_size):  # Создаётся столько же потомков, сколько и особей в популяции
                x_a, x_b = self._choose_pair()  # Выбор пары
                x_new = self._crossover(x_a, x_b)  # Скрещивание
                x_new = self._mutate(x_new)  # Мутациия
                H_new.append(x_new)  # Добавление новой особи в популяцию
            H_new = self.H + H_new  # Добавление старых особей в популяцию

            #Построение графиков
            plt.grid(True)
            plt.xlabel("x1")
            plt.ylabel("x2")
            if i == 0:
                x = np.array(list(map(self.decode_func, H_new)))
                plt.plot(x[:, 0], x[:, 1], 'go')
            elif i == int(n_steps/3):
                x = np.array(list(map(self.decode_func, H_new)))
                plt.plot(x[:, 0], x[:, 1], 'bo')
            elif i == n_steps - 1:
                x = np.array(list(map(self.decode_func, H_new)))
                plt.plot(x[:, 0], x[:, 1], 'ro')

            self.H = self._selection(H_new)  # Селекция

            result.append(self.get_fitness(self.H[0]))

            print("\n", i)
            self.print_info()
        plt.show()

        plt.close()
        plt.grid(True)
        plt.xlabel("Iteration number")
        plt.ylabel("F(X)")
        plt.plot(result)
        plt.show()

    def _choose_pair(self):
        """Выбор пары"""
        # Выбор первого родитея. Используется распределение Парето.
        # Массив популяции отсортирован. Чем выше приспособленность, тем больше вероятность стать родителем
        x_a_number = int(np.random.pareto(self.k_pareto) * self.population_size)
        if x_a_number > self.population_size - 1:  # Если номер родителя вышел за пределы массива
            x_a_number = self.population_size - 1
        # Выбор второго родитея
        x_b_number = int(np.random.pareto(self.k_pareto) * self.population_size)
        if x_b_number > self.population_size - 1:
            x_b_number = self.population_size - 1

        while x_a_number == x_b_number:  # Обоими родителями не может быть одна и та же особь
            x_b_number = int(np.random.pareto(self.k_pareto) * self.population_size)  # Поиск другой особи
            if x_b_number > self.population_size - 1:
                x_b_number = self.population_size - 1

        return [self.H[x_a_number], self.H[x_b_number]]

    @staticmethod
    def _crossover(x_a, x_b):
        """Скрещивание"""
        # С вероятностью 0,5 выбираются гены первого или второго родителя
        return [a if np.random.sample() >= 0.5 else b for a, b in zip(x_a, x_b)]

    def _mutate(self, x):
        """Мутация"""
        # С вероятностью p_mut_gene происходит инверсия гена
        return [g if np.random.sample() >= self.p_mut_gene else self._inverse(g) for g in x]

    @staticmethod
    def _inverse(g):
        """Инверсия гена, заданного целым числом 1 или 0"""
        if g == 0:
            return 1
        else:
            return 0

    def _selection(self, H):
        """Селекция"""
        # Сортируем массив популяции по убыванию приспособленности
        H = sorted(H, key=self.get_fitness, reverse=True)
        mask = [1 if np.random.sample() < prob else 0 for prob in self.probabilities]
        # Если методом рулетки добавлено недостаточно особей, добавляем лучшие
        i = 0
        while sum(mask) < self.population_size:
            mask[i] = 1
            i += 1
        # Добавляем выбранные особи в нвую популяцию
        H_new = []
        for x, m in zip(H, mask):
            if m == 1:
                H_new.append(x)
            if len(H_new) == self.population_size:  # Если выбрано слишком много особей
                break
        return H_new


def gray2int(gray_code):
    """Получить целое число из массива с кодом Грея"""
    binary = gray2bin(gray_code)
    base = 1
    result = 0
    binary.reverse()
    for bit in binary:
        result += bit * base
        base *= 2
    return result


def gray2bin(gray_code):
    """Перевести массив с кодом Грея в массив с двоичной СИ"""
    gray_code = list(map(bool, gray_code))
    binary = gray_code.copy()
    shifted = gray_code
    for _ in range(len(gray_code) - 1):
        shifted = [False] + shifted[:-1]
        binary = [b != bs for b, bs in zip(binary, shifted)]
    binary = list(map(int, binary))
    return binary

def func(x):
    """Целевая функция - функция приспособленности"""
    a1 = 0.15
    a2 = 0.2
    a3 = 0.15
    a4 = 0.15
    a5 = -2
    a6 = 10
    x1, x2 = dec_func(x)
    return -(a1 * x1*x1 + a2 * x1*x2 + a3 * x2*x2 + a4 * x1 + a5 * x2 + a6)


def dec_func(x):
    """Функция декодирования вектора генов"""
    precision = 10
    var_len = int(len(x)/2)
    bias = 2**(var_len-1)/precision
    x1 = gray2int(x[:var_len])/precision - bias
    x2 = gray2int(x[var_len:])/precision - bias
    return [x1, x2]

# Запуск генетической оптимизации
population = Population(fit_func=func, decode_func=dec_func, population_size=10, gen_length=30, k_pareto=4,
                        p_mut=0.8, a_select=0.95)
population.evolute(n_steps=250)
