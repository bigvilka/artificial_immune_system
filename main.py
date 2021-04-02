import csv
from math import log
from random import uniform

import matplotlib.pyplot as plt
import numpy as np

RANGE = 100000
MUTATION_RANGE = 500
CSV_PATH = 'input.csv'
SSE_BREAKPOINT = 50
INPUT_POINTS_COUNT = 100



class Chromosome:
    a = 0
    b = 0
    c = 0

    def __init__(self):
        self.a = uniform(-RANGE, RANGE)
        self.b = uniform(-RANGE, RANGE)
        self.c = uniform(-RANGE, RANGE)

    def copy(self):
        copy = Chromosome()
        copy.a = self.a
        copy.b = self.b
        copy.c = self.c
        return copy

    def fitness_function(self, x):
        return self.a + (self.b * x) + (self.c * x**2)

    def mutation(self, attr):
        self.__setattr__(
            attr,
            self.__getattribute__(attr) + uniform(-MUTATION_RANGE, MUTATION_RANGE)
        )


class Creator:

    def __init__(self, generations=100):
        self.generations = generations
        self.input_points = self._read_points()

    def _read_points(self):
        with open(CSV_PATH, 'r') as file_:
            return [row for row in csv.reader(file_)]

    def init_population(self, size=81):
        self.population_size = size
        self.population = [Chromosome() for _ in range(size)]

    def selection(self):
        chromosome_power = {}
        for chromosome in self.population:
            sse = 0
            for row in self.input_points:
                y = chromosome.fitness_function(float(row[0]))
                sse = sse + (y - float(row[1]))**2
            chromosome_power[chromosome] = sse

            if sse < SSE_BREAKPOINT:
                self.population = [chromosome]
                self.population_size = 1
                print('BREAKPOINT')
                return None

        chromosome_power_sorted = dict(sorted(chromosome_power.items(), key=lambda item: item[1]))
        selected = list(chromosome_power_sorted.keys())[:(self.population_size // 3)]
        return selected

    def init_mutation(self, selected):
        self.population = []
        for chromosome in selected:
            chromosome_a = chromosome.copy()
            chromosome_a.mutation('a')
            chromosome_b = chromosome.copy()
            chromosome_b.mutation('b')
            chromosome_c = chromosome.copy()
            chromosome_c.mutation('c')
            self.population.extend([chromosome_a, chromosome_b, chromosome_c])

    def evolution(self):
        for _ in range(self.generations):
            selected = self.selection()
            if not selected:
                break
            self.init_mutation(selected)

        for _ in range(int(log(self.population_size, 3))):
            self.population = self.selection()
            self.population_size = len(self.population)


class Sample:
    a = 0
    b = 0
    c = 0
    input_points = []

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

        for i in range(0, INPUT_POINTS_COUNT):
            y = self.real_function(i)
            self.input_points.append((i, y))

        with open(CSV_PATH, 'w', newline='') as file_:
            writer = csv.writer(file_)
            writer.writerows(self.input_points)

    def real_function(self, x):
            return self.a + (self.b * x) + (self.c * x**2)


if __name__ == '__main__':
    sample = Sample(0, -35, 6)
    print('\n Real function:')
    print(f'    y(x) = {sample.a} + {sample.b}*x + {sample.c}*x^2')

    creator = Creator(generations=100)
    creator.init_population(size=81)
    print(f'\n Generations: {creator.generations}, population size: {creator.population_size}')

    creator.evolution()
    best_chromosome = creator.population[0]
    print('\n Best function:')
    print(f'    y(x) = {best_chromosome.a} + {best_chromosome.b}*x + {best_chromosome.c}*x^2')

    plt.title('Полиномиальное уравнение регрессии')
    plt.xlabel('x')
    plt.ylabel('y')
    x = np.linspace(0, INPUT_POINTS_COUNT , num=10)
    y = creator.population[0].fitness_function(x)
    y0 = sample.real_function(x)
    plt.plot(x, y0, label='Исходная функция')
    plt.plot(x, y, label='Найденная функция')
    plt.legend()
    plt.show()
