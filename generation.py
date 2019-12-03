import random
import chromosome
from statistics import mean
from matplotlib import pyplot as plt


class Generation():
    """
    Args:
        data [np array]: 2d array of input values for the neural network.
        expected [np array]: 2d array of expected output values for the neural network.
        size [int]: Value for specifying the population size of each generation.

    Attributes:
        ID [int]:
        data [np array]: Where data is stored.
        expected [np array]: Where expected is stored.
        size [int]: Where size is stored.
        total_history [list]:
        best_history [list]:
        worst_history [list]:
        avg_history [list]:
        best_scaled_history [list]:
        worst_scaled_history [list]:
        generation [list]:
        fitness [list]:
        total [float]:
        best [float]:
        worst [float]:
        avg [float]:
        scaled_fitness: [list]:
        best_scaled [float]:
        worst_scaled [float]:
        new_generation [list]:
        roulette [list]:

    TODO:
        [x] Implement fitness scaling algorithm.
        [] Complete Docstrings
    """

    def __init__(self, data, expected, size):
        """ Instantiates generation 0 of the genetic algorithm."""
        self.ID = 0
        self.data = data
        self.expected = expected
        self.size = size

        self.init_fitness_history()
        self.init_generation()

    def init_fitness_history(self):
        """ YOUR TEXT HERE"""
        self.total_history = []
        self.best_history = []
        self.worst_history = []
        self.avg_history = []
        self.best_scaled_history = []
        self.worst_scaled_history = []

    def init_generation(self):
        """ YOUR TEXT HERE"""
        self.generation = []
        for i in range(self.size):
            self.generation.append(chromosome.Chromosome(self.data, self.expected))

    def calc_fitness(self):
        """ YOUR TEXT HERE"""
        self.gen_init_text()

        for i in range(self.size):
            chromosome = self.generation[i]
            chromosome.init_network()
            chromosome.get_fitness()
            self.print_train_info(chromosome, i)

        self.organize()

    def gen_init_text(self):
        """ YOUR TEXT HERE"""
        s = "# GENERATION " + str(self.ID) + " INITIALIZED #"
        t = '#'
        for i in range(len(s) - 1):
            t += '#'
        print(t)
        print(s)
        print(t)

    def print_train_info(self, chromosome, i):
        """ YOUR TEXT HERE"""
        print("Network %s has finished training.\r" %str(i+1), end="")

    def organize(self):
        """ YOUR TEXT HERE"""
        self.sort_generation()
        self.get_fitness_extrema()
        self.update_fitness_histories()
        self.print_gen_info()

    def sort_generation(self):
        """ Sorts generation in decending order by total fitness"""
        self.generation.sort(key=lambda x: x.fitness, reverse=True)
        self.fitness = [x.fitness for x in self.generation]

    def get_fitness_extrema(self):
        """ YOUR TEXT HERE"""
        self.total = sum(self.fitness)
        self.best = self.fitness[0]
        self.worst = self.fitness[-1]
        self.avg = mean(self.fitness)
        self.rescale()

    def rescale(self):
        """ YOUR TEXT HERE"""
        c = 1.2
        if (self.avg - self.worst) <= (self.best - self.avg) / (c - 1):
            a = (c - 1) / (self.best - self.avg)
            b = (self.best - (c * self.avg)) / (self.best - self.avg)

        else:
            a = 1 / (self.avg - self.worst)
            b = -self.worst / (self.avg - self.worst)

        self.rescale_fitness(a, b)

    def rescale_fitness(self, a, b):
        """ YOUR TEXT HERE"""
        self.scaled_fitness = [(a*fit + b) for fit in self.fitness]
        self.best_scaled = self.scaled_fitness[0]
        self.worst_scaled = self.scaled_fitness[-1]

    def update_fitness_histories(self):
        """ YOUR TEXT HERE"""
        self.total_history.append(self.total)
        self.best_history.append(self.best)
        self.worst_history.append(self.worst)
        self.avg_history.append(self.avg)
        self.best_scaled_history.append(self.best_scaled)
        self.worst_scaled_history.append(self.worst_scaled)

    def print_gen_info(self):
        """ YOUR TEXT HERE"""
        print("---------------------------------------------------")
        print("Summary:")
        print()
        print("Total Generation Fitness: " + str(self.total))
        print("Best Generation Fitness: " + str(self.best))
        print("Worst Generation Fitness: " + str(self.worst))
        print("Average Generation Fitness: " + str(self.avg))
        print("---------------------------------------------------")
        print()

    def next_generation(self):
        """ YOUR TEXT HERE"""
        self.new_generation = []
        self.elitism()
        self.repopulate()
        self.update_generation()

    def elitism(self):
        """ YOUR TEXT HERE"""
        elite1 = self.generation[0]
        elite2 = self.generation[1]

        zygote1, zygote2 = elite1.get_it_on(elite2)
        child1 = self.birth(zygote1)
        child2 = self.birth(zygote2)

        self.new_generation.append(elite1)
        self.new_generation.append(elite2)
        self.new_generation.append(child1)
        self.new_generation.append(child2)

    def repopulate(self):
        """ YOUR TEXT HERE"""
        self.roulette = self.create_roulette()
        self.sex()

    def create_roulette(self):
        """ YOUR TEXT HERE"""
        norms = [x/self.size for x in self.scaled_fitness]
        return [sum(norms[:i]) for i in range(1, self.size + 1)]

    def sex(self):
        """ YOUR TEXT HERE"""
        n = (self.size - 4) // 2
        for i in range(n):
            parent1 = self.select_parent()
            parent2 = self.select_parent()

            zygote1, zygote2 = parent1.get_it_on(parent2)
            child1 = self.birth(zygote1)
            child2 = self.birth(zygote2)

            self.new_generation.append(child1)
            self.new_generation.append(child2)

    def select_parent(self):
        """ YOUR TEXT HERE"""
        r = random.random()

        for i in range(self.size):
            if r <= self.roulette[i]:
                return self.generation[i]

    def birth(self, zygote):
        """ YOUR TEXT HERE"""
        return chromosome.Chromosome(self.data, self.expected, dna=zygote)

    def update_generation(self):
        """ YOUR TEXT HERE"""
        self.generation = self.new_generation
        self.ID += 1

    def plot_total_fitness(self):
        """ YOUR TEXT HERE"""
        plt.style.use('seaborn')
        plt.plot(self.total_history, color="black")
        plt.xlabel("Number of Generations")
        plt.ylabel("Fitness")
        plt.show()

    def plot_fitness(self):
        """ YOUR TEXT HERE"""
        plt.style.use('seaborn')
        plt.plot(self.best_history, color="red")
        plt.plot(self.avg_history, color="green")
        plt.plot(self.worst_history, color="blue")
        plt.legend(["Best of Generation", "Average Fitness", "Worst of Generation"])
        plt.xlabel("Number of Generations")
        plt.ylabel("Fitness")
        plt.show()

    def plot_scaled_fitness(self):
        """ YOUR TEXT HERE"""
        plt.style.use('seaborn')
        plt.plot(self.best_scaled_history, color="red")
        plt.plot(self.worst_scaled_history, color="blue")
        plt.legend(["Best of Generation (scaled)", "Worst of Generation (scaled)"])
        plt.xlabel("Number of Generations")
        plt.ylabel("Fitness")
        plt.show()
