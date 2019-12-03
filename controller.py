import generation
import numpy as np

GENERATION_SIZE = 30
NUMBER_OF_GENS = 100


class Controller():
    """
    The Class for controlling all object instantiation, algorithm evolution, and network training.

    Attributes:
        data [np array]:
        exp [np array]:
        size [int]:
        gens [int]:
        pop [Generation]:

    TODO:
        [] Complete Docstrings
    """

    def __init__(self):
        """ Preprocesses the data and expected values for the networks and runs the main evolution function."""
        self.data = self.get_data()
        self.exp = self.get_exp()
        self.size = GENERATION_SIZE # Population size for each generation.
        self.gens = NUMBER_OF_GENS # Total number of generations.
        self.main()

    def main(self):
        """ Instantiates generation 0, evolves the population, plots fitness values."""
        self.pop = generation.Generation(self.data, self.exp, self.size)
        self.evolve()
        self.print_best_net()
        self.plot_results()

    def evolve(self):
        """ Evolves the network through a number of generations depending on NUMBER_OF_GENS."""
        for i in range(self.gens):
            self.pop.calc_fitness()
            self.pop.next_generation()
        self.pop.calc_fitness()
        self.pop.print_gen_info()

    def print_best_net(self):
        """ Prints the hyperparameters of the best network of the final generation."""
        print("#############################################")
        print("# Best Network Configuration DNA: " + self.pop.generation[0].dna + " #")
        print("#############################################")
        print("---------------------------------------------------")

        if self.pop.generation[0].network_type:
            print("Network Type: Multi-Layer Perceptron")
            print("Learning Rate: " + str(self.pop.generation[0].learn_rate))
            print("Number of Hidden Layer Nodes: "\
            + str(self.pop.generation[0].hidden_nodes))

        else:
            print("Network Type: Multi-Layer Perceptron")
            print("Learning Rate: " + str(self.pop.generation[0].learn_rate))
            print("Number of Hidden Layer Nodes: "\
            + str(self.pop.generation[0].hidden_nodes))
        print("---------------------------------------------------")

    def plot_results(self):
        """ YOUR TEXT HERE"""
        self.pop.plot_total_fitness()
        self.pop.plot_fitness()
        self.pop.plot_scaled_fitness()
        self.pop.generation[0].network.plot_error()
        #self.pop.generation[0].network.plot_avg_error()

    def get_data(self):
        """ Creates the input data array (based on XOR gate)."""
        data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        ])
        return data

    def get_exp(self):
        """ Creates the expected output data (based on XOR gate)."""
        exp = np.array([
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [0],
        ])
        return exp
