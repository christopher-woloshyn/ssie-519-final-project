import random
from cwnetworks import perceptron
from cwnetworks import multilayerperceptron as mlp

GLOBAL_EPOCH = 500


class Chromosome():
    """
    A set of instructions for creating an instance of a neural network.

    Args:
        data [np array]: 2d array of input values for the neural network.
        expected [np array]: 2d array of expected output values for the neural network.
        dna='' [str]:

    Attributes:
        data [np array]:
        expected [np array]:
        dna [str]:
        genes [list]:
        network_type [int]:
        learn_rate [float]:
        hidden_nodes [int]:
        network [MultiLayerPerceptron/Perceptron]:
        fitness [float]:

    TODO:
        [] Complete Docstrings
    """

    def __init__(self, data, expected, dna=''):
        """ YOUR TEXT HERE"""
        self.data = data
        self.expected = expected

        if dna == '':
            self.dna = self.random_dna()
        else:
            self.dna = dna

        self.genes = self.get_genes()

    def random_dna(self):
        """ YOUR TEXT HERE"""
        s = ''
        for i in range(9):
            s += random.choice(['1', '0'])

        return s

    def get_genes(self):
        """ YOUR TEXT HERE"""
        splits = [0, 1, 5, len(self.dna)]
        N = len(splits)

        genes = []
        for i in range(N - 1):
            s = splits[i]
            t = splits[i+1]
            genes.append(self.dna[s:t])

        return genes

    def preprocess(self):
        """ YOUR TEXT HERE"""
        self.network_type = int(self.genes[0])
        self.learn_rate = (int(self.genes[1], 2) + 1) / 40
        self.hidden_nodes = int(self.genes[2], 2) + 1

    def init_network(self):
        """ YOUR TEXT HERE"""
        self.preprocess()
        if self.network_type:
            self.network = mlp.MultiLayerPerceptron(\
                        self.data, self.expected, self.learn_rate,\
                        self.hidden_nodes, epochs=GLOBAL_EPOCH, seed=True)
        else:
            self.network = perceptron.Perceptron(\
                        self.data, self.expected, self.learn_rate,\
                        epochs=GLOBAL_EPOCH, seed=True)

    def print_gene_info(self):
        """ YOUR TEXT HERE"""
        pass


    def get_fitness(self):
        """ YOUR TEXT HERE"""
        self.network.train()
        self.fitness = 1 - self.network.avg_valid_error_history[-1]

    def get_it_on(self, mate):
        """ YOUR TEXT HERE"""
        splits = [1, 5] # denotes splits of genes in each dna string
        s = random.choice(splits)
        zygote1 = self.mutate(self.dna[:s] + mate.dna[s:])
        zygote2 = self.mutate(mate.dna[:s] + self.dna[s:])

        return zygote1, zygote2

    def mutate(self, string):
        """ YOUR TEXT HERE"""
        out = ''
        for i in range(len(string)):
            if random.random() < .05:
                out += str(1 - int(string[i]))
            else:
                out += string[i]

        return out
