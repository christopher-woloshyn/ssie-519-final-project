import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    """ The sigmoid activation function."""
    return 1. / (1. + np.exp(-x))


class Perceptron():
    """
    A classic Multi-Layer Perceptron neural network with 1 hidden layer.

    Args:
        self [MultiLayerPerceptron]:
        data [np array]: 2d array of input values.
        expected [np array]: 2d array of expected output values.
        rate [float]: Learning rate for gradient descent.
        split=0.7 [float]: % of input data that is for training between 0 and 1.
        batch=1. [float]: Splits the training data into batches for stochastic gradient decsent.
        epochs=0 [int]: Specifies the total number of epochs. If its 0 then
        seed=False [bool]: Instanziates the network with set seed making all random numbers consistent across instances.
        DEBUG=False [bool]: Enables debug features throughout the code.

    Attributes:
        DEBUG [bool]: An indicator for debug mode by storing DEBUG.
        rate [float]: Where the learning rate is stored.
        train_data [np array]: A portion of the total data set partitioned for training by the split value.
        valid_data [np array]: A portion of the total data set partitioned for validation by the split value.
        train_expected [np array]: The corresponding expected values for the training data.
        valid_expected [np array]: The corresponding expected values for the validation data.
        batch_size [int]: (NOT YET IMPLEMENTED) An integer value specified by a fraction of the training data. Used for stochastic gradient decent.
        in_dim [int]: The length of the input vector.
        out_dim [int]: The length of the output vector.
        w [np array]: Weights matrix for the network.
        b [np array]: Bias vector for network.
        nabla_w1 [np array]: Partial derivatives accumulation for w.
        nabla_b1 [np array]: Partial derivatives accumulation for b.
        train_error [np array]: The mean squarred error vector from training example.
        train_error_history [list]: Accumulation of the training errors (on the last data point in the epoch) over the total number of epochs.
        valid_error [np array]: The mean squarred error vector from validation example.
        valid_error_history [list]: Accumulation of the validation errors (on the last data point in the epoch) over the total number of epochs.
        avg_train_error [float]: Average training error accross each node in a certain data point.
        avg_train_error_history [list]: Accumulation of the average errors (on the last data point in the epoch) over the total number of epochs.
        avg_valid_error [float]: Average validation error accross each node in a certain data point.
        avg_valid_error_history [list]: Accumulation of the average errors (on the last data point in the epoch) over the total number of epochs.
        epochs [int]: The total number of epochs.
        a0 [np array]: The activation of the input layer (equivalent to an input data point).
        s1 [np array]: The linear basis of the output layer.
        a1 [np array]: The activation of the output layer.
        error [np array]: The raw error vector of a specific data point.

    TODO:
        [] Generalize activation functions!
        [] Implement functionality of batch updating/stochastic gradient decent.
        [] Create separate activation functions folder? (import activations)
        [] Generalize training so it stops automatically to prevent overfitting.
    """

    def __init__(self, data, expected, rate, split=0.7, batch=1., epochs=0, seed=False, DEBUG=False):
        """ Instantiates a Single Layer Perceptron neural network (SLP)."""
        self.DEBUG = DEBUG
        self.seed = seed

        if self.seed:
            np.random.seed(1)

        self.rate = rate
        self.split_data(data, expected, split, batch)
        self.set_layer_dims()
        self.init_weights()
        self.init_error(epochs)

    def split_data(self, data, expected, split, batch):
        """ Splits data into training data and validation data.

        Args:
            data [np array]:
            expected [np array]:
            split [float]: % of input data that is for training between 0 and 1.
            batch [float]: Splits the training data into batches for stochastic gradient descent.

        """
        s = round(len(data) * split)
        N = len(data)

        self.train_data = data[0:s]
        self.valid_data = data[s:N]
        self.train_expected = expected[0:s]
        self.valid_expected = expected[s:N]

        self.batch_size = round(len(self.train_data) * batch)

    def set_layer_dims(self):
        """ Sets dimension of each of the network's layers."""
        self.in_dim = len(self.train_data[0])
        self.out_dim = len(self.train_expected[0])

    def init_weights(self):
        """ Initializes weights and biases between +- 2/3 based on each layer's dimension."""
        self.b =\
        (4 * np.random.rand(self.out_dim) - 2) / 3
        self.w =\
        (4 * np.random.rand(self.out_dim, self.in_dim) - 2) / 3

        self.nabla_w = np.zeros((self.out_dim, self.in_dim))
        self.nabla_b = np.zeros(self.out_dim)

        if self.DEBUG:
            self.print_current_weights()

    def print_current_weights(self):
        """ Prints current weights and bias matrices after each epoch."""
        print("---------------------------------------------------")
        print("Initial Weights:\n", self.w)
        print()
        print("Initial Biases:\n", self.b)
        print("---------------------------------------------------")

    def init_error(self, epochs):
        """ Initializes the error variables for tracking error."""
        self.train_error = 0
        self.train_error_history = []
        self.valid_error = 0
        self.valid_error_history = []

        self.avg_train_error = 0
        self.avg_train_error_history = []
        self.avg_valid_error = 0
        self.avg_valid_error_history = []

        if epochs > 0:
            self.epochs = epochs
        else:
            self.epochs = 0

    def train(self):
        """ Adjusts the weights iteratively through gradient descent.

        Args:
            self [Perceptron]:
        """
        epoch_count = 1
        while self.epochs:
            if len(self.valid_data) > 0:
                self.calc_valid_error()

            for i in range(self.batch_size):
                self.forward_prop(i)
                self.calc_train_error(i)
                self.back_prop()

            if self.DEBUG:
                self.print_progress(epoch_count)

            self.update()
            epoch_count += 1

        if self.DEBUG:
            self.print_results()

    def calc_valid_error(self):
        """ Calculates error on validation data."""
        for i in range(len(self.valid_data)):
            a0 = self.valid_data[i]
            s1 = np.dot(self.w, a0) + self.b
            a1 = sigmoid(s1)
            e = a1 - self.valid_expected[i]
            self.valid_error += np.dot(e, e) / 2

            self.avg_valid_error += sum(abs(e)) / len(e)

        self.avg_valid_error /= len(self.valid_data)

    def forward_prop(self, i):
        """ Applies linear basis and an activation function to feed forward."""
        self.a0 = self.train_data[i]
        self.s1 = np.dot(self.w, self.a0) + self.b
        self.a1 = sigmoid(self.s1)

    def calc_train_error(self, i):
        """ Calculates error on training data."""
        self.error = self.a1 - self.train_expected[i]
        self.train_error += np.dot(self.error, self.error) / 2

    def back_prop(self):
        """ Caclulates partial derivatives using the error on training data."""
        partial = self.error * self.a1 * (1 - self.a1) # sigmoid derivative
        self.nabla_w += np.outer(partial, self.a0)
        self.nabla_b += partial

    def update(self):
        """ Calculates the validation error, batch updates the weights."""
        self.update_weights()
        self.reset_nablas()
        self.update_and_reset_errors()
        self.epochs -= 1

    def update_weights(self):
        """ Adjusts the weights based on learning factor and partial derivatives."""
        self.w -= self.rate * self.nabla_w
        self.b -= self.rate * self.nabla_b

    def reset_nablas(self):
        """ Resets partial derivative matrices to 0 for next epoch."""
        self.nabla_w = np.zeros((self.out_dim, self.in_dim))
        self.nabla_b = np.zeros(self.out_dim)

    def update_and_reset_errors(self):
        """ Appendes error histories and resets current errors to 0 for next epoch."""
        self.train_error_history.append(self.train_error)
        self.train_error = 0
        self.valid_error_history.append(self.valid_error)
        self.valid_error = 0
        self.avg_valid_error_history.append(self.avg_valid_error)
        self.avg_valid_error = 0

    def print_progress(self, epoch):
        """ Prints the new error values after each epoch. Only on in debug mode."""
        print("epoch count: " + str(epoch))
        print("Error on Training Data: " + str(self.train_error))
        print("Error on Validation Data: " + str(self.valid_error))
        print("Average Error on Validation Data: " + str(self.avg_valid_error))
        print()

    def print_results(self):
        """ Prints final weights and bias matrix, and final average validation error."""
        print("---------------------------------------------------")
        print("Final Weights Matrix:\n" + str(self.w) + "\n")
        print("Final Bias Vector:\n" + str(self.b) + "\n")
        print("Final Average Error on Validation Data: " +\
                str(self.avg_valid_error_history[-1]))
        print("---------------------------------------------------")
        print()

    def plot_error(self):
        """ Plots the training and validation error with respect to the number of epochs."""
        plt.style.use('seaborn')
        plt.plot(self.train_error_history, color="red")
        plt.plot(self.valid_error_history, color="blue")
        plt.legend(["Training Error", "Validation Error"])
        plt.xlabel("No. of Epochs")
        plt.ylabel("Error")
        plt.show()

    def plot_avg_error(self):
        """ Plots the average validation error with respect to the number of epochs."""
        plt.style.use('seaborn')
        plt.plot(self.avg_valid_error_history, color='green')
        plt.legend(["Average Error"])
        plt.xlabel("No. of Epochs")
        plt.ylabel("Average Error")
        plt.show()
