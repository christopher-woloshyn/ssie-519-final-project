import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    """ The sigmoid activation function."""
    return 1. / (1. + np.exp(-x))


class MultiLayerPerceptron():
    """
    A classic Multi-Layer Perceptron neural network with 1 hidden layer.

    Args:
        self [MultiLayerPerceptron]:
        data [np array]: 2d array of input values.
        expected [np array]: 2d array of expected output values.
        hidden [int]: Number of hidden nodes in the hidden layer.
        rate [float]: Learning rate for gradient descent.
        split=0.7 [float]: % of input data that is for training between 0 and 1.
        batch=1. [float]: Splits the training data into batches for stochastic gradient descent.
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
        hid_dim [int]: The number of hidden layer nodes in the network.
        out_dim [int]: The length of the output vector.
        w1 [np array]: Weights matrix for the input layer to the hidden layer.
        b1 [np array]: Bias vector for the input layer to the hidden layer.
        w2 [np array]: Weights matrix for the hidden layer to the output layer.
        b2 [np array]: Bias vector for the hidden layer to the output layer.
        nabla_w1 [np array]: Partial derivatives accumulation for w1.
        nabla_b1 [np array]: Partial derivatives accumulation for b1.
        nabla_w2 [np array]: Partial derivatives accumulation for w2.
        nabla_b2 [np array]: Partial derivatives accumulation for b2.
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
        s1 [np array]: The linear basis of the hidden layer.
        a1 [np array]: The activation of the hidden layer.
        s2 [np array]: The linear basis of the output layer.
        a2 [np array]: The activation of the output layer.
        error [np array]: The raw error vector of a specific data point.

    TODO:
        [] Generalize activation functions!
        [] Implement functionality of batch updating/stochastic gradient decent.
        [] Create separate activation functions folder? (import activations)
        [] Generalize training so it stops automatically to prevent overfitting.
    """

    def __init__(self, data, expected, rate, hidden, split=0.7, batch=1., epochs=0, seed=False, DEBUG=False):
        """Instantiates a Multi-Layer Perceptron neural network (MLP)."""
        self.DEBUG = DEBUG

        if seed:
            np.random.seed(1)

        self.rate = rate
        self.split_data(data, expected, split, batch)
        self.set_layer_dims(hidden)
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

    def set_layer_dims(self, hidden):
        """ Sets dimension of each of the network's layers."""
        self.in_dim = len(self.train_data[0])
        self.hid_dim = hidden
        self.out_dim = len(self.train_expected[0])

    def init_weights(self):
        """ Initializes weights and biases between +- 2/3 based on each layer's dimension."""
        self.w1 =\
        (4 * np.random.rand(self.hid_dim, self.in_dim) - 2) / 3
        self.b1 =\
        (4 * np.random.rand(self.hid_dim) - 2) / 3
        self.w2 =\
        (4 * np.random.rand(self.out_dim, self.hid_dim) - 2) / 3
        self.b2 =\
        (4 * np.random.rand(self.out_dim) - 2) / 3

        self.nabla_w1 = np.zeros((self.hid_dim, self.in_dim))
        self.nabla_b1 = np.zeros(self.hid_dim)
        self.nabla_w2 = np.zeros((self.out_dim, self.hid_dim))
        self.nabla_b2 = np.zeros(self.out_dim)

        if self.DEBUG:
            self.print_current_weights()

    def print_current_weights(self):
        """ Prints current weights and bias matrices after each epoch."""
        print("---------------------------------------------------")
        print("Initial Weights, Input to Hidden Layer:\n", self.w1)
        print()
        print("Initial Biases, Input to Hidden Layer:\n", self.b1)
        print()
        print("Initial Weights, Hidden Layer to Output:\n", self.w2)
        print()
        print("Initial Biases, Hidden Layer to Outpur:\n", self.b2)
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

            s1 = np.dot(self.w1, a0) + self.b1
            a1 = sigmoid(s1)

            s2 = np.dot(self.w2, a1) + self.b2
            a2 = sigmoid(s2)

            e = a2 - self.valid_expected[i]
            self.valid_error += np.dot(e, e) / 2

            self.avg_valid_error += sum(abs(e)) / len(e)

        self.avg_valid_error /= len(self.valid_data)

    def forward_prop(self, i):
        """ Applies linear basis and an activation function to feed forward."""
        self.a0 = self.train_data[i]

        self.s1 = np.dot(self.w1, self.a0) + self.b1
        self.a1 = sigmoid(self.s1)

        self.s2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = sigmoid(self.s2)

    def calc_train_error(self, i):
        """ Calculates error on training data."""
        self.error = self.a2 - self.train_expected[i]
        self.train_error += np.dot(self.error, self.error) / 2

    def back_prop(self):
        """ Caclulates partial derivatives using the error on training data."""
        partial2 = self.error * self.a2 * (1 - self.a2) # sigmoid derivative
        self.nabla_w2 += np.outer(partial2, self.a1)
        self.nabla_b2 += partial2

        temp_error = partial2.dot(self.w2)
        partial1 = temp_error * self.a1 * (1 - self.a1)
        self.nabla_w1 += np.outer(partial1, self.a0)
        self.nabla_b1 += partial1

    def update(self):
        """ Calculates the validation error, batch updates the weights."""
        self.update_weights()
        self.reset_nablas()
        self.update_and_reset_errors()
        self.epochs -= 1

    def update_weights(self):
        """ Adjusts the weights based on learning factor and partial derivatives."""
        self.w1 -= self.rate * self.nabla_w1
        self.b1 -= self.rate * self.nabla_b1
        self.w2 -= self.rate * self.nabla_w2
        self.b2 -= self.rate * self.nabla_b2

    def reset_nablas(self):
        """ Resets partial derivative matrices to 0 for next epoch."""
        self.nabla_w1 = np.zeros((self.hid_dim, self.in_dim))
        self.nabla_b1 = np.zeros(self.hid_dim)
        self.nabla_w2 = np.zeros((self.out_dim, self.hid_dim))
        self.nabla_b2 = np.zeros(self.out_dim)

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
        """ Prints final weights and bias matrices, and final average validation error."""
        print("---------------------------------------------------")
        print("Final Weights Matrix, Input to Hidden Layer:\n" +\
                str(self.w1) + "\n")
        print("Final Bias Vector, Input to Hidden Layer:\n" +\
                str(self.b1) + "\n")
        print("Final Weights Matrix, Hidden Layer to Ouput:\n" +\
                str(self.w2) + "\n")
        print("Final Bias Vector, Input to Hidden Layer:\n" +\
                str(self.b2) + "\n")
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
