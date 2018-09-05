import pandas as pd
import numpy as np
import collections
import itertools
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# =================
# Utility functions
# =================

def readin(filepath):
    df = pd.read_csv(filepath, header=None)
    return df.values

def convert_to_onehot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, z * alpha)

def relu_deriv(y):
    return (y > 0).astype(int)*1.0

def leaky_relu_deriv(y, alpha=0.01):
    dy = np.ones_like(y)
    dy[y < 0] = alpha
    return dy

def softmax(z):
    """
    Calculate the softmax over n input samples

    @param x: of shape(n, m). n is the number of data samples, m is the dimension
    """
    z_exp = np.exp(z)
    z_sum = np.sum(z_exp, axis=1, keepdims=True)
    s = z_exp / z_sum
    return s

def softmax_stable(z):
    """
    Compute the softmax of vector x in a numerically stable way.
    """
    shiftz = z - np.max(z)
    z_exp = np.exp(shiftz)
    z_sum = np.sum(z_exp, axis=1, keepdims=True)
    s = z_exp / z_sum
    return s

# ====================
# Define Layer Objects
# ====================

class Layer(object):
    """Base class for the different layers."""

    def get_params_iter(self):
        """
        Return an iterator over the parameters (if any).
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place.
        """
        return []

    def get_params_grad(self, X, output_grad):
        """"
        Return a list of gradients over the parameters.
        The list has the same order as the get_params_iter iterator.

        :param X: the input
        :param output_grad: the gradient at the output of this layer
        :return: a list of gradients over the parameters
        """
        return []

    def get_output(self, X):
        """ Perform the forward step linear transformation."""
        pass

    def get_input_grad(self, Y, output_grad=None, T=None):
        """
        Return the gradient at the inputs of this layer.

        :param Y: the pre-computed output of this layer (not needed in this case).
        :param output_grad: the gradient at the output of this layer
         (gradient at input of next layer).
        :param T: will be assigned if used for Output layer;
        the gradient will be based on the output error instead of output_grad
        """
        pass


class LinearLayer(Layer):
    """Linear layer performs a linear transformation to its input."""

    def __init__(self, n_in, n_out):
        """
        Initialize hidden layer parameters.

        :param n_in: the number of input variables.
        :param n_out: the number of output variables.
        """
        self.W = np.random.randn(n_in, n_out)/np.sqrt(n_out)
        self.b = np.random.randn(n_out)
        self.n_in = n_in
        self.n_out = n_out

    def __str__(self):
        return "Linear Layer: " + str(self.n_in) + ", " + str(self.n_out)

    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                               np.nditer(self.b, op_flags=['readwrite']))

    def get_output(self, X):
        """Perform the forward step linear transformation."""
        return X.dot(self.W) + self.b

    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters."""
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]

    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return output_grad.dot(self.W.T)


class ReluLayer(Layer):
    """Relu layer performs relu activation to its input."""
    def __str__(self):
        return "Relu Layer"

    def get_output(self, X):
        return relu(X)

    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(relu_deriv(Y), output_grad)


class SoftmaxOutputLayer(Layer):
    """Softmax output layer computes the classification propabilities at the output."""

    def __str__(self):
        return "Softmax Layer"

    def get_output(self, X):
        """Perform the forward step transformation."""
        return softmax_stable(X)

    def get_input_grad(self, Y, T):
        """Return the gradient at the inputs of this layer."""
        return (Y - T) / Y.shape[0]

    def get_cost(self, Y, T):
        """Return the cost at the output of this output layer."""
        return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]


# ===================
# Iteration functions
# ===================
def forward_step(input_samples, layers):
    """
    Compute the forward activations of each layer.

    :param input_samples: a matrix of input samples (each row is an input vector)
    :param layers: a list of layers
    :return: a list of activations. activations[0] contains the input
    """
    activations = [input_samples]

    X_in = input_samples.copy()
    for layer in layers:
        Y = layer.get_output(X_in)
        activations.append(Y)
        X_in = activations[-1]
    return activations


def backward_step(activations, targets, layers):
    """
    Perform the backpropagation step over all the layers and return the parameter gradients.

    :param activations: A list of forward step activations where the activation at
        each index i+1 corresponds to the activation of layer i in layers.
        activations[0] contains the input samples.
    :param targets: The output targets of the output layer.
    :param layers: A list of Layers corresponding that generated the outputs in activations.
    :return: A list of parameter gradients where the gradients at each index corresponds to
        the parameters gradients of the layer at the same index in layers.
    """
    param_grads = collections.deque()  # List of parameter gradients for each layer
    output_grad = None  # The error gradient at the output of the current layer

    # Propagate the error backwards through all the layers.
    # Use reversed to iterate backwards over the list of layers.
    for layer in reversed(layers):
        Y = activations.pop()

        # The output layer error is calculated different then hidden layer error.
        if output_grad is None:
            input_grad = layer.get_input_grad(Y, targets)
        else:
            input_grad = layer.get_input_grad(Y, output_grad)

        # Get the input of this layer (activations of the previous layer)
        X = activations[-1]

        # Compute the layer parameter gradients used to update the parameters
        grads = layer.get_params_grad(X, output_grad)
        param_grads.appendleft(grads)

        # Compute gradient at output of previous layer (input of current layer):
        output_grad = input_grad
    return list(param_grads)


def update_params(layers, param_grads, learning_rate):
    """
    Function to update the parameters of the given layers with the given gradients
    by gradient descent with the given learning rate.
    """
    for layer, layer_backprop_grads in zip(layers, param_grads):
        for param, grad in zip(layer.get_params_iter(), layer_backprop_grads):
            # The parameter returned by the iterator point to the memory space of
            #  the original layer and can thus be modified inplace.
            param -= learning_rate * grad  # Update each parameter


def gradient_check(X_train, T_train, layers):
    nb_samples_gradientcheck = 10 # Test the gradients on a subset of the data
    X_temp = X_train[0:nb_samples_gradientcheck,:]
    T_temp = T_train[0:nb_samples_gradientcheck,:]

    # Get the parameter gradients with backpropagation
    activations = forward_step(X_temp, layers)
    param_grads = backward_step(activations, T_temp, layers)

    # Set the small change to compute the numerical gradient
    eps = 0.0001
    # Compute the numerical gradients of the parameters in all layers.
    for idx in range(len(layers)):
        layer = layers[idx]
        layer_backprop_grads = param_grads[idx]
        # Compute the numerical gradient for each parameter in the layer
        for p_idx, param in enumerate(layer.get_params_iter()):
            grad_backprop = layer_backprop_grads[p_idx]
            # + eps
            param += eps
            plus_cost = layers[-1].get_cost(forward_step(X_temp, layers)[-1], T_temp)
            # - eps
            param -= 2 * eps
            min_cost = layers[-1].get_cost(forward_step(X_temp, layers)[-1], T_temp)
            # reset param value
            param += eps
            # calculate numerical gradient
            grad_num = (plus_cost - min_cost)/(2*eps)
            # Raise error if the numerical grade is not close to the backprop gradient
            if not np.isclose(grad_num, grad_backprop):
                raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_backprop)))
    print('No gradient errors found')


# ==================
# Plotting functions
# ==================
def plot_costs(minibatch_costs, training_costs, validation_costs, nb_of_iterations, nb_of_batches, filename):
    minibatch_x_inds = np.linspace(0, nb_of_iterations, num=nb_of_iterations * nb_of_batches)
    iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations)
    plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')
    plt.plot(iteration_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
    plt.plot(iteration_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set')
    plt.xlabel('iteration')
    plt.ylabel('$\\xi$', fontsize=15)
    plt.title('Decrease of cost over backprop iteration')
    plt.legend()
    plt.axis((0, nb_of_iterations, 0, 2.5))
    plt.grid()
    plt.savefig(filename)
    plt.close()


def plot_accuracys(train_accuracys, validation_accuracys, nb_of_iterations, filename):
    iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations)
    plt.plot(iteration_x_inds, train_accuracys, 'r-', linewidth=2, label='acc. full training set')
    plt.plot(iteration_x_inds, validation_accuracys, 'b-', linewidth=3, label='acc. validation set')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.title('Increase of accuracy over backprop iteration')
    plt.legend(loc=4)
    plt.axis((0, nb_of_iterations, 0, 1.0))
    plt.grid()
    plt.savefig(filename)
    plt.close()


# =================
# Process functions
# =================
def prepare_data(x_train_path, y_train_path, x_test_path, y_test_path):
    # Prepare data for NN training
    X_train_all = readin(x_train_path)
    Y_train_all = readin(y_train_path)
    T_train_all = convert_to_onehot(Y_train_all, 4).T
    X_test = readin(x_test_path)
    Y_test = readin(y_test_path)
    T_test = convert_to_onehot(Y_test, 4).T
    X_train, X_validation, T_train, T_validation = \
        train_test_split(X_train_all, T_train_all, test_size=0.3, random_state=42)
    return (X_train, X_validation,  X_test, T_train, T_validation, T_test)


def build_layers(layer_dims):
    """
    Build NN layers based on the layer_dims given.
    The network built by this function is fixed to have Relu activations in the middle,
    and a softmax output layer.

    :param layer_dims: a list of integers for the number of neurons in each layer.
        The 1st element in the list is the dimension of input layer
    :return: a list of Layer() objects
    """
    out_dim = layer_dims.pop()
    layers = []
    for i in range(len(layer_dims)-1):
        layers.append(LinearLayer(layer_dims[i], layer_dims[i+1]))
        layers.append(ReluLayer())
    layers.append(LinearLayer(layer_dims[-1], out_dim))
    layers.append(SoftmaxOutputLayer())

    return layers


def get_cost_accuracy(X, T, layers):
    y_true = np.argmax(T, axis=1)
    activations = forward_step(X, layers)
    cost = layers[-1].get_cost(activations[-1], T)
    y_pred = np.argmax(activations[-1], axis=1)
    accuracy = metrics.accuracy_score(y_true, y_pred)

    return (cost, accuracy)


def train_minibatch_SGD(X_train, T_train, X_validation, T_validation, layers,
                        batch_size = 32, max_num_iterations = 150, learning_rate=0.001):
    num_batchs = X_train.shape[0]  // batch_size + 1
    XT_batches = list(zip(np.array_split(X_train, num_batchs, axis=0), np.array_split(T_train, num_batchs, axis=0)))

    minibatch_costs = []
    train_costs = []
    validation_costs = []
    train_accuracies = []
    validation_accuracies = []

    for iteration in range(max_num_iterations):
        for (X, T) in XT_batches:
            activations = forward_step(X, layers)
            minibatch_cost = layers[-1].get_cost(activations[-1], T)
            minibatch_costs.append(minibatch_cost)
            param_grads = backward_step(activations, T, layers)
            update_params(layers, param_grads, learning_rate)

        train_cost, train_accuracy = get_cost_accuracy(X_train, T_train, layers)
        train_costs.append(train_cost)
        train_accuracies.append(train_accuracy)

        validation_cost, validation_accuracy = get_cost_accuracy(X_validation, T_validation, layers)
        validation_costs.append(validation_cost)
        validation_accuracies.append(validation_accuracy)

        print('iter {}: train loss {:.4f} acc {:.4f}, val loss {:.4f} acc {:.4f}'
              .format(iteration + 1, train_cost, train_accuracy, validation_cost, validation_accuracy))

        if len(validation_costs) > 5:
            # Stop training if the cost on the validation set doesn't decrease
            # for 5 iterations
            if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3] >= \
                    validation_costs[-4] >= validation_costs[-5]:
                break
    return (num_batchs, minibatch_costs, train_costs, validation_costs, train_accuracies, validation_accuracies)