from q2_ref import *

if __name__ == "__main__":
    x_train_path = "./data/x_train.csv"
    y_train_path = "./data/y_train.csv"
    x_test_path = "./data/x_test.csv"
    y_test_path = "./data/y_test.csv"

    (X_train, X_validation, X_test, T_train, T_validation, T_test) = \
        prepare_data(x_train_path, y_train_path, x_test_path, y_test_path)
    print(X_train.shape, X_validation.shape, X_test.shape)
    print(T_train.shape, T_validation.shape, T_test.shape)

    layer_dims = [X_train.shape[1]]
    layer_dims.extend([14] * 28)
    layer_dims.extend([T_train.shape[1]])
    layers = build_layers(layer_dims)

    (num_batches, minibatch_costs, train_costs, validation_costs, train_accuracies, validation_accuracies) = \
        train_minibatch_SGD(X_train, T_train, X_validation, T_validation, layers, learning_rate=0.001)
    num_iterations = len(train_costs)

    test_cost, test_accuracy = get_cost_accuracy(X_test, T_test, layers)
    print('The final accuracy on the test set is {:.4f}'.format(test_accuracy))

    plot_costs(minibatch_costs, train_costs, validation_costs, num_iterations, num_batches, "net3_cost.png")
    plot_accuracys(train_accuracies, validation_accuracies, num_iterations, "net3_acc.png")