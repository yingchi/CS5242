from q2_ref import *
import csv

if __name__ == "__main__":
    w_path = "./data/c/w-14-28-4.csv"
    b_path = "./data/c/b-14-28-4.csv"

    X_train = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]]).astype(np.float128)
    T_train = np.array([[0, 0, 0, 1]])
    layer_dims = [X_train.shape[1]]
    layer_dims.extend([14] * 28)
    layer_dims.extend([T_train.shape[1]])
    layers = build_layers(layer_dims)

    weights = pd.read_csv(w_path, header=None).drop([0], axis=1)
    weights = weights.values
    biases = pd.read_csv(b_path, header=None).drop([0], axis=1)
    biases = biases.values

    for (idx, l) in enumerate(layers):
        if idx % 2 == 0:
            num_rows = layer_dims[idx//2]
            weights_block = weights[:num_rows]
            weights_usable = weights_block[:,~np.isnan(weights_block).all(0)]
            # print(weights_usable.shape)
            layers[idx].W = weights_usable
            weights = np.delete(weights, slice(0, num_rows), axis=0)

            biases_block = biases[0]
            biases_usable = biases_block[:, ~np.isnan(biases_block).all(0)]
            biases_usable = biases_usable[~np.isnan(biases_usable).all(1)].flatten()
            # print(biases_usable.shape)
            # print(biases_usable)
            layers[idx].b = biases_usable
            biases = np.delete(biases, 0, axis=0)

    # for l in layers:
    #     print(l)

    activations = forward_step(X_train, layers)
    param_grads = backward_step(activations, T_train, layers)

    with open('./output/dW-14-28-4.csv', 'w') as csvfile:
        myWriter = csv.writer(csvfile)
        for j in range((len(layers) // 2)):
            for i in range(layers[j * 2].W.shape[0]):
                myWriter.writerow(param_grads[j * 2][(i * layers[j * 2].W.shape[1]):((i + 1) * layers[j * 2].W.shape[1])])

    with open('./output/db-14-28-4.csv', 'w') as csvfile:
        myWriter = csv.writer(csvfile)

        for i in range((len(layers) // 2)):
            start_nb = layers[i * 2].W.shape[0] * layers[i * 2].W.shape[1]
            myWriter.writerow(param_grads[i * 2][start_nb:])
