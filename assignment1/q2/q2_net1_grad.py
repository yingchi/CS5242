from q2_ref import *

if __name__ == "__main__":
    w_path = "./data/w-100-40-4.csv"
    b_path = "./data/b-100-40-4.csv"

    X_train = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]]).astype(np.float128)
    T_train = np.array([[0, 0, 0, 1]])
    layer_dims = [X_train.shape[1]]
    layer_dims.extend([100, 40])
    layer_dims.extend([T_train.shape[1]])
    layers = build_layers(layer_dims)

    weights = pd.read_csv(w_path, header=None).drop(axis=1).values