import pandas as pd
import numpy as np

def readin(filepath):
    df = pd.read_csv(filepath, header=None)
    df.drop([0], axis=1, inplace=True)
    return df

def linear_forward(A, W, b):
    return np.dot(W, A) + b

def model_forward(X, weights, biases, m_layers):
    """
    Forward propagation for a fully-connected model"""
    A = X
    for l in range(m_layers):
        A_prev = A
        A = linear_forward(A_prev, weights[l], biases[l])
    return A

if __name__ == "__main__":
    filepath_weights = "./data/a/a_w.csv"
    filepath_biases = "./data/a/a_b.csv"
    out_weights = "./output/a-w.csv"
    out_biases = "./output/a-b.csv"
    weights = readin(filepath_weights)
    biases = readin(filepath_biases)

    m_layers = len(biases)
    n_nodes = len(biases.columns)
    w = []
    b = []

    for i in range(m_layers):
        w.append(weights[i*n_nodes:(i+1)*n_nodes].T.values)
        b.append(biases[i:i+1].values.squeeze())

    b_new = b[0]
    w_new = w[0]
    for i in range(1, m_layers):
        b_new = np.dot(w[i], b_new).astype(np.float32) + b[i].astype(np.float32)
        w_new = np.dot(w[i], w_new).astype(np.float32)

    pd.DataFrame(w_new.T).to_csv(out_weights, index=False, header=False, float_format='%.16f')
    pd.DataFrame(b_new).transpose().to_csv(out_biases, index=False, header=False, float_format='%.16f')

    # test
    # X_test1 = np.array([0,0,0,0,1])
    # model_forward(X_test1, w, b, 3)
    # model_forward(X_test1, np.array([w_new]), np.array([b_new]), 1)