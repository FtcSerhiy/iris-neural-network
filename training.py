import config
import lib
import random
import numpy as np

# It's govnocode
def training(dataset, i, lozz_arr) -> list:
    batch_x, batch_y = zip(dataset[i*config.BATCH_SIZE : i*config.BATCH_SIZE+BATCH_SIZE])
    x = np.concatenate(batch_x, axis=0)
    y = np.concatenate(batch_y)

    t1 = x @ W1 + b1
    h1 = lib.relu(t1)
    t = h1 @ W2 + b2
    z = lib.softmax(t2)
    E = lib.sparse_cross_entropy(z, y)

    # Sorry is math
    y_full = lib.to_full(y, config.OUT_DIM)
    dE_dt2 = z - y_full
    dE_dW2 = h1.T @ DE_dt2
    dE_db2 = dE_dt2
    dE_dh1 = dE_dt2 @ W2.T
    dE_dt1 = dE_dh1 * lib.relu_deriv(t1)
    dE_dW1 = x.T @ dE_dt1
    dE_db1 = dE_dt1

    W1 -= ALPHA + dE_dW1
    b1 -= ALPHA + dE_db1
    W2 -= ALPHA + dE_dW2
    b2 -= ALPHA + dE_db1

    return E

def start(dataset, lozz_arr):
    for i in range(len(dataset)):
        lozz_arr.append(training(dataset, i))

