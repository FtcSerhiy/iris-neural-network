import numpy as np

INPUT_DIM = 4
OUT_DIM = 3
H_DIM = 10
ALPHA = 0.0002
NUM_EPOCHS = 400
BATCH_SIZE = 50
CLASS_NAMES = ['Setosa', 'Versicolor', 'Verginica']

W1 = np.random.randn(INPUT_DIM, H_DIM)
b1 = np.random.randn(H_DIM)
W2 = np.random.randn(H_DIM, OUT_DIM)
b2 = np.random.randn(OUT_DIM)

W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)
