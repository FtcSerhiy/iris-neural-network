import unittest
import numpy as np
import app

class TestApp(unittest.TestCase):
    def test_neutal(self):
        inputs = np.array([1,1,0])
        neural = app.Neural()
        neural.training()
        print(neural.run(inputs))

