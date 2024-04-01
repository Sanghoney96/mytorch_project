import unittest
import numpy as np

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from variable import Variable
from function import square
from utils import numerical_grad


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()

        num_grad = numerical_grad(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


unittest.main()
