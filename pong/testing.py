import mlx.core as mx
import numpy as np

a = mx.ones((5,))
b = mx.ones((5,))*3

c = a*b
print(mx.sum(c) - a@b)
print(mx.sum(c), a@b)