import mlx.core as mx
import numpy as np

a = mx.ones((5,5))
b = a.__copy__()
b[1,1] = 2
b = mx.clip(b,-1,0.5)
print(b)
print(a)