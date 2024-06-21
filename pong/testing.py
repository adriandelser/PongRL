import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt


import numpy as np

a = mx.ones((1,5))
b = mx.zeros((1,5))

c = mx.concatenate((a,b), axis = 0)
print(c)