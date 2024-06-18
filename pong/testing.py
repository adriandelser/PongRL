import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt

def compute_discounted_rewards(rewards, gamma:float = 0.99):
    #ensure rewards array is of float dtype
    discounted_rewards = mx.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

a = mx.zeros((5))
a[-1] = -1
print(compute_discounted_rewards(a))
# print(a,b)

diff = np.load('obs.npy')
plt.imshow(diff)
plt.show()
