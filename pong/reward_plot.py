import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt


n = 50
r1 = np.load('returns_1.npy')
r2 = np.load('returns_2.npy')
r3 = np.load('returns_3.npy')
r4 = np.load('returns_4.npy')
r5 = np.load('returns_5.npy')
r6 = np.load('returns_6.npy')
r7 = np.load('returns_7.npy')



r = np.concatenate((r1, r2, r3, r4, r5, r6,r7))
cumul_r = np.cumsum(r)
print(f"Best return so far = {r.max()}")
# r = np.pad(r, ((n-1)//2, (n-1)//2), 'constant', constant_values=-1)
# print(r, len(r))
r = np.convolve(r, [1/n]*n, mode='valid')

# print(r, len(r))
plt.xlabel('Number of full matches played (first to 21)')
# plt.ylabel('Return per match (min -21, max 21)')
plt.ylabel('Cumulative return')

plt.plot(cumul_r)
# plt.plot(r4)
plt.show()

