import mlx.core as mx
import matplotlib.pyplot as plt
from pathlib import Path

w = mx.load('weights.safetensors')

print(w.keys())
w1 = w['layers.0.weight']
print(w1.shape)
w1 = mx.reshape(w1,(-1,80,80))

fig, ax = plt.subplots(10,20)
for i, w in enumerate(w1):
    # print(w)
    # print(i, i//10, i%10)
    axs = ax[i//20,i%20]
    axs.imshow(w)
    axs.grid(False)
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    axs.axis('off')  # This removes the axis completely

plt.show()

# import mlx.core as mx
# import matplotlib.pyplot as plt
# import numpy as np

# w = mx.load('../weights.safetensors')
# w1 = w['fc1.weight']
# print(w1.shape)
# w1 = mx.reshape(w1, (-1, 80, 80))

# # Number of images per row and column
# num_cols = 20
# num_rows = 10

# # Image dimensions
# img_height, img_width = 80, 80

# # Create a large array to hold all the smaller images
# big_image = np.zeros((num_rows * img_height, num_cols * img_width))

# for i, weight in enumerate(w1):
#     row = i // num_cols
#     col = i % num_cols
#     big_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = weight

# plt.figure(figsize=(15, 7))
# plt.imshow(big_image, cmap='gray')
# plt.axis('off')  # Turn off the axis

# # Save the large image if desired
# plt.savefig('big_image.png', bbox_inches='tight', pad_inches=0)

# plt.show()
