import numpy as np


class nNet():
    def __init__(self):
        super(nNet, self).__init__()

    def maxPool2d(image):
        # 2x2 kernel
        k_size = (2, 2)

        height, width = image.shape
        new_height = height // k_size[0]
        new_width = width // k_size[1]

        # Pad image with zeros if necessary
        pad_h = k_size[0] - (image.shape[0] % k_size[0])
        pad_w = k_size[1] - (image.shape[1] % k_size[1])
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')

        # Reshape image into 2x2 blocks
        reshaped = padded.reshape(new_height, k_size[0], new_width, k_size[1])
        # Take max of each block
        pooled = np.max(reshaped, axis=(1, 3))

    def relu(x):
        for i, row in enumerate(x):
            for j, col in enumerate(row):
                # Keep positive values, set negative values to 0
                x[i][j] = max(0, x[i][j])
        return x

    def conv2d():
        pass

    def linear():
        pass
