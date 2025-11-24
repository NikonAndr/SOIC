import numpy as np
import matplotlib.pyplot as plt

def mean_kernel(size):
    return np.ones((size, size)) / (size*size)

def convolve2d(image, kernel):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image, dtype=float)

    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i+k_h, j:j+k_w]
            output [i, j] = np.sum(region * kernel)

    return output 

def downsample(image, factor, kernel_size):
    K = mean_kernel(kernel_size)
    blurred = convolve2d(image, K)
    return blurred[::factor, ::factor]

img = plt.imread("lab1/gigachad.png")

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Oryginalny obraz")
plt.axis('off')

K = mean_kernel(3)
blurred = convolve2d(img, K)

plt.subplot(1,3,2)
plt.imshow(blurred, cmap='gray')
plt.title("Po konwolucji")
plt.axis('off')

small = downsample(img, 4, 3)

plt.subplot(1,3,3)
plt.imshow(small, cmap='gray')
plt.title("Po downsample")
plt.axis('off')
plt.show()



