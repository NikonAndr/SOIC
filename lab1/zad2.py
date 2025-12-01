import numpy as np
import matplotlib.pyplot as plt

def h1(t):
    if (t >= 0 and t < 1):
        return 1
    else: 
        return 0
    
def h2(t):
    if (t >= -0.5 and t < 0.5):
        return 1
    else:
        return 0
    
def h3(t):
    if (abs(t) <= 1):
        return 1 - abs(t)
    else:
        return 0

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

def interpolate_1d(signal, factor, kernel):
    N = len(signal)
    spacing = 1

    x_old = np.arange(N)
    x_new = np.linspace(0, N - 1, N * factor)

    out = np.zeros(len(x_new))

    for i in range(len(x_new)):
        val = 0
        for j in range(N):
            t = (x_new[i] - x_old[j]) / spacing
            val += signal[j] * kernel(t)
        out[i] = val

    return out 

def upsample(image, factor, kernel):
    img_h, img_w = image.shape

    expanded_rows = np.zeros((img_h, img_w * factor))

    for i in range(img_h):
        expanded_rows[i] = interpolate_1d(image[i], factor, kernel)

    new_h = img_h * factor
    new_w = img_w * factor
    expanded = np.zeros((new_h, new_w))

    for j in range(new_w):
        col = expanded_rows[:, j]
        expanded[:, j] = interpolate_1d(col, factor, kernel)

    return expanded

def maxpool(image, factor):
    h, w = image.shape

    new_h = int(np.ceil(h / factor))
    new_w = int(np.ceil(w / factor))

    out = np.zeros((new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            block = image[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            out[i, j] = np.max(block)

    return out


def mse(a, b):
    return np.mean((a - b) ** 2)

img = plt.imread("lab1/image.png")

if img.ndim == 3:
    img = img.mean(axis=2)

#Original -> Convolve -> Downsample test
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

K = mean_kernel(3)
blurred = convolve2d(img, K)

plt.subplot(1,3,2)
plt.imshow(blurred, cmap='gray')
plt.title("Convolve (3x3)")
plt.axis('off')

small_avg = downsample(img, 4, 3)

plt.subplot(1,3,3)
plt.imshow(small_avg, cmap='gray')
plt.title("Downsample (factor 4)")
plt.axis('off')

plt.show()


#Original -> Downsample -> Upsample test
factor = 4
kernel = h3  

small = downsample(img, factor, 3)
restored = upsample(small, factor, kernel)

H, W = img.shape
restored = restored[:H, :W]

error_avg = mse(img, restored)
print("MSE (średnia → upsample):", error_avg)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(small, cmap='gray')
plt.title(f"Downsample ×{factor}")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(restored, cmap='gray')
plt.title("Upsample")
plt.axis('off')

plt.show()


#Original -> Downsample (maxpooling) -> Upsample
small_max = maxpool(img, factor)
restored_max = upsample(small_max, factor, kernel)
restored_max = restored_max[:H, :W]

error_max = mse(img, restored_max)
print("MSE (maxpool → upsample):", error_max)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(small_max, cmap='gray')
plt.title(f"Downsample ×{factor} (maxpool)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(restored_max, cmap='gray')
plt.title("Upsample")
plt.axis('off')

plt.show()
