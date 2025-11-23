import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(pow(x, -1))

def f3(x):
    return np.sign(np.sin(8 * x))

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

def f_i(N, RANGE, f, factor, kernel_f):

    x = np.linspace(RANGE[0], RANGE[1], N)
    y = f(x)

    x_m = np.linspace(RANGE[0], RANGE[1], N * factor)
    y_m = np.zeros_like(x_m)

    w = (RANGE[1] - RANGE[0]) / (N - 1)

    for i in range(len(x_m)):
        for j in range(len(x)):
            t = (x_m[i] - x[j]) / w
            weight = kernel_f(t)
            y_m[i] += y[j] * weight

    return x, y, x_m, y_m

def mse(x_m, y_m, f):
    y_true = f(x_m)
    return np.mean((y_true - y_m) ** 2)

#MAIN
functions = [(f1, "f1(x) = sin(x)"), 
             (f2, "f2(x) = sin (x^-1)"),
             (f3, "f3(x) = sign(sin(8x))")]

kernels = [(h1, "h1 (box left)"),
           (h2, "h2 (box center)"),
           (h3, "h3 (triangle)")]

factors = [2, 4, 10]

N = 100
RANGE = (-np.pi, np.pi)

for f, fname in functions:
    fig, axes = plt.subplots(3, 3, figsize=(12,10))
    fig.suptitle(f"Interpolation results for {fname}", fontsize=16)

    for ki, (kernel, kname) in enumerate(kernels):
        for fi, factor in enumerate(factors):
            
            ax = axes[ki, fi]

            x, y, x_m, y_m = f_i(N, RANGE, f, factor, kernel)

            error = mse(x_m, y_m, f)

            ax.plot(x, y, 'o', markersize=3, label='Original')
            ax.plot(x_m, y_m, '-', label=f"{kname} x{factor}")

            ax.set_title(f"{kname}, ×{factor}\nMSE={error:.7f}")
            ax.grid(True)
    plt.tight_layout()
    plt.show()

#Ns to examine the impact of the number and distribution of sample points on the quality of covolution
Ns = [20, 50, 100, 200]
results = []

for N in Ns:
    x_m, y_m = f_i(N, RANGE, f1, 4, h3)[2:4]
    results.append(mse(x_m, y_m, f1))

for N, result in zip(Ns, results):
    print(f"N={N} -> MSE={result:.7f}")





    
