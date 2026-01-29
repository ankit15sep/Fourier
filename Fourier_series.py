import numpy as np
import matplotlib.pyplot as plt


def create_triangular_hat_function(l, points=100):
    """Create a hat function over one period."""
    x = np.linspace(0, l, points)
    condlist = [x < l / 4, (l / 4 <= x) & (x < l / 2), (l / 2 <= x) & (x <= 3 * l / 4), x > 3 * l / 4]
    funclist = [0, lambda x: (4/l * x) - 1, lambda x: - (4 / l) * x + 3, 0]
    y = np.piecewise(x, condlist, funclist)
    return x, y

def create_hat_function(l, points=100):
    """Create a hat function over one period."""
    x = np.linspace(0, l, points)
    condlist = [x < l / 4, (l / 4 <= x) & (x < 3*l / 4), x > 3 * l / 4]
    funclist = [0, 1, 0]
    y = np.piecewise(x, condlist, funclist)
    return x, y

def fourier_coefficients(f, x, l):
    def a(k):
        summand = np.dot(f, np.cos(2 * np.pi * k * x / l))
        coeff_a = (2 / l) * summand * (x[1] - x[0])
        return coeff_a
    def b(k):
        summand = np.dot(f, np.sin(2 * np.pi * k * x / l))
        coeff_b = (2 / l) * summand * (x[1] - x[0])
        return coeff_b
    return a, b


# x,y = create_hat_function(1, points=1000)
l = 4
x,y = create_triangular_hat_function(l, points=500)
# print(1/(x[1] - x[0]))
# plt.plot(x, y, linewidth=2, label='Hat Function', color='black')
# plt.show()
# ft = 0
a0 = (1/2) * fourier_coefficients(y, x, l)[0](0)
ft = a0
ak_list = []
bk_list = []
acc_error = []
for k in range(1, 251):
    a, b = fourier_coefficients(y, x, l)
    ft += a(k) * np.cos(2 * np.pi * k * x/l) + b(k) * np.sin(2 * np.pi * k * x/l)
    error = np.linalg.norm(y - ft) / np.linalg.norm(y)
    acc_error.append(error)
    ak_list.append(a(k))
    bk_list.append(b(k))
# plt.plot(x, ft, label=f'Fourier Series up to k={k}', linewidth=0.8)
#
# plt.legend()
# plt.show()
# plt.plot(np.arange(0, len(ak_list)), ak_list, label='ak coefficients')
# plt.semilogy()
# plt.legend()
# plt.show()
# plt.plot(np.arange(0, len(bk_list)), bk_list, label='bk coefficients')
# plt.semilogy()
# plt.legend()
# plt.show()
plt.plot(np.arange(1, len(acc_error)+1), acc_error, label='Accumulated Error')
plt.semilogy()
plt.legend()
plt.show()
