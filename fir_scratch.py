import numpy as np
import matplotlib.pyplot as plt

"""
# scratch implementation of FIR filter
# h[n] = sin(2*pi*fc*n/fs) / (pi*n)   --> sinc function
# w[n] = 0.54 + 0.46*cos(2*pi*n/(N-1)) , N = 101  --> Hamming window
"""


fc = 50
fs = 250
N = 1024
h = np.zeros(N)
t = np.zeros(N)
w = np.zeros(N)
h_pre = np.zeros(N)
low = int(-(N-1)/2)
high = int((N-1)/2 + 1)

for i, n in enumerate(range(low, high)):
    if n == 0:
        w[i] = 0.54 + 0.46*np.cos(2*np.pi*n/(101-1))
        h_pre[i] = 2*fc/fs
        h[i] = h_pre[i] * w[i]
        t[i] = n
    else:
        w[i] = 0.54 + 0.46 * np.cos(2 * np.pi * n / (101 - 1))
        h_pre[i] = np.sin(2*np.pi*fc*n/fs)/(np.pi*n)
        h[i] = h_pre[i] * w[i]
        t[i] = n

plt.plot(t, h)
plt.plot(t, h_pre)
plt.title("Ideal Low-Pass Filter Impulse Response")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid()
plt.show()

coeff = np.fft.fft(h, n=1024)
freqs = np.fft.fftfreq(1024, d=1/fs)
freqs_shifted = np.fft.fftshift(freqs)
plt.plot(freqs_shifted, np.abs(np.fft.fftshift(coeff)))
coeff_pre = np.fft.fft(h_pre, n=1024)
plt.plot(freqs_shifted, np.abs(np.fft.fftshift(coeff_pre)))
plt.show()