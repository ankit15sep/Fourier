import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def generate_random_noise(mean, std, num_samples):
    # generate white noise
    white_noise = np.random.normal(loc=mean, scale=std, size=num_samples)
    return white_noise

# parameters
mean = 0
std = 1
num_samples = 1024
dx = 0.004
end_t = num_samples * dx


# Create a pulse signal
sig = np.zeros(100)
sig[10:20] = 1.0
# sig = generate_random_noise(mean, std, num_samples)
# Filter the signal
b, a = signal.butter(2, fs=250, Wn=100)
filtered_1 = signal.lfilter(b, a, sig)
b, a = signal.butter(4, fs=250, Wn=100)
filtered_2 = signal.lfilter(b, a, sig)
filtered_2 = signal.filtfilt(b, a, sig)
# filtered = signal.lfilter(b, a, sig)
taps = signal.firwin(11, 100, fs=250)
filtered_fir = signal.lfilter(taps, 1.0, sig)

plt.figure(figsize=(10, 4))
plt.plot(sig, label="Original Pulse", alpha=0.5)
plt.plot(filtered_1, label="IIR Filtered (Butterworth 6)", linewidth=2)
plt.plot(filtered_2, label="IIR Filtered (Butterworth 1)", linewidth=2)
plt.plot(filtered_fir, label="FIR Filtered", linewidth=2)
plt.title("Time Domain Effect: Smoothing and 'Ringing' Tail")
plt.legend()
plt.show()


# 3. Calculate Group Delay
w_iir, gd_iir = signal.group_delay((b, a), fs=250)
w_fir, gd_fir = signal.group_delay((taps, [1.0]), fs=250)

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.plot(w_fir, gd_fir, label='FIR (Linear Phase)', linewidth=2)
plt.plot(w_iir, gd_iir, label='IIR (Butterworth)', linewidth=2)

plt.title('Phase Distortion: FIR vs IIR (Group Delay)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Delay (Samples)')
plt.axvline(100, color='red', linestyle='--', label='Cutoff Frequency')
plt.grid(True)
plt.legend()
plt.show()