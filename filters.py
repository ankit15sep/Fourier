import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def generate_random_noise(mean, std, num_samples):
    # generate white noise
    white_noise = np.random.normal(loc=mean, scale=std, size=num_samples)
    return white_noise


def generate_sin_func(f, end_t, dx):
    # generate sine function
    t = np.arange(0, end_t, dx)
    sin_func = np.sin(2*np.pi*f*t)
    return sin_func

def generate_multi_sin_func(frequencies, end_t, dx):
    # generate multiple sine functions
    t = np.arange(0, end_t, dx)
    multi_sin_func = np.zeros_like(t)
    for f in frequencies:
        multi_sin_func += np.sin(2*np.pi*f*t)
    return multi_sin_func


def create_rectangular_mask(cutoff_freq, freqs):
    # create rectangular mask
    condlist = [np.abs(freqs) <= cutoff_freq, np.abs(freqs) > cutoff_freq]
    funclist = [1, 0]
    mask = np.piecewise(freqs, condlist, funclist)
    return mask


def create_fir_filter(numtaps, fs, cutoff):
    # create FIR filter
    taps = signal.firwin(numtaps, cutoff, fs=fs)
    return taps


def create_iir_filter(order, fs, cutoff):
    # create IIR filter
    b, a = signal.butter(order, Wn=cutoff, fs=fs)
    return b, a

# parameters
mean = 0
std = 1
num_samples = 1024
dx = 0.004
end_t = num_samples * dx

# Parameters for FIR filter
numtaps = 11      # Length of the filter (must be odd for some types)
fs = 250          # Sampling frequency in Hz
cutoff = 50        # Cutoff frequency in Hz


# generate signal and plot
# y = generate_random_noise(mean, std, num_samples)
y = generate_sin_func(20, end_t, dx)
frequencies = [20, 50]
# y = generate_multi_sin_func(frequencies, end_t, dx)

"""
. compute fourier coefficients. Len(fourier_coeff) = num_samples
. DFT -> {F_hat} = [Fourier matrix] * {f}
. Fourier matrix: N x N matrix where N = num_samples, has to be square to be invertible
. These are complex terms mirrored about the Nyquist frequency. 
  . The negative frequency terms are just the complex conjugates of the 
    positive frequency terms and are needed to reconstruct real signals. 
. For FFT, we ignore the negative frequency and scale the positive frequency 
  terms by 2/N to account for the energy in the negative frequencies.
. Get the frequencies corresponding to the Fourier coefficients in the same order.
. Note fourier coeffs are scaled by num_samples
"""
fourier_coeff = np.fft.fft(y)
freqs = np.fft.fftfreq(num_samples, d=dx)


"""
. plot shifted version, centered at 0 frequency. 
. Default output is from 0 to Nyquist to negative Nyquist to just below 0. 
. Not scaling by 2 here as plotting both pos and neg freqs.
"""
fourier_coeff_shifted = np.fft.fftshift(fourier_coeff)
freqs_shifted = np.fft.fftshift(freqs)

# apply rectangular filter in frequency domain
mask = create_rectangular_mask(cutoff, freqs_shifted)
fourier_coeff_filtered = fourier_coeff_shifted * mask

# inverse FFT on mask
filter_time_domain = np.fft.ifft(mask)
shifted_time_signal = np.fft.ifftshift(filter_time_domain)

# inverse FFT on filtered fourier coeffs to get filtered signal
y_filtered = np.fft.ifft(fourier_coeff_filtered)
y_og = np.fft.ifft(fourier_coeff)


# FIR filtering
taps = create_fir_filter(numtaps, fs, cutoff)
y_fir = signal.lfilter(taps, [1.0], y)
# Compute frequency response
w_fir, h_fir = signal.freqz(taps)
fft_y_fir = np.fft.fft(y_fir)
fft_y_fir_shifted = np.fft.fftshift(fft_y_fir)
fft_y_fir_freq = np.fft.fftfreq(len(fft_y_fir), d=dx)
fft_y_fir_freq_shifted = np.fft.fftshift(fft_y_fir_freq)

# IIR filtering
b, a = create_iir_filter(order=4, fs=fs, cutoff=cutoff)
y_iir = signal.filtfilt(b, a, y)
w_iir, h_iir = signal.freqz(b, a)
fft_y_iir = np.fft.fft(y_iir)
fft_y_iir_shifted = np.fft.fftshift(fft_y_iir)
fft_y_iir_freq = np.fft.fftfreq(len(fft_y_iir), d=dx)
fft_y_iir_freq_shifted = np.fft.fftshift(fft_y_iir_freq)



# plotting
# plt.plot(np.arange(0, end_t, dx), y, label='Original signal')   # original signal
# plt.plot(freqs[:num_samples//2], (np.abs(fourier_coeff[:num_samples//2]*(1/num_samples)*2))) # FFT for original signal
# plt.plot(freqs_shifted, np.abs(fourier_coeff_shifted), label='Original signal') # shifted FFT for original signal
# plt.plot(np.arange(0, end_t, dx), np.real(y_og))    # original signal from ifft of shifted fourier coeffs
# plt.plot(freqs_shifted, mask) # rectangular mask
# plt.plot(freqs_shifted, (np.abs(fourier_coeff_filtered)), label='Rectangular Mask') # FFT after applying rectangular mask
# plt.plot(np.arange(0, end_t, dx), np.real(shifted_time_signal), label='Rectangular Mask') # rectangular mask in time domain
# plt.plot(np.arange(0, end_t, dx), np.real(y_filtered), label='Rectangular Mask')    # filtered signal using rectangular mask
# plt.plot(taps, label='FIR Taps')  # FIR filter taps
plt.plot(w_fir * fs / (2 * np.pi), np.abs(h_fir), label='FIR Filter Response')  # FIR filter frequency response
plt.plot(w_iir * fs / (2 * np.pi), np.abs(h_iir), label='IIR Filter Response')  # IIR filter frequency response
# plt.plot(np.arange(0, end_t, dx), y_fir, label='FIR Filtered Signal')  # FIR filtered signal
# plt.plot(np.arange(0, end_t, dx), y_iir, label='IIR Filtered Signal')  # IIR filtered signal
# plt.plot(fft_y_fir_freq_shifted, np.abs(fft_y_fir_shifted), label='FIR Filter')  # FFT of FIR filtered signal
# plt.plot(fft_y_iir_freq_shifted, np.abs(fft_y_iir_shifted), label='IIR Filter')  # FFT of IIR filtered signal
# plt.axvline(cutoff, color='red', linestyle='--', label='Cutoff Frequency')
plt.semilogy()
plt.legend()
plt.show()