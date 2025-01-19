import numpy as np
import matplotlib.pyplot as plt

def generate_2d_cosine(freq_x, freq_y, size=256):
    spectrum = np.zeros((size, size), dtype=complex)
    shifted_spectrum = np.fft.fftshift(spectrum)

    mid = size // 2
    shifted_spectrum[mid + freq_x, mid + freq_y] = 1
    shifted_spectrum[mid - freq_x, mid - freq_y] = 1

    image = np.fft.ifft2(np.fft.ifftshift(shifted_spectrum))
    return np.real(image), np.abs(shifted_spectrum)

size = 256

freq_x, freq_y = 4, 0
image, spectrum = generate_2d_cosine(freq_x, freq_y, size)
plt.imsave('cosine_wave_4_0.pdf', image, cmap='gray')
plt.imsave('spectrum_4_0.pdf', spectrum, cmap='hot')

freq_x, freq_y = 10, 0
image, spectrum = generate_2d_cosine(freq_x, freq_y, size)
plt.imsave('cosine_wave_10_0.pdf', image, cmap='gray')
plt.imsave('spectrum_10_0.pdf', spectrum, cmap='hot')

freq_x, freq_y = 0, 4
image, spectrum = generate_2d_cosine(freq_x, freq_y, size)
plt.imsave('cosine_wave_0_4.pdf', image, cmap='gray')
plt.imsave('spectrum_0_4.pdf', spectrum, cmap='hot')

freq_x, freq_y = 0, 10
image, spectrum = generate_2d_cosine(freq_x, freq_y, size)
plt.imsave('cosine_wave_0_10.pdf', image, cmap='gray')
plt.imsave('spectrum_0_10.pdf', spectrum, cmap='hot')

freq_x, freq_y = 10, 10
image, spectrum = generate_2d_cosine(freq_x, freq_y, size)
plt.imsave('cosine_wave_10_10.pdf', image, cmap='gray')
plt.imsave('spectrum_10_10.pdf', spectrum, cmap='hot')
