import numpy as np


def dft_matrix(N, inverse=False):
    """
    Forward:
        W[k, n] = 1/sqrt(N) * exp(-j * 2 * pi * k * n / N)

    Inverse:
        W[k, n] = 1/sqrt(N) * exp(j * 2 * pi * k * n / N)

    """
    n = np.arange(N)
    k = n.reshape((N, 1))

    sign = 1 if inverse else -1
    W = np.exp(sign * 2j * np.pi * k * n / N)
    return W / np.sqrt(N)


def dft2d(image):
    """
    Forward DFT of a 2D image.
    """
    image = np.asarray(image, dtype=np.float64)
    M, N = image.shape
    W_M = dft_matrix(M, inverse=False)
    W_N = dft_matrix(N, inverse=False)
    return np.dot(W_M, np.dot(image, W_N))


def idft2d(F):
    """
    Inverse DFT of a 2D image.
    """
    F = np.asarray(F, dtype=np.complex128)
    M, N = F.shape
    W_M_inv = dft_matrix(M, inverse=True)
    W_N_inv = dft_matrix(N, inverse=True)

    image = np.dot(W_M_inv, np.dot(F, W_N_inv))
    return np.real(image)


def fftshift2d(F):
    """
    Shifts the zero-frequency component to the center of the spectrum.
    """
    M, N = F.shape
    return np.roll(np.roll(F, M // 2, axis=0), N // 2, axis=1)


def ifftshift2d(F):
    """
    Shifts the zero-frequency component back to the top-left corner of the spectrum.
    """
    M, N = F.shape
    return np.roll(np.roll(F, -(M // 2), axis=0), -(N // 2), axis=1)


def normalize_image(img):
    """
    Normalize the image to the range [0, 1].
    """
    img = np.real(img)
    min_val = img.min()
    max_val = img.max()

    if max_val - min_val < 1e-12:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)
