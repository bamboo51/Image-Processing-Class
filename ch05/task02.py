import cv2
import matplotlib.pyplot as plt
import numpy as np
from custom_fft import dft2d, fftshift2d, idft2d, ifftshift2d, normalize_image


def show_spectrum(F, title="Magnitude Spectrum", save_title=None):
    """
    Display log magnitude spectrum.
    """
    F_shifted = fftshift2d(F)

    magnitude = np.log1p(np.abs(F_shifted))
    magnitude_norm = normalize_image(magnitude)

    if save_title is not None:
        cv2.imwrite(f"{save_title}.png", (magnitude_norm * 255).astype(np.uint8))

    plt.figure(figsize=(5, 5))
    plt.imshow(magnitude_norm, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def create_sinusoidal_image(M=128, N=128, P=16, Q=32):
    """
    Creates a sinusoidal image of size MxN with a sinusoidal pattern of size PxQ.
    - sine variation in vertical direction with period P
    - sine variation in horizontal direction with period Q
    """
    img = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        for n in range(N):
            vertical_direction_wave = np.sin(2.0 * np.pi * m / P)
            horizontal_direction_wave = np.sin(2.0 * np.pi * n / Q)

            img[m, n] = vertical_direction_wave + horizontal_direction_wave
    return normalize_image(img)


def directional_y_filter(M, N, band=1):
    """
    Creates a directional x-axis filter of size MxN with a cutoff frequency.
    """
    H = np.zeros((M, N))
    cx = N // 2
    H[:, cx - band : cx + band] = 1
    return H


def directional_x_filter(M, N, band=1):
    """
    Creates a directional y-axis filter of size MxN with a cutoff frequency.
    """
    H = np.zeros((M, N))
    cy = M // 2
    H[cy - band : cy + band, :] = 1
    return H


if __name__ == "__main__":
    # Params
    M = 128
    N = 128
    P = [8, 16, 64, 128]
    Q = [8, 32, 64, 128]

    fig, axes = plt.subplots(len(P), 4, figsize=(16, 16))

    for idx, (p, q) in enumerate(zip(P, Q)):
        # Create image
        img = create_sinusoidal_image(M, N, p, q)

        # DFT
        F = dft2d(img)

        # Spectrum
        spectrum = np.log1p(np.abs(fftshift2d(F)))
        spectrum = normalize_image(spectrum)

        # y-axis filter
        Y = directional_y_filter(M, N)

        F_filtered_shifted = fftshift2d(F) * Y
        F_filtered = ifftshift2d(F_filtered_shifted)

        # Filtered spectrum
        filtered_spectrum = np.log1p(np.abs(F_filtered_shifted))
        filtered_spectrum = normalize_image(filtered_spectrum)

        # Inverse DFT
        filtered_img = np.real(idft2d(F_filtered))
        filtered_img = normalize_image(filtered_img)

        # Original image
        axes[idx, 0].imshow(img, cmap="gray")
        axes[idx, 0].set_title(f"Image\nP={p}, Q={q}")
        axes[idx, 0].axis("off")

        # Original spectrum
        axes[idx, 1].imshow(spectrum, cmap="gray")
        axes[idx, 1].set_title("Spectrum")
        axes[idx, 1].axis("off")

        # Filtered spectrum
        axes[idx, 2].imshow(filtered_spectrum, cmap="gray")
        axes[idx, 2].set_title("Filtered Spectrum")
        axes[idx, 2].axis("off")

        # Filtered image
        axes[idx, 3].imshow(filtered_img, cmap="gray")
        axes[idx, 3].set_title("Filtered Image")
        axes[idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig("y-axis-pass_filter.png")
    plt.show()

    fig, axes = plt.subplots(len(P), 4, figsize=(16, 16))
    for idx, (p, q) in enumerate(zip(P, Q)):
        # Create image
        img = create_sinusoidal_image(M, N, p, q)

        # DFT
        F = dft2d(img)

        # Spectrum
        spectrum = np.log1p(np.abs(fftshift2d(F)))
        spectrum = normalize_image(spectrum)

        # x-axis filter
        X = directional_x_filter(M, N)

        F_filtered_shifted = fftshift2d(F) * X
        F_filtered = ifftshift2d(F_filtered_shifted)

        # Filtered spectrum
        filtered_spectrum = np.log1p(np.abs(F_filtered_shifted))
        filtered_spectrum = normalize_image(filtered_spectrum)

        # Inverse DFT
        filtered_img = np.real(idft2d(F_filtered))
        filtered_img = normalize_image(filtered_img)

        # Original image
        axes[idx, 0].imshow(img, cmap="gray")
        axes[idx, 0].set_title(f"Image\nP={p}, Q={q}")
        axes[idx, 0].axis("off")

        # Original spectrum
        axes[idx, 1].imshow(spectrum, cmap="gray")
        axes[idx, 1].set_title("Spectrum")
        axes[idx, 1].axis("off")

        # Filtered spectrum
        axes[idx, 2].imshow(filtered_spectrum, cmap="gray")
        axes[idx, 2].set_title("Filtered Spectrum")
        axes[idx, 2].axis("off")

        # Filtered image
        axes[idx, 3].imshow(filtered_img, cmap="gray")
        axes[idx, 3].set_title("Filtered Image")
        axes[idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig("x-axis-pass_filter.png")
    plt.show()
