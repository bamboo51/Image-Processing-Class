import cv2
import matplotlib.pyplot as plt
import numpy as np
from custom_fft import dft2d, fftshift2d, normalize_image


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


if __name__ == "__main__":
    # Params
    M = 128
    N = 128
    P_list = [8, 16, 64, 128]
    Q_list = [8, 32, 64, 128]

    # 4 rows × 2 columns
    fig, axes = plt.subplots(len(P_list), 2, figsize=(10, 16))

    for idx, (P, Q) in enumerate(zip(P_list, Q_list)):
        # =====================================================
        # Create image
        # =====================================================

        img = create_sinusoidal_image(M, N, P, Q)

        # =====================================================
        # DFT
        # =====================================================

        F = dft2d(img)

        spectrum = np.log1p(np.abs(fftshift2d(F)))
        spectrum = normalize_image(spectrum)

        # =====================================================
        # Plot image
        # =====================================================

        axes[idx, 0].imshow(img, cmap="gray")
        axes[idx, 0].set_title(f"Image\nP={P}, Q={Q}")
        axes[idx, 0].axis("off")

        # =====================================================
        # Plot spectrum
        # =====================================================

        axes[idx, 1].imshow(spectrum, cmap="gray")
        axes[idx, 1].set_title("DFT Spectrum")
        axes[idx, 1].axis("off")

        # =====================================================
        # Save images
        # =====================================================

        cv2.imwrite(f"sinusoidal_{P}_{Q}.png", (img * 255).astype(np.uint8))

        cv2.imwrite(f"dft_spectrum_{P}_{Q}.png", (spectrum * 255).astype(np.uint8))

    plt.tight_layout()
    plt.savefig("sinusoidal_and_spectra.png")
    plt.show()
