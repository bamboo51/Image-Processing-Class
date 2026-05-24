import matplotlib.pyplot as plt
import numpy as np
from custom_fft import (
    dft2d,
    fftshift2d,
    idft2d,
    ifftshift2d,
    normalize_image,
)
from PIL import Image


def create_ideal_lowpass(M, N, cutoff=30):
    cy, cx = M // 2, N // 2
    y, x = np.ogrid[:M, :N]
    distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return (distance <= cutoff).astype(np.float64)


def create_ideal_highpass(M, N, cutoff=30):
    return 1.0 - create_ideal_lowpass(M, N, cutoff)


def create_gaussian_lowpass(M, N, sigma=30):
    cy, cx = M // 2, N // 2
    y, x = np.ogrid[:M, :N]
    distance2 = (y - cy) ** 2 + (x - cx) ** 2
    return np.exp(-distance2 / (2 * sigma**2))


def create_gaussian_highpass(M, N, sigma=30):
    return 1.0 - create_gaussian_lowpass(M, N, sigma)


def apply_frequency_filter(F, H):
    F_shifted = fftshift2d(F)
    F_filtered_shifted = F_shifted * H
    F_filtered = ifftshift2d(F_filtered_shifted)
    filtered_img = np.real(idft2d(F_filtered))
    filtered_img = normalize_image(filtered_img)
    filtered_spectrum = np.log1p(np.abs(F_filtered_shifted))
    filtered_spectrum = normalize_image(filtered_spectrum)
    return filtered_img, filtered_spectrum


if __name__ == "__main__":
    image = Image.open("dataset-card.png").convert("L")
    img = np.asarray(image, dtype=np.float64)
    img = normalize_image(img)

    M, N = img.shape

    # DFT
    F = dft2d(img)
    original_spectrum = np.log1p(np.abs(fftshift2d(F)))
    original_spectrum = normalize_image(original_spectrum)

    # Filters
    H_ideal_low = create_ideal_lowpass(M, N, cutoff=40)
    H_ideal_high = create_ideal_highpass(M, N, cutoff=40)
    H_gauss_low = create_gaussian_lowpass(M, N, sigma=40)
    H_gauss_high = create_gaussian_highpass(M, N, sigma=40)

    # Filters params
    cutoff = 40
    sigma = 40

    # Apply filters
    ideal_low_img, ideal_low_spec = apply_frequency_filter(F, H_ideal_low)
    ideal_high_img, ideal_high_spec = apply_frequency_filter(F, H_ideal_high)
    gauss_low_img, gauss_low_spec = apply_frequency_filter(F, H_gauss_low)
    gauss_high_img, gauss_high_spec = apply_frequency_filter(F, H_gauss_high)

    # Plot
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(original_spectrum, cmap="gray")
    axes[0, 1].set_title("Original Spectrum")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(ideal_low_img, cmap="gray")
    axes[1, 0].set_title("Ideal Low-pass")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(ideal_low_spec, cmap="gray")
    axes[1, 1].set_title(f"Ideal LPF Spectrum; cutoff={cutoff}")
    axes[1, 1].axis("off")

    axes[2, 0].imshow(ideal_high_img, cmap="gray")
    axes[2, 0].set_title("Ideal High-pass")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(ideal_high_spec, cmap="gray")
    axes[2, 1].set_title(f"Ideal HPF Spectrum; cutoff={cutoff}")
    axes[2, 1].axis("off")

    axes[3, 0].imshow(gauss_low_img, cmap="gray")
    axes[3, 0].set_title(f"Gaussian Low-pass")
    axes[3, 0].axis("off")

    axes[3, 1].imshow(gauss_low_spec, cmap="gray")
    axes[3, 1].set_title(f"Gaussian LPF Spectrum; sigma={sigma}")
    axes[3, 1].axis("off")

    axes[4, 0].imshow(gauss_high_img, cmap="gray")
    axes[4, 0].set_title("Gaussian High-pass")
    axes[4, 0].axis("off")

    axes[4, 1].imshow(gauss_high_spec, cmap="gray")
    axes[4, 1].set_title(f"Gaussian HPF Spectrum; sigma={sigma}")
    axes[4, 1].axis("off")

    plt.tight_layout()
    plt.savefig("frequency_filter_results.png", dpi=150)
    plt.show()
    plt.close(fig)
