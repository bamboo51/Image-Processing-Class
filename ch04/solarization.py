import os
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def solarization(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    y = 127 * (1 + np.cos(3 * 2 * np.pi * img / 255))
    return np.clip(y, 0, 254)


def saveas_pgm(img: np.ndarray, img_name: str) -> None:
    """
    Save a grayscale image as a PGM file.
    """
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    cv2.imwrite("kadai0502/" + img_name + ".png", img)

    with open(f"kadai0502/{img_name}.pgm", "w") as f:
        f.write("P2\n")
        f.write(f"{img.shape[1]} {img.shape[0]}\n")
        f.write("255\n")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                f.write(f"{img[i, j]} ")
            f.write("\n")


def plt_compare(
    img1_list: List[np.ndarray],
    img2_list: List[np.ndarray],
    names: List[str],
    output_name: str,
) -> None:
    rows = len(img1_list)

    plt.figure(figsize=(12, 4 * rows))

    for i, (img1, img2, name) in enumerate(zip(img1_list, img2_list, names)):
        saveas_pgm(img2, name)

        img1_u8 = np.clip(np.rint(img1), 0, 255).astype(np.uint8)
        img2_u8 = np.clip(np.rint(img2), 0, 255).astype(np.uint8)

        plt.subplot(rows, 3, i * 3 + 1)
        im1 = plt.imshow(img1_u8, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{name} Original")
        plt.axis("off")
        plt.colorbar(im1, fraction=0.046, pad=0.04)

        plt.subplot(rows, 3, i * 3 + 2)
        im2 = plt.imshow(img2_u8, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{name} Filtered")
        plt.axis("off")
        plt.colorbar(im2, fraction=0.046, pad=0.04)

        # Difference
        plt.subplot(rows, 3, i * 3 + 3)
        diff = img2.astype(np.float32) - img1.astype(np.float32)
        im3 = plt.imshow(diff, cmap="gray")  # auto-scale is better here
        plt.title(f"{name} Difference")
        plt.axis("off")
        plt.colorbar(im3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{output_name}.png")
    plt.show()


def plt_img_hist_compare(
    img1_list: List[np.ndarray],
    img2_list: List[np.ndarray],
    names: List[str],
    output_name: str,
) -> None:

    rows = len(img1_list)

    plt.figure(figsize=(16, 4 * rows))

    for i, (img1, img2, name) in enumerate(zip(img1_list, img2_list, names)):
        saveas_pgm(img2, name)

        img1_u8 = np.clip(np.rint(img1), 0, 255).astype(np.uint8)
        img2_u8 = np.clip(np.rint(img2), 0, 255).astype(np.uint8)

        # --- Original Image ---
        plt.subplot(rows, 4, i * 4 + 1)
        plt.imshow(img1_u8, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{name} Original")
        plt.axis("off")

        # --- Filtered Image ---
        plt.subplot(rows, 4, i * 4 + 2)
        plt.imshow(img2_u8, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{name} Solarized")
        plt.axis("off")

        # --- Original Histogram ---
        plt.subplot(rows, 4, i * 4 + 3)
        plt.hist(img1_u8.ravel(), bins=256, range=(0, 255))
        plt.title("Original Hist")
        plt.xlabel("Gray level")
        plt.ylabel("Freq")

        # --- Filtered Histogram ---
        plt.subplot(rows, 4, i * 4 + 4)
        plt.hist(img2_u8.ravel(), bins=256, range=(0, 255))
        plt.title("Solarized Hist")
        plt.xlabel("Gray level")
        plt.ylabel("Freq")

    plt.tight_layout()
    plt.savefig(f"kadai0502/{output_name}.png")
    plt.show()


if __name__ == "__main__":
    IMAGE_DIR = Path("../images")
    img_c = cv2.imread(str(IMAGE_DIR / "C.pgm"), cv2.IMREAD_GRAYSCALE)
    img_d = cv2.imread(str(IMAGE_DIR / "D.pgm"), cv2.IMREAD_GRAYSCALE)

    if not os.path.exists("kadai0502"):
        os.makedirs("kadai0502")

    x = np.arange(0, 256)
    y = 127 * (1 + np.cos(3 * 2 * np.pi * x / 255))

    plt.figure(figsize=(6, 5))
    plt.plot(x, y)
    plt.xlim(0, 255)

    plt.xlabel("Input intensity (0–255)")
    plt.ylabel("Output intensity (0–255)")
    plt.title("Solarization Curve")

    plt.grid(True)
    plt.savefig("kadai0502/solarization_curve.png")
    plt.show()

    img_c_solarized = solarization(img_c)
    img_d_solarized = solarization(img_d)
    plt_compare(
        [img_c, img_d],
        [img_c_solarized, img_d_solarized],
        ["C_Solarized", "D_Solarized"],
        "Solarization",
    )
    plt_img_hist_compare(
        [img_c, img_d],
        [img_c_solarized, img_d_solarized],
        ["C", "D"],
        "Solarization_All",
    )
