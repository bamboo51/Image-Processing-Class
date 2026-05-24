import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def saveas_pgm(img: np.ndarray, img_name: str) -> None:
    """
    Save a grayscale image as a PGM file.
    """
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    cv2.imwrite("kadai0503/" + img_name + ".png", img)

    with open(f"kadai0503/{img_name}.pgm", "w") as f:
        f.write("P2\n")
        f.write(f"{img.shape[1]} {img.shape[0]}\n")
        f.write("255\n")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                f.write(f"{img[i, j]} ")
            f.write("\n")


def alpha_blending_horizontal(img_c: np.ndarray, img_d: np.ndarray) -> np.ndarray:
    img_c = img_c.astype(np.float32)
    img_d = img_d.astype(np.float32)

    h, w = img_c.shape

    alpha = np.linspace(1.0, 0.0, w, dtype=np.float32)
    alpha = alpha.reshape(1, w)

    blended = alpha * img_c + (1 - alpha) * img_d

    return blended


if __name__ == "__main__":
    IMAGE_DIR = Path("../images")
    img_c = cv2.imread(str(IMAGE_DIR / "C.pgm"), cv2.IMREAD_GRAYSCALE)
    img_d = cv2.imread(str(IMAGE_DIR / "D.pgm"), cv2.IMREAD_GRAYSCALE)

    img_blend = alpha_blending_horizontal(img_c, img_d)

    saveas_pgm(img_blend, "C_D_alpha_blending_horizontal")

    plt.figure(figsize=(6, 5))
    plt.imshow(
        np.clip(np.rint(img_blend), 0, 255).astype(np.uint8),
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    plt.title("Horizontal Alpha Blending: C → D")
    plt.axis("off")
    plt.show()
