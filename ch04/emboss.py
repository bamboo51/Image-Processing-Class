import os
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    img = img.astype(np.float32)
    shifted = np.zeros_like(img)

    h, w = img.shape

    shifted[dy:h, dx:w] = img[0 : h - dy, 0 : w - dx]

    return shifted


def saveas_pgm(img: np.ndarray, img_name: str) -> None:
    """
    Save a grayscale image as a PGM file.
    """
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    cv2.imwrite("kadai0504/" + img_name + ".png", img)

    with open(f"kadai0504/{img_name}.pgm", "w") as f:
        f.write("P2\n")
        f.write(f"{img.shape[1]} {img.shape[0]}\n")
        f.write("255\n")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                f.write(f"{img[i, j]} ")
            f.write("\n")


def emboss(img: np.ndarray, shift: np.ndarray = np.array([1, 1])) -> List[np.ndarray]:
    img = img.astype(np.float32)

    result = []

    neg = 255 - img
    result.append(neg)

    move = shift_image(neg, dx=shift[0], dy=shift[1])
    result.append(move)

    embossed = img + move - 128
    embossed = np.clip(embossed, 0, 255)
    result.append(embossed)

    return result


if __name__ == "__main__":
    IMAGE_DIR = Path("../images")
    img_c = cv2.imread(str(IMAGE_DIR / "C.pgm"), cv2.IMREAD_GRAYSCALE)
    img_d = cv2.imread(str(IMAGE_DIR / "D.pgm"), cv2.IMREAD_GRAYSCALE)

    os.makedirs("./kadai0504", exist_ok=True)

    result_c = emboss(img_c, np.array([5, 0]))
    result_d = emboss(img_d, np.array([5, 0]))
    saveas_pgm(result_c[-1], "C_embossed-shift-right")
    saveas_pgm(result_d[-1], "D_embossed-shift-right")

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 4, 1)
    plt.imshow(img_c, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("C original")

    for i, img in enumerate(result_c):
        plt.subplot(2, 4, i + 2)
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.title(f"C step {i + 1}")

    plt.subplot(2, 4, 5)
    plt.imshow(img_d, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("D original")

    for i, img in enumerate(result_d):
        plt.subplot(2, 4, i + 6)
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.title(f"D step {i + 1}")

    plt.tight_layout()
    plt.savefig("kadai0504/kadai0504-shift-right.png")
    plt.show()

    result_c = emboss(img_c, np.array([0, 5]))
    result_d = emboss(img_d, np.array([0, 5]))
    saveas_pgm(result_c[-1], "C_embossed-shift-down")
    saveas_pgm(result_d[-1], "D_embossed-shift-down")

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 4, 1)
    plt.imshow(img_c, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("C original")

    for i, img in enumerate(result_c):
        plt.subplot(2, 4, i + 2)
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.title(f"C step {i + 1}")

    plt.subplot(2, 4, 5)
    plt.imshow(img_d, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("D original")

    for i, img in enumerate(result_d):
        plt.subplot(2, 4, i + 6)
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.title(f"D step {i + 1}")

    plt.tight_layout()
    plt.savefig("kadai0504/kadai0504-shift-down.png")
    plt.show()
