import os
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def filtering(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a 2D linear filter to a grayscale image.
    Padding is done with reflection.
    """
    img = img.astype(np.float32)
    kernel = kernel.astype(np.float32)
    h, w = img.shape[0], img.shape[1]
    new_img = np.zeros(img.shape, dtype=np.float32)
    cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2

    for y in range(h):
        for x in range(w):
            s = 0.0
            for m in range(-cy, cy + 1):
                for n in range(-cx, cx + 1):
                    yy = y + m
                    xx = x + n

                    if yy < 0:
                        yy = -yy
                    elif yy >= h:
                        yy = 2 * h - yy - 2

                    if xx < 0:
                        xx = -xx
                    elif xx >= w:
                        xx = 2 * w - xx - 2

                    s += img[yy, xx] * kernel[m + cy, n + cx]

            new_img[y, x] = s

    return new_img


def saveas_pgm(img: np.ndarray, img_name: str) -> None:
    """
    Save a grayscale image as a PGM file.
    """
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    cv2.imwrite("kadai0501/" + img_name + ".png", img)

    with open(f"kadai0501/{img_name}.pgm", "w") as f:
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
    if not os.path.exists("kadai0501"):
        os.makedirs("kadai0501")

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
    plt.savefig(f"kadai0501/{output_name}.png")
    plt.show()


if __name__ == "__main__":
    IMAGE_DIR = Path("../images")
    img_g = cv2.imread(str(IMAGE_DIR / "G.pgm"), cv2.IMREAD_GRAYSCALE)
    img_h = cv2.imread(str(IMAGE_DIR / "H.pgm"), cv2.IMREAD_GRAYSCALE)

    # sharpening
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_g_sharpen = filtering(img_g, kernel_sharpen)
    img_h_sharpen = filtering(img_h, kernel_sharpen)
    plt_compare(
        [img_g, img_h],
        [img_g_sharpen, img_h_sharpen],
        ["G_Sharpening", "H_Sharpening"],
        "普通の鮮鋭化",
    )
    visualize_overshoot_undershoot(img_g_sharpen, "G_sharpening")

    kernel_sharpen_all = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_g_sharpen_all = filtering(img_g, kernel_sharpen_all)
    img_h_sharpen_all = filtering(img_h, kernel_sharpen_all)
    plt_compare(
        [img_g, img_h],
        [img_g_sharpen_all, img_h_sharpen_all],
        ["G_Sharpening_All", "H_Sharpening_All"],
        "全方向の鮮鋭化",
    )
