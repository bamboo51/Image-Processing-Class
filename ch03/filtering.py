from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def saveas_pgm(img: np.ndarray, img_name: str) -> None:
    """
    Save a grayscale image as a PGM file.
    """
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    cv2.imwrite(img_name + ".png", img)

    with open(f"{img_name}.pgm", "w") as f:
        f.write("P2\n")
        f.write(f"{img.shape[1]} {img.shape[0]}\n")
        f.write("255\n")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                f.write(f"{img[i, j]} ")
            f.write("\n")


def filtering(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a 2D linear filter to a grayscale image.
    Padding is done with reflection (OpenCV default)
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


def compare_multiple(
    img_g: np.ndarray,
    img_h: np.ndarray,
    results_g: list,
    results_h: list,
    titles: list,
    save_name: str,
) -> None:

    # ---------- Plot G ----------
    fig, ax = plt.subplots(2, 5, figsize=(15, 6))

    images = [img_g] + results_g
    titles_all = ["Original"] + titles

    for i in range(10):
        r = i // 5
        c = i % 5

        if i < len(images):
            if i == 0:
                ax[r, c].imshow(images[i], cmap="gray", vmin=0, vmax=255)
            else:
                ax[r, c].imshow(images[i], cmap="seismic", vmin=-1, vmax=1)

            ax[r, c].set_title(f"G ({titles_all[i]})", pad=6)
        else:
            ax[r, c].axis("off")

        ax[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_name}_G.png")
    plt.show()

    # ---------- Plot H ----------
    fig, ax = plt.subplots(2, 5, figsize=(15, 6))

    images = [img_h] + results_h

    for i in range(10):
        r = i // 5
        c = i % 5

        if i < len(images):
            if i == 0:
                ax[r, c].imshow(images[i], cmap="gray", vmin=0, vmax=255)
            else:
                ax[r, c].imshow(images[i], cmap="seismic", vmin=-1, vmax=1)

            ax[r, c].set_title(f"H ({titles_all[i]})", pad=6)
        else:
            ax[r, c].axis("off")

        ax[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_name}_H.png")
    plt.show()


def normalize_signed(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)

    max_val = np.max(np.abs(img))
    if max_val == 0:
        return np.zeros_like(img)

    return img / max_val  # range: [-1, 1]


def to_grayscale(img: np.ndarray) -> np.ndarray:
    img = np.abs(img.astype(np.float32))
    max_val = img.max()

    if max_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    img = 255.0 * img / max_val
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def save_filter_comparison(
    img_g: np.ndarray,
    out_g: np.ndarray,
    img_h: np.ndarray,
    out_h: np.ndarray,
    filter_name: str,
    save_name: str,
) -> None:
    """
    Save one figure per filter:
    Row 1: G original, G grayscale filtered, G seismic filtered
    Row 2: H original, H grayscale filtered, H seismic filtered
    """
    out_g_gray = to_grayscale(out_g)
    out_h_gray = to_grayscale(out_h)

    out_g_seismic = normalize_signed(out_g)
    out_h_seismic = normalize_signed(out_h)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # G row
    ax[0, 0].imshow(img_g, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("G (Original)")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(out_g_gray, cmap="gray", vmin=0, vmax=255)
    ax[0, 1].set_title(f"G ({filter_name}, grayscale)")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(out_g_seismic, cmap="seismic", vmin=-1, vmax=1)
    ax[0, 2].set_title(f"G ({filter_name}, seismic)")
    ax[0, 2].axis("off")

    # H row
    ax[1, 0].imshow(img_h, cmap="gray", vmin=0, vmax=255)
    ax[1, 0].set_title("H (Original)")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(out_h_gray, cmap="gray", vmin=0, vmax=255)
    ax[1, 1].set_title(f"H ({filter_name}, grayscale)")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(out_h_seismic, cmap="seismic", vmin=-1, vmax=1)
    ax[1, 2].set_title(f"H ({filter_name}, seismic)")
    ax[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_name}_{filter_name}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # save grayscale filtered images as pgm/png too
    saveas_pgm(out_g_gray, f"G_{filter_name}")
    saveas_pgm(out_h_gray, f"H_{filter_name}")


if __name__ == "__main__":
    IMAGE_DIR = Path("../../images")
    img_g = cv2.imread(str(IMAGE_DIR / "G.pgm"), cv2.IMREAD_GRAYSCALE)
    img_h = cv2.imread(str(IMAGE_DIR / "H.pgm"), cv2.IMREAD_GRAYSCALE)

    # 縦微分フィルター
    kernel_diff_v = np.zeros((3, 3))
    kernel_diff_v[0, 1] = -0.5
    kernel_diff_v[2, 1] = 0.5

    # 横微分フィルター
    kernel_diff_h = np.zeros((3, 3))
    kernel_diff_h[1, 0] = -0.5
    kernel_diff_h[1, 2] = 0.5

    # prewittフィルター
    kernel_prewitt_v = np.array(
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1],
        ]
    )
    kernel_prewitt_h = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ]
    )

    # sobelフィルター
    kernel_sobel_v = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ]
    )
    kernel_sobel_h = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]
    )

    # 縦2次微分フィルター
    kernel_diff2_v = np.zeros((3, 3))
    kernel_diff2_v[0, 1] = 1
    kernel_diff2_v[1, 1] = -2
    kernel_diff2_v[2, 1] = 1
    # 横2次微分フィルター
    kernel_diff2_h = np.zeros((3, 3))
    kernel_diff2_h[1, 0] = 1
    kernel_diff2_h[1, 1] = -2
    kernel_diff2_h[1, 2] = 1

    # ラプラシアンフィルター
    kernel_laplacian = np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ]
    )

    kernels = [
        ("Diff_V", kernel_diff_v.astype(np.float32)),
        ("Diff_H", kernel_diff_h.astype(np.float32)),
        ("Prewitt_V", kernel_prewitt_v.astype(np.float32)),
        ("Prewitt_H", kernel_prewitt_h.astype(np.float32)),
        ("Sobel_V", kernel_sobel_v.astype(np.float32)),
        ("Sobel_H", kernel_sobel_h.astype(np.float32)),
        ("Second_V", kernel_diff2_v.astype(np.float32)),
        ("Second_H", kernel_diff2_h.astype(np.float32)),
        ("Laplacian", kernel_laplacian.astype(np.float32)),
    ]

    results_g = []
    results_h = []
    titles = []

    for name, kernel in kernels:
        out_g = filtering(img_g, kernel)
        out_h = filtering(img_h, kernel)

        # for display with seismic colormap
        out_g_disp = normalize_signed(out_g)  # [-1, 1]
        out_h_disp = normalize_signed(out_h)  # [-1, 1]

        results_g.append(out_g_disp)
        results_h.append(out_h_disp)
        titles.append(name)

        # for saving as grayscale
        out_g_gray = to_grayscale(out_g)
        out_h_gray = to_grayscale(out_h)

        saveas_pgm(out_g_gray, f"G_{name}")
        saveas_pgm(out_h_gray, f"H_{name}")

    compare_multiple(img_g, img_h, results_g, results_h, titles, "comparison")

    for name, kernel in kernels:
        out_g = filtering(img_g, kernel)
        out_h = filtering(img_h, kernel)

        save_filter_comparison(
            img_g=img_g,
            out_g=out_g,
            img_h=img_h,
            out_h=out_h,
            filter_name=name,
            save_name="comparison",
        )
