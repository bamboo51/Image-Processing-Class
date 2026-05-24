from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def saveas_pgm(img: np.ndarray, img_name: str) -> None:
    """
    Save a grayscale image as a PGM file.
    """
    cv2.imwrite(img_name + ".png", img)
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    with open(f"{img_name}.pgm", "w") as f:
        f.write(f"P2\n")
        f.write(f"{img.shape[1]} {img.shape[0]}\n")

        f.write(f"255\n")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                f.write(f"{img[i, j]} ")
            f.write("\n")


def plot_hist(img: np.ndarray, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Compute histogram and flatten it
    hist = np.zeros(256, dtype=np.int32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            hist[img[y, x]] += 1

    # Image display
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title(title)
    ax[0].axis("off")

    # Histogram plot
    ax[1].bar(range(256), hist, width=2)
    ax[1].set_title("Intensity Histogram")
    ax[1].set_xlabel("Pixel Value (0–255)")
    ax[1].set_ylabel("Frequency")
    ax[1].set_xlim([-2, 257])
    ax[1].grid(True)

    # Improve layout
    plt.tight_layout()

    # Safer filename
    filename = title.lower().replace(" ", "_")
    plt.savefig(f"./hist_{filename}.png", dpi=150)

    plt.close()


def michelson_contrast(img: np.ndarray) -> float:
    img = img.astype(np.float32)
    img_max = img.max()
    img_min = img.min()
    denom = img_max + img_min
    if denom == 0:
        return 0.0
    return (img_max - img_min) / denom


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

    return new_img.astype(np.uint8)


def insertion_sort(arr: np.ndarray) -> np.ndarray:
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def median_filtering(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    img = img.astype(np.float32)
    h, w = img.shape
    r = ksize // 2

    new_img = np.zeros_like(img, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            values = []
            for m in range(-r, r + 1):
                for n in range(-r, r + 1):
                    # 範囲外のピクセルは反転させる
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

                    values.append(img[yy, xx])
            values = insertion_sort(np.array(values))
            new_img[y, x] = values[len(values) // 2]
    return new_img.astype(np.uint8)


def compare_single(before_img, after_img, save_name):
    before = before_img.astype(np.float32)
    after = after_img.astype(np.float32)

    diff = after - before

    max_val = np.max(np.abs(diff))
    if max_val > 0:
        diff_disp = diff / max_val
    else:
        diff_disp = diff

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(before_img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Before")
    ax[0].axis("off")

    ax[1].imshow(after_img, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("After")
    ax[1].axis("off")

    ax[2].imshow(diff_disp, cmap="seismic", vmin=-1, vmax=1)
    ax[2].set_title("Diff (signed)")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_name + "_diff.png")
    plt.show()

    return diff


def compare_multiple(
    img_k: np.ndarray,
    img_l: np.ndarray,
    results_k: list,
    results_l: list,
    titles: list,
    save_name: str,
) -> None:
    """
    Row 1: K images
    Row 2: K histograms
    Row 3: L images
    Row 4: L histograms
    """

    n = len(results_k) + 1  # +1 for original
    fig, ax = plt.subplots(
        4, n, figsize=(4.2 * n, 12), gridspec_kw={"height_ratios": [1, 0.8, 1, 0.8]}
    )

    def hist(img: np.ndarray) -> np.ndarray:
        return cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    # Precompute histograms
    imgs_k_all = [img_k] + list(results_k)
    imgs_l_all = [img_l] + list(results_l)
    titles_k_all = ["Original"] + list(titles)
    titles_l_all = ["Original"] + list(titles)

    hists_k_all = [hist(img) for img in imgs_k_all]
    hists_l_all = [hist(img) for img in imgs_l_all]

    # Shared y-limits for fair comparison
    ymax_k = max(h.max() for h in hists_k_all) * 1.05
    ymax_l = max(h.max() for h in hists_l_all) * 1.05

    # --- Row 1: K images ---
    for i, (img, title) in enumerate(zip(imgs_k_all, titles_k_all)):
        ax[0, i].imshow(img, cmap="gray", vmin=0, vmax=255)
        ax[0, i].set_title(f"K: {title}", fontsize=11)
        ax[0, i].axis("off")

    # --- Row 2: K histograms ---
    for i, (h, title) in enumerate(zip(hists_k_all, titles_k_all)):
        ax[1, i].bar(np.arange(256), h, width=1.0, color="gray")
        ax[1, i].set_xlim(-2, 257)
        ax[1, i].set_ylim(0, ymax_k)
        ax[1, i].set_xticks([0, 64, 128, 192, 255])
        ax[1, i].grid(True, axis="y", alpha=0.3)
        if i == 0:
            ax[1, i].set_ylabel("Freq")
        else:
            ax[1, i].set_yticklabels([])
        ax[1, i].set_xlabel("Intensity", fontsize=9)

    # --- Row 3: L images ---
    for i, (img, title) in enumerate(zip(imgs_l_all, titles_l_all)):
        ax[2, i].imshow(img, cmap="gray", vmin=0, vmax=255)
        ax[2, i].set_title(f"L: {title}", fontsize=11)
        ax[2, i].axis("off")

    # --- Row 4: L histograms ---
    for i, (h, title) in enumerate(zip(hists_l_all, titles_l_all)):
        ax[3, i].bar(np.arange(256), h, width=1.0, color="gray")
        ax[3, i].set_xlim(-2, 257)
        ax[3, i].set_ylim(0, ymax_l)
        ax[3, i].set_xticks([0, 64, 128, 192, 255])
        ax[3, i].grid(True, axis="y", alpha=0.3)
        if i == 0:
            ax[3, i].set_ylabel("Freq")
        else:
            ax[3, i].set_yticklabels([])
        ax[3, i].set_xlabel("Intensity", fontsize=9)

    plt.tight_layout(pad=1.5)
    plt.savefig(f"{save_name}.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    IMAGE_DIR = Path("../../images")
    img_a = cv2.imread(str(IMAGE_DIR / "A.pgm"), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(IMAGE_DIR / "B.pgm"), cv2.IMREAD_GRAYSCALE)

    # Plot histograms
    plot_hist(img_a, "Image A")
    plot_hist(img_b, "Image B")

    # Invert
    img_a_inv = 255 - img_a
    img_b_inv = 255 - img_b
    plot_hist(img_a_inv, "Image A (Inverted)")
    plot_hist(img_b_inv, "Image B (Inverted)")
    saveas_pgm(img_a_inv, "A_inv")
    saveas_pgm(img_b_inv, "B_inv")

    # calculate the contrast of each image
    print(f"Contrast of Image A: {michelson_contrast(img_a)}")
    print(f"Contrast of Image B: {michelson_contrast(img_b)}")
    print(f"Contrast of Image A (Inverted): {michelson_contrast(img_a_inv)}")
    print(f"Contrast of Image B (Inverted): {michelson_contrast(img_b_inv)}")

    img_k = cv2.imread(str(IMAGE_DIR / "K.pgm"), cv2.IMREAD_GRAYSCALE)
    img_l = cv2.imread(str(IMAGE_DIR / "L.pgm"), cv2.IMREAD_GRAYSCALE)

    filter_average_5 = 1 / 25 * np.ones((5, 5))
    img_k_5 = filtering(img_k, filter_average_5)
    img_l_5 = filtering(img_l, filter_average_5)
    saveas_pgm(img_k_5, "K_5")
    saveas_pgm(img_l_5, "L_5")
    compare_single(img_k, img_k_5, "K_5")
    compare_single(img_l, img_l_5, "L_5")

    filter_weighted_average_5 = (
        np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ]
        )
        / 256
    )
    img_k_5_weighted = filtering(img_k, filter_weighted_average_5)
    img_l_5_weighted = filtering(img_l, filter_weighted_average_5)
    saveas_pgm(img_k_5_weighted, "K_5_weighted")
    saveas_pgm(img_l_5_weighted, "L_5_weighted")
    compare_single(img_k, img_k_5_weighted, "K_5_weighted")
    compare_single(img_l, img_l_5_weighted, "L_5_weighted")

    filter_direction_9 = np.zeros((9, 9))
    for i in range(filter_direction_9.shape[0]):
        filter_direction_9[i, i] = 1 / 9

    img_k_9 = filtering(img_k, filter_direction_9)
    img_l_9 = filtering(img_l, filter_direction_9)
    saveas_pgm(img_k_9, "K_9")
    saveas_pgm(img_l_9, "L_9")
    compare_single(img_k, img_k_9, "K_9")
    compare_single(img_l, img_l_9, "L_9")

    compare_multiple(
        img_k,
        img_l,
        [img_k_5, img_k_5_weighted, img_k_9],
        [img_l_5, img_l_5_weighted, img_l_9],
        ["Avg 5x5", "Weighted 5x5", "Directional 9x9"],
        "filters",
    )

    img_k_med = median_filtering(img_k, 3)
    img_l_med = median_filtering(img_l, 3)
    saveas_pgm(img_k_med, "K_med")
    saveas_pgm(img_l_med, "L_med")
    compare_single(img_k, img_k_med, "K_med")
    compare_single(img_l, img_l_med, "L_med")
    plot_hist(img_k, "K")
    plot_hist(img_l, "L")
    plot_hist(img_k_med, "K_med_hist")
    plot_hist(img_l_med, "L_med_hist")
