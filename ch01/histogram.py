import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and display the histogram of a PGM image.")
    parser.add_argument("image_path", help="Path to the PGM image file")
    args = parser.parse_args()

    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not read the image file.")
        exit(1)

    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.flatten()
    print("Pixel Value : Frequency")
    for i in range(256):
        if histogram[i] > 0:
            print(f"{i} : {histogram[i]}")

    plt.figure(figsize=(6, 6))
    plt.bar(range(256), histogram, width=2.0)
    plt.xlim([-2, 258])
    plt.xticks(range(0, 256, 25))
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"{args.image_path}_histogram.png")
    plt.show()

    cv2.imwrite(f"{args.image_path}_image.png", image)