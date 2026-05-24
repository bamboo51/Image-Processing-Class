import os

import cv2

# Create output directory if it doesn't exist
if not os.path.exists("../ascii"):
    os.makedirs("../ascii")

for image in os.listdir("../images"):
    # Construct full path
    input_path = os.path.join("../images", image)

    # Skip directories if any
    if not os.path.isfile(input_path):
        continue

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Proceed only if the image was loaded successfully
    if img is not None:
        output_filename = image.replace(".pgm", "_ascii.pgm")
        output_path = os.path.join("../ascii", output_filename)

        with open(output_path, "w") as f:
            f.write("P2\n")
            # PGM Comments must start with #
            f.write(f"# image {image}\n")
            f.write(f"{img.shape[1]} {img.shape[0]}\n")
            f.write("255\n")

            for row in img:
                # Convert the row of integers to a string of space-separated values
                # map(str, row) converts [0, 255, ...] to ['0', '255', ...]
                # " ".join(...) creates "0 255 ..."
                line = " ".join(map(str, row))
                f.write(line + "\n")

print("Conversion complete.")
