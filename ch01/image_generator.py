import numpy as np

image = np.zeros((256, 256), dtype=np.uint8)
KADAI = 12

period = 256 // 8 
half = period // 2

for i in range(256):
    for j in range(256):
        sq_i = 1 if (i // half) % 2 == 0 else -1
        sq_j = 1 if (j // half) % 2 == 0 else -1
        image[i, j] = 127 + 30*sq_i + 30*sq_j


with open(f"kadai{KADAI}.pgm", "w") as f:
    f.write("P2\n") 
    f.write(f"# 課題1.0{KADAI}\n") 
    f.write("256 256\n")
    f.write("255\n") 
    f.write("#%%#\n")
    
    for i in range(256):
        for j in range(256):
            f.write(f"{image[i, j]} ")
        f.write("\n")
