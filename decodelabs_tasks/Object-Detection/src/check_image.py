
from PIL import Image
import numpy as np

img = Image.open("sample_images/download (1).webp")
print(f"Pillow: size={img.size}, mode={img.mode}")


img.convert("RGB").save("sample_images/test_converted.jpg")
print("Saved test_converted.jpg")


arr = np.array(img)
print(f"Pixel stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
