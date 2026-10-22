"""
Generate diverse photographic-style synthetic images for Parallel benchmarking and testing.
"""
from PIL import Image, ImageDraw
import numpy as np
import os

def create_sample_images(samples_dir="samples"):
    os.makedirs(samples_dir, exist_ok=True)
    
    # 1. Sample Landscape / Gradient (1080p: 1920x1080)
    w, h = 1920, 1080
    x = np.linspace(-3, 3, w)
    y = np.linspace(-3, 3, h)
    xx, yy = np.meshgrid(x, y)
    
    r = np.sin(xx**2 + yy**2)
    g = np.cos(xx * 2 + yy)
    b = np.sin(xx * yy * 0.5)
    
    rgb = np.stack([
        ((r + 1) * 127.5).astype(np.uint8),
        ((g + 1) * 127.5).astype(np.uint8),
        ((b + 1) * 127.5).astype(np.uint8),
    ], axis=-1)
    
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    # Draw geometric high-frequency patterns for edge detection testing
    for i in range(100, 1800, 150):
        draw.rectangle([i, 200, i + 80, 800], outline=(255, 255, 255), width=4)
        draw.ellipse([i + 20, 300, i + 60, 500], fill=(240, 200, 50), outline=(20, 20, 20), width=2)
    
    sample_path = os.path.join(samples_dir, "sample.png")
    img.save(sample_path)
    print(f"Generated {sample_path} ({w}x{h})")

    # 2. Geometric Checkerboard (512x512)
    checker = np.indices((512, 512)).sum(axis=0) % 64 < 32
    checker_img = Image.fromarray((checker * 255).astype(np.uint8))
    checker_path = os.path.join(samples_dir, "checkerboard.png")
    checker_img.save(checker_path)
    print(f"Generated {checker_path} (512x512)")

if __name__ == "__main__":
    create_sample_images()
