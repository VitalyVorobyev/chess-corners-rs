import argparse
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Overlay ChESS corners on an image.")
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        help="Input image path (optional when --json is provided and contains an 'image' field).",
    )
    args = parser.parse_args()

    img = Image.open(args.image).convert("L")
    snap_idx = 5
    col_lo, col_hi = snap_idx * 720, (snap_idx + 1) * 720
    img_np = np.array(img)[:, col_lo:col_hi]

    print(f"Image shape: {img_np.shape}")
    # save image
    out_path = args.image + f".cropped{snap_idx}.png"
    Image.fromarray(img_np).save(out_path)
    print(f"Cropped image saved to: {out_path}")

if __name__ == "__main__":
    main()
