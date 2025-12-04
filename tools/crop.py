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
    img_np = np.array(img)[:, :720]

    print(f"Image shape: {img_np.shape}")
    # save image
    out_path = args.image + ".cropped.png"
    Image.fromarray(img_np).save(out_path)
    print(f"Cropped image saved to: {out_path}")

if __name__ == "__main__":
    main()
