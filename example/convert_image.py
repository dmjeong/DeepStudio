"""PNG/JPEG -> packed 8-bit BGR/GRAY for the C# example. No resize/normalization."""
import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gray", action="store_true")
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = source.convert("L" if args.gray else "RGB")
        args.output.write_bytes(image.tobytes() if args.gray else image.tobytes("raw", "BGR"))
        print(f"width={image.width} height={image.height} channels={1 if args.gray else 3}")


if __name__ == "__main__":
    main()
