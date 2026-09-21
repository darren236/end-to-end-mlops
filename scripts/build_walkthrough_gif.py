"""Build an optimized animated walkthrough from captured UI frames."""

import argparse
from pathlib import Path

from PIL import Image

FRAME_NAMES = (
    "ui-overview.png",
    "ui-training.png",
    "ui-evaluation.png",
    "ui-prediction.png",
    "ui-drift.png",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("docs/assets"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--duration", type=int, default=1800, help="Milliseconds per frame")
    args = parser.parse_args()

    frames: list[Image.Image] = []
    for name in FRAME_NAMES:
        with Image.open(args.assets / name) as source:
            height = round(source.height * args.width / source.width)
            resized = source.convert("RGB").resize((args.width, height), Image.Resampling.LANCZOS)
            frames.append(
                resized.quantize(
                    colors=160,
                    method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.FLOYDSTEINBERG,
                )
            )

    output = args.assets / "mlops-walkthrough.gif"
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Wrote {output} with {len(frames)} frames")


if __name__ == "__main__":
    main()
