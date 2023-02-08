import sys
import argparse
import shutil
import os
from pathlib import Path
from loguru import logger


def main(args):
    idx = 0
    output_path = Path(args.output)
    os.makedirs(args.output, exist_ok=True)
    if args.copy:
        for set in args.datasets:
            data_path = Path(set)
            imgs = sorted(data_path.rglob("*.png"))
            for img in imgs:
                filename = Path.joinpath(
                    output_path, "{:06d}_{}.png".format(idx, data_path.name)
                )
                logger.info("Copying {} to {}".format(img, filename))
                shutil.copy(img, filename)
                idx += 1
    layout_path = output_path.joinpath("ImageSets")
    imgs_concat = sorted([im for im in output_path.rglob("*.png")])
    logger.debug(f"Concat imgs : {imgs_concat}")
    with open(layout_path.joinpath("unlabeled.txt"), "w+") as f:
        for img in imgs_concat:
            f.write(f"{img.name.replace('.png', '')}\n")
        logger.info(f"{len(imgs_concat)} imgs written to unlabeled dataset")
        f.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", help="Directories to concatenate")
    parser.add_argument("--output", default="./out", type=str, help="Output directory")
    parser.add_argument("--copy", action="store_true", help="Copy images to new dir")
    opts = parser.parse_args()
    main(opts)
