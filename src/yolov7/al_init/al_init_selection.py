from loguru import logger
import os
import sys
import argparse
import math
import random
from glob import glob


# change to environment variable pointing to dataset absolute directory
DATASETS_DIR = os.environ["DATASETS_CV_DIR"]
DATASET_NAME = "picam_data/Full_dataset_AL"


def choose_images(
    files, num_chosen, chosen, ext, train_file, unlabeled_file, train_abs
):
    files_to_choose = [f for f in files if f not in chosen]
    chosen_train = random.sample(files_to_choose, num_chosen)
    logger.debug(f"Chosen for training: {chosen_train}")
    chosen += chosen_train
    for file in chosen_train:
        basename = os.path.basename(file).replace(ext, "")
        with open(unlabeled_file, "r+") as f_unlabeled:
            lines = f_unlabeled.readlines()
            f_unlabeled.seek(0)
            for line in lines:
                if line != basename + "\n":
                    f_unlabeled.write(line)
                else:
                    logger.debug(f"Removed {basename} from {unlabeled_file}")
            f_unlabeled.truncate()
        with open(train_file, "a+") as f:
            f.write(f"{basename}\n")
            logger.info(f"Wrote {basename} to {train_file}")
        with open(train_abs, "a+") as f:
            abs_file = os.path.abspath(file)
            f.write(f"{abs_file}\n")
            logger.info(f"Wrote {abs_file} to {train_abs}")

    return chosen_train


def main(opts):
    dir, sets, ext = opts.dir, opts.sets, opts.extension
    percent = opts.percentage
    train_file = os.path.join(dir, "ImageSets/train.txt")
    train_abs_file = os.path.join(dir, "CVAT_upload/train.txt")
    unlabeled_file = os.path.join(dir, "ImageSets/unlabeled.txt")
    img_dir = os.path.join(dir, "Images")
    chosen = []
    # all_files = os.listdir(img_dir)
    for set in sets:
        files = glob(img_dir + f"/*{set}*{ext}")
        num_of_files = len(files)
        logger.info(f"Found {num_of_files} images in {set}")
        num_chosen = math.floor((percent / 100) * num_of_files)
        logger.debug(f"Choosing {num_chosen} images from set {set}")
        choose_images(
            files=files,
            num_chosen=num_chosen,
            chosen=chosen,
            ext=ext,
            train_file=train_file,
            unlabeled_file=unlabeled_file,
            train_abs=train_abs_file
        )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    parser = argparse.ArgumentParser()
    # Dataset should initially have train, test, val dirs, all images in test directory
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default=os.path.join(DATASETS_DIR, DATASET_NAME),
        help="Dataset directory",
    )
    # In test dir, images are named: {dataset}*.png
    parser.add_argument(
        "-s",
        "--sets",
        nargs="+",
        default=[
            "Tierankatu_nodrone",
            "Tierankatu_hasdroneorange",
            "Tierankatu_hasdronetello",
            "closeperson",
            "Hevolinna_hasdrone",
            "Hevolinna_nodrone",
            "Narnia",
        ],
        help="List of datasets",
    )
    parser.add_argument(
        "-x",
        "--extension",
        type=str,
        default=".png",
        help="File extension",
    )
    parser.add_argument(
        "-a",
        "--addition",
        action="store_true",
        help="File extension",
    )
    parser.add_argument(
        "-p",
        "--percentage",
        type=float,
        default=0,
        help=(
            "Percentage of image to take from dataset ",
            "(use either this or number of images, ",
            "will prioritize this if both are selected)",
        ),
    )
    opts = parser.parse_args()
    main(opts=opts)
