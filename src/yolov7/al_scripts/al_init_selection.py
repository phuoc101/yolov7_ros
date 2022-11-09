from loguru import logger
import os
import sys
import argparse
import math
from glob import glob


# change to environment variable pointing to dataset absolute directory
DATASETS_DIR = os.environ["DATASETS_CV_DIR"]
DATASET_NAME = "picam_data/Full_train_test_val_5"


def choose_images(files, percent, num_of_imgs):
    files_total = len(files)
    if percent > 0:
        logger.debug(
            f"Choosing {percent}% of images in set, which is {files_total * percent / 100} imgs"
        )
        percent /= 100
        step = round(1 / percent)
        return files[0:files_total:step], files[1:files_total:step]
    elif num_of_imgs > 0:
        logger.debug(f"Choosing {num_of_imgs} images in set")
        step = files_total // num_of_imgs
        return (
            files[0 : num_of_imgs * step : step],
            files[1 : num_of_imgs * step + 1 : step],
        )
    else:
        logger.error("Need to choose percentage or num of imgs to take")
        sys.exit(1)


def move_to_train_val(chosen_train, train_dir, chosen_val, val_dir):
    for file in chosen_train:
        basename = os.path.basename(file)
        new_file = os.path.join(train_dir, basename)
        os.replace(file, new_file)
        logger.info(f"{file} -> {new_file}")
    for file in chosen_val:
        basename = os.path.basename(file)
        new_file = os.path.join(val_dir, basename)
        os.replace(file, new_file)
        logger.info(f"{file} -> {new_file}")


def main(opts):
    train_dir = os.path.join(opts.dir, "train/images")
    val_dir = os.path.join(opts.dir, "val/images")
    test_dir = os.path.join(opts.dir, "test/images")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    if len(os.listdir(val_dir)) != 0 or len(os.listdir(train_dir)):
        logger.error(
            "Validation and Training directories need to be both empty to initialize!"
        )
        sys.exit(1)
    else:
        logger.info(
            "Both training and validation directories are empty, start initializing"
        )
    sets = opts.sets
    ext = opts.extension
    percent = opts.percentage
    num_of_imgs = opts.num_of_imgs
    chosen_imgs = []
    for set in sets:
        files = glob(test_dir + f"/{set}*{ext}")
        logger.info(f"Found {len(files)} images in {set}")
        chosen_train, chosen_val = choose_images(
            files=files, percent=percent, num_of_imgs=num_of_imgs
        )
        logger.debug(f"Chosen for training: {chosen_train}")
        logger.debug(f"Chosen for validation: {chosen_val}")
        move_to_train_val(
            chosen_train=chosen_train,
            train_dir=train_dir,
            chosen_val=chosen_val,
            val_dir=val_dir,
        )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
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
            "Hevolinna1_no_drone",
            "Hevolinna1_drone",
            "Hevolinna2_no_drone",
            "Hevolinna2_drone",
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
        "-p",
        "--percentage",
        type=float,
        default=0,
        help="Percentage of image to take from each set (use either this or number of images, will prioritize this if both are selected)",
    )
    parser.add_argument(
        "-n",
        "--num-of-imgs",
        type=int,
        default=0,
        help="Number of images to take from each set (use either this or percentage)",
    )
    opts = parser.parse_args()
    main(opts=opts)
