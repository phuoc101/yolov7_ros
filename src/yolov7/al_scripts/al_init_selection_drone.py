from loguru import logger
import shutil
import os
import sys
import argparse
import math
import random
from glob import glob


# change to environment variable pointing to dataset absolute directory
DATASETS_DIR = os.environ["DATASETS_CV_DIR"]
DATASET_NAME = "picam_data/Full_train_test_val_5percent"


def choose_images(files, num_chosen, chosen):
    files_to_choose = [f for f in files if f not in chosen]
    chosen_train = random.sample(files_to_choose, num_chosen)
    chosen += chosen_train

    files_to_choose = [f for f in files if f not in chosen]
    chosen_val = random.sample(files_to_choose, num_chosen)
    chosen += chosen_val
    return chosen_train, chosen_val


def move_chosen(chosen_train, train_dir, chosen_val=[], val_dir=[], addition=False):
    if not addition:
        for file in chosen_train:
            basename = os.path.basename(file)
            new_file = os.path.join(train_dir, basename)
            shutil.move(file, new_file)
            logger.info(f"{file} -> {new_file}")
        for file in chosen_val:
            basename = os.path.basename(file)
            new_file = os.path.join(val_dir, basename)
            shutil.move(file, new_file)
            logger.info(f"{file} -> {new_file}")
    else:
        for file in chosen_train:
            basename = os.path.basename(file)
            new_file = os.path.join(train_dir, basename)
            shutil.move(file, new_file)
            logger.info(f"{file} -> {new_file}")


def main(opts):
    dir, sets, ext = opts.dir, opts.sets, opts.extension
    percent = opts.percentage
    train_dir = os.path.join(dir, "train/images")
    val_dir = os.path.join(dir, "val/images")
    test_dir = os.path.join(dir, "test/images")
    addition_dir = os.path.join(dir, "addition/images")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(addition_dir, exist_ok=True)
    # if len(os.listdir(val_dir)) != 0 or len(os.listdir(train_dir)):
    #     logger.error(
    #         "Validation and Training directories need to be both empty to initialize!"
    #     )
    #     sys.exit(1)
    # else:
    #     logger.info(
    #         "Both training and validation directories are empty, start initializing"
    #     )
    chosen = []
    all_files = os.listdir(test_dir)
    num_of_files = len(all_files)
    num_chosen = math.floor((percent / 100) * num_of_files / len(sets))
    logger.debug(f"Choosing {num_chosen} images from each set")
    for set in sets:
        files = glob(test_dir + f"/*{set}*{ext}")
        logger.info(f"Found {len(files)} images in {set}")
        chosen_train, chosen_val = choose_images(
            files=files, num_chosen=num_chosen, chosen=chosen
        )
        logger.debug(f"Chosen for training: {chosen_train}")
        logger.debug(f"Chosen for validation: {chosen_val}")
        if not opts.addition:
            move_chosen(chosen_train, train_dir, chosen_val, val_dir)
        else:
            move_chosen(chosen_train, addition_dir, addition=opts.addition)
            # comply with CVAT
            with open(
                os.path.abspath(os.path.join(addition_dir, "../train.txt")), "a+"
            ) as f:
                lines = [
                    "data/obj_train_data/" + os.path.basename(file) + "\n"
                    for file in chosen_train
                ]
                f.writelines(lines)


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
        default=["no_drone", "has_drone", "has_close_person", "has_close_drone"],
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
