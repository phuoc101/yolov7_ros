import os
import sys
import shutil
import argparse
from loguru import logger

DATASET_DIR = os.getenv("DATASETS_CV_DIR")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(opts):
    addition_set_img = os.path.join(opts.dataset, "addition/images")
    addition_set_prelbl = os.path.join(opts.dataset, "addition/pre_labels")
    os.makedirs(addition_set_img, exist_ok=True)
    os.makedirs(addition_set_prelbl, exist_ok=True)
    prelbl_dir = opts.prelabels
    with open(opts.additional_txt, "r") as f:
        frames = f.readlines()
        for frame in frames:
            frame = frame.replace("\n", "")
            frame_name = os.path.basename(frame)
            frame_name_txt = frame_name.replace(opts.extension, ".txt")
            prelabel = os.path.join(prelbl_dir, frame_name_txt)
            logger.debug(f"Frame read: {frame}")
            # move selected frames to addition
            dst_file = os.path.abspath(os.path.join(addition_set_img, frame_name))
            shutil.copy(src=frame, dst=dst_file)
            logger.info(f"Moved file {frame} -> {dst_file}")
            # Move prelabel to addition
            dst_file_prelbl = os.path.abspath(
                os.path.join(addition_set_prelbl, frame_name_txt)
            )
            shutil.copy(src=prelabel, dst=dst_file_prelbl)
            logger.info(f"Moved file {prelabel} -> {dst_file_prelbl}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default=os.path.join(DATASET_DIR, "picam_data/Full_train_test_val_5"),
        help="Dataset dir",
    )
    parser.add_argument(
        "-l",
        "--prelabels",
        required=True,
        help="Pre-labels from previous model",
    )
    parser.add_argument(
        "-x",
        "--extension",
        default=".png",
        help="img extension",
    )
    parser.add_argument(
        "-a",
        "--additional-txt",
        default=os.path.join(CURRENT_DIR, "addition.txt"),
        help="Path to additional files",
    )
    opts = parser.parse_args()
    main(opts)
