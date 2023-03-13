import random
import argparse


def main(opts):
    to_label = []
    with open(opts.unlabeled_set, "r") as f:
        files = [fi for fi in f.readlines()]
        to_label = random.sample(files, opts.num_of_imgs)
        unlabeled = [fi for fi in files if fi not in to_label]
        f.close()
    with open("unlabeled_random.txt", "w+") as u:
        u.writelines(unlabeled)
        u.close()
    with open("to_label_random.txt", "w+") as t:
        t.writelines(to_label)
        t.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unlabeled-set",
        "-u",
        default="../../yolov7/datasets/picam_data/Full_dataset_AL/CVAT_upload/unlabeled.txt",
        help="relative path to unlabeled images",
    )
    parser.add_argument(
        "--num-of-imgs",
        "-n",
        type=int,
        default=100,
        help="num of images to select",
    )
    opts = parser.parse_args()
    main(opts)
