import sys
import os
import os.path as osp
import argparse
from loguru import logger

OUTPUT_FILE = "addition.txt"
DATASET_DIR = os.getenv("DATASETS_CV_DIR")
NUM_OF_CLASSES = 2  # drone and person
FALSE_PRED = -1
TRUE_PRED = 1


def get_label_dict(labels, dir, set_name):
    preds_dict = dict()
    label_dir = os.path.join(dir, "labels")
    for file in os.listdir(label_dir):
        if set_name in file:
            with open(osp.join(label_dir, file), "r") as f:
                file_name = file.split(".")[0]
                file_pred = dict()
                for lbl in labels:
                    file_pred[lbl] = []
                for line in f.readlines():
                    lbl, _, _, _, _, conf = line.split()
                    lbl = int(lbl)
                    conf = float(conf)
                    file_pred[lbl].append(conf)
                # penalty if picture has drone but doesn't detect drone
                if "no_drone" not in file_name and not file_pred[0]:
                    file_pred[0].append(FALSE_PRED)
                # penalty if picture has no drone but detect drone
                elif "no_drone" in file_name and file_pred[0]:
                    file_pred[0].append(FALSE_PRED)
                # reward if picture has no drone and doesn't detect drone
                elif "no_drone" in file_name and not file_pred[0]:
                    file_pred[0].append(TRUE_PRED)
                # penalty if doesn't detect any person
                if not file_pred[1]:
                    file_pred[1].append(FALSE_PRED)
                preds_dict[file_name] = file_pred
                f.close()
    return preds_dict


def accumulate_predictions(preds_dict):
    preds_accumulate = preds_dict
    for frame, pred in preds_dict.items():
        for cls in pred.keys():
            all_preds = preds_accumulate[frame][cls]
            if len(all_preds) != 0:
                preds_accumulate[frame][cls] = sum(all_preds) / len(all_preds)
            else:
                preds_accumulate[frame][cls] = 0
    return preds_accumulate


def search_frame(preds_accumulate, lbl, conf, ext, selected):
    for frame, pred in preds_accumulate.items():
        if pred[lbl] == conf and frame not in selected:
            # logger.debug(f"Selectd: {selected}")
            logger.debug(f"Return frame {frame}, class: {lbl}, conf: {conf}")
            return frame
    return "None"
    # logger.error("Frame not found")


def choose_lowest_conf_frames(
    preds_accumulate, num_of_imgs, labels, ext, select_drone=True
):
    if not select_drone:
        labels.remove(0)
    preds_by_class = {}
    for _, pred in preds_accumulate.items():
        for cls, conf in pred.items():
            if conf != 0:
                if cls not in preds_by_class.keys():
                    preds_by_class[cls] = []
                preds_by_class[cls].append(conf)
    top_frames_dict = {}
    for cls in labels:
        top_frames_dict[cls] = []
        preds_by_class[cls].sort()
    selected_frames = []
    for cls in labels:
        img_taken = 0
        while img_taken < num_of_imgs:
            top_conf = preds_by_class[cls].pop(0)
            top_frame = search_frame(
                preds_accumulate, cls, top_conf, ext, selected_frames
            )
            if top_frame == "None":
                logger.info("can't find")
            elif top_frame not in selected_frames:
                selected_frames.append(top_frame)
                top_frames_dict[cls].append(top_frame)
                img_taken += 1
    return top_frames_dict


def main(opts):
    labels = list(range(opts.classes))
    top_frames_to_label = {}
    for lbl in labels:
        top_frames_to_label[lbl] = []
    for set in opts.sets:
        preds_dict = get_label_dict(labels=labels.copy(), dir=opts.dir, set_name=set)
        logger.debug(f"preds_dict: {preds_dict}")
        preds_accumulate = accumulate_predictions(preds_dict=preds_dict)
        top_frames = choose_lowest_conf_frames(
            preds_accumulate=preds_accumulate,
            num_of_imgs=opts.num_of_imgs,
            labels=labels.copy(),
            ext=opts.ext,
            select_drone="no_drone" not in set,
        )
        for cls in top_frames.keys():
            top_frames_to_label[cls] += top_frames[cls]
    logger.info(f"Top frames: {top_frames_to_label}")
    with open(OUTPUT_FILE, "w") as f:
        for _, frames in top_frames_to_label.items():
            f.writelines(
                os.path.abspath(opts.test_ds) + "/" + frame + opts.ext + "\n"
                for frame in frames
            )
        logger.info(f"Saved selected frames in {OUTPUT_FILE}")
        f.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", required=True, help="Directory with labels to read"
    )
    parser.add_argument(
        "-test-ds",
        default=os.path.join(
            DATASET_DIR, "picam_data/Full_train_test_val_5/test/images"
        ),
        help="Path to original test set of dataset",
    )
    parser.add_argument(
        "-x",
        "--ext",
        type=str,
        default=".png",
        help="Image file extension",
    )
    parser.add_argument(
        "-n",
        "--num-of-imgs",
        type=int,
        default=5,
        help="Num of images to label per set",
    )
    parser.add_argument(
        "-s",
        "--sets",
        nargs="+",
        default=["Hevolinna1", "Hevolinna2", "Narnia_no_drone"],
        help="Num of images to label per set",
    )
    # parser.add_argument(
    #     "-c",
    #     "--classes",
    #     type=int,
    #     default=2,
    #     help="Number of classes, default: 2 for person and drone",
    # )
    opts = parser.parse_args()
    opts.classes = NUM_OF_CLASSES
    main(opts)
