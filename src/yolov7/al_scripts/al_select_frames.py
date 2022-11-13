import os
import os.path as osp
import argparse
from loguru import logger


def get_label_dict(labels, dir):
    preds_dict = dict()
    for file in os.listdir(dir):
        with open(osp.join(dir, file), "r") as f:
            file_name = file.split(".")[0]
            file_pred = dict()
            for lbl in labels:
                file_pred[lbl] = []
            for line in f.readlines():
                lbl, _, _, _, _, conf = line.split()
                lbl = int(lbl)
                conf = float(conf)
                file_pred[lbl].append(conf)
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


def search_frame(preds_accumulate, lbl, conf, ext):
    for frame, pred in preds_accumulate.items():
        if pred[lbl] == conf:
            logger.debug(f"Return frame {frame}, class: {lbl}, conf: {conf}")
            return frame + ext
    logger.error("Frame not found")


def choose_lowest_conf_frames(preds_accumulate, num_of_imgs, labels, ext):
    preds_by_class = {}
    for _, pred in preds_accumulate.items():
        for cls, conf in pred.items():
            if conf != 0:
                if cls not in preds_by_class.keys():
                    preds_by_class[cls] = []
                preds_by_class[cls].append(conf)
    top_frames_list = []
    for cls in labels:
        preds_by_class[cls].sort()
    for cls in labels:
        img_taken = 0
        pop_idx = 0
        while img_taken < num_of_imgs:
            top_conf = preds_by_class[cls].pop(pop_idx)
            top_frame = search_frame(preds_accumulate, cls, top_conf, ext)
            pop_idx += 1
            if top_frame not in top_frames_list:
                top_frames_list.append(top_frame)
                img_taken += 1
    return top_frames_list


def main(opts):
    labels = list(range(opts.classes))
    preds_dict = get_label_dict(labels=labels, dir=opts.dir)
    preds_accumulate = accumulate_predictions(preds_dict=preds_dict)
    top_frames_list = choose_lowest_conf_frames(
        preds_accumulate=preds_accumulate,
        num_of_imgs=opts.num_of_imgs,
        labels=labels,
        ext=opts.ext,
    )
    logger.info(f"Top frames: {top_frames_list}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", required=True, help="Directory with detections to read"
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
        default=10,
        help="Num of images to label",
    )
    # parser.add_argument(
    #     "-c",
    #     "--classes",
    #     default=2,
    #     type=int,
    #     help='Number of classes. Default: 2 ("drone" and "person")',
    # )
    opts = parser.parse_args()
    main(opts)
