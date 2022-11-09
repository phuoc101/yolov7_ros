import os
import os.path as osp
import argparse


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


def main(opts):
    labels = list(range(opts.classes))
    preds_dict = get_label_dict(labels=labels, dir=opts.dir)
    print(preds_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", required=True, help="Directory with detections to read"
    )
    parser.add_argument(
        "-c",
        "--classes",
        default=2,
        type=int,
        help='Number of classes. Default: 2 ("drone" and "person")',
    )
    opts = parser.parse_args()
    main(opts)
