import os
import glob
import argparse
from pylabel import importer

def coco2yolo(dataset_name, dataset_path):
    json_path = os.path.join(dataset_path, 'annotations/instances_default.json')
    importer.ImportCoco(json_path).export.ExportToYoloV5(
        output_path=os.path.join(dataset_path, 'labels'), yaml_file=dataset_name + '.yaml', cat_id_index=int(0))

def export(opts, dataset_path):
    if opts.train is None:
        all_files = glob.glob(dataset_path + '/images/*.PNG')
        all_files.sort()
        with open(os.path.join(dataset_path, f"{opts.dataset}.txt"), 'w+') as f:
            for file in all_files:
                f.write(file + '\n')
            f.close()
    else:
        train_files = glob.glob(dataset_path + opts.train)
        train_files.sort()
        with open(os.path.join(dataset_path, "train.txt"), 'w+') as f:
            for file in train_files:
                f.write(file + '\n')
            f.close()
        val_files = glob.glob(dataset_path + opts.val)
        val_files.sort()
        with open(os.path.join(dataset_path, "val.txt"), 'w+') as f:
            for file in val_files:
                f.write(file + '\n')
            f.close()

def main(opts):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(ROOT, opts.dataset)
    export(opts, DATA_PATH)
    coco2yolo(opts.dataset, DATA_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='picam_safe', help='name of dataset')
    parser.add_argument('--train', type=str, default=None, help='name of train dataset')
    parser.add_argument('--val', type=str, default=None, help='name of val dataset')
    opts = parser.parse_args()
    main(opts)
