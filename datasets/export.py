import os
import glob
import argparse

def main(opts):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(ROOT, opts.dataset)
    if opts.train is None:
        all_files = glob.glob(DATA_PATH + opts.extension)
        all_files.sort()
        with open(os.path.join(DATA_PATH, f"{opts.dataset}.txt"), 'w+') as f:
            for file in all_files:
                f.write(file + '\n')
            f.close()
    else:
        train_files = glob.glob(DATA_PATH + opts.train)
        train_files.sort()
        with open(os.path.join(DATA_PATH, "train.txt"), 'w+') as f:
            for file in train_files:
                f.write(file + '\n')
            f.close()
        val_files = glob.glob(DATA_PATH + opts.val)
        val_files.sort()
        with open(os.path.join(DATA_PATH, "val.txt"), 'w+') as f:
            for file in val_files:
                f.write(file + '\n')
            f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='picam_safe', help='name of dataset')
    parser.add_argument('--extension', type=str, default='/images/*.PNG', help='name of dataset')
    parser.add_argument('--train', type=str, default=None, help='name of train dataset')
    parser.add_argument('--val', type=str, default=None, help='name of val dataset')
    opts = parser.parse_args()
    main(opts)
