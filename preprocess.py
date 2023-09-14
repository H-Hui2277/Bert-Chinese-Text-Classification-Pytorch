import argparse

from utils import dataset_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin-file', default='', help='origin data file.')
    parser.add_argument('--save-dir', default='', help='dir for saving data.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    origin_file = args.origin_file
    save_dir = args.save_dir

    dataset_transform(origin_file, save_dir)
