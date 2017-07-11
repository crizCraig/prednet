from __future__ import print_function
import fnmatch
import os
import sys
import shutil

# Usage:
# Copy this file into the parent directory you want to split and run it from there

CURR_DIR = dir_path = os.path.dirname(os.path.realpath(__file__))

# Customize the following to your needs
TRAIN_PCT = 95.
TEST_PCT = 2.5
VALID_PCT = 2.5
OUT_DIR = '/home/jetta/src/prednet2/data'
FILE_TYPE = 'png'
SHUFFLE = False
_try_out = False
IMAGE_DIR = '/home/jetta/src/prednet2/data_raw'
DRY_RUN = False


def main():
    if _try_out:
        create_test_files()

    print('Creating ', OUT_DIR)
    if not DRY_RUN:
        os.makedirs(os.path.join(IMAGE_DIR, OUT_DIR))
        os.makedirs(os.path.join(IMAGE_DIR, OUT_DIR, 'train'))
        os.makedirs(os.path.join(IMAGE_DIR, OUT_DIR, 'test'))
        os.makedirs(os.path.join(IMAGE_DIR, OUT_DIR, 'val'))

    matches = []
    for root, dir_names, file_names in os.walk(IMAGE_DIR):
        for filename in fnmatch.filter(file_names, '*.' + FILE_TYPE):
            matches.append(os.path.join(root, filename))

    if SHUFFLE:
        from random import shuffle
        shuffle(matches)
    else:
        matches = sorted(matches)

    print('splitting', len(matches), 'files')

    train_end = int(len(matches) * float(TRAIN_PCT) / 100.)
    test_end = int(train_end + len(matches) * float(TEST_PCT) / 100.)
    assert VALID_PCT == 100 - TRAIN_PCT - TEST_PCT

    def copy_to(src, dest):
        full_dest = os.path.join(OUT_DIR, dest)
        print('copying', src, 'to', full_dest)
        if not DRY_RUN:
            shutil.copy2(src, full_dest)

    train_finish_group = None
    test_finish_group = None
    for i, match in enumerate(matches):
        group = match[-21:-15]
        if i < train_end:
            copy_to(match, 'train')
            train_finish_group = group
        elif group == train_finish_group:
            copy_to(match, 'train')
        elif i < test_end:
            copy_to(match, 'test')
            test_finish_group = group
        elif group == test_finish_group:
            copy_to(match, 'test')
        else:
            copy_to(match, 'val')
        print(i)

    print('train end', train_end, 'test end', test_end)

def create_test_files():
    if os.path.exists(os.path.join(IMAGE_DIR, OUT_DIR)):
        shutil.rmtree(os.path.join(IMAGE_DIR, OUT_DIR))

    some_dir = 'some_dir'
    if not os.path.exists(os.path.join(IMAGE_DIR, some_dir)):
        print('Creating ', some_dir)
        os.makedirs(os.path.join(IMAGE_DIR, some_dir))

    for i in range(10):
        file_name = 'test_' + str(i) + '.' + FILE_TYPE
        full_path = os.path.join(IMAGE_DIR, some_dir, file_name)
        if not os.path.exists(full_path):
            with open(full_path, 'w+') as _:
                pass


if __name__ == '__main__':
    sys.exit(main())
