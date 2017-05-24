'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''
from __future__ import print_function
import os
import requests
from bs4 import BeautifulSoup
import urllib
import numpy as np
from scipy.misc import imread, imresize
import hickle as hkl
from kitti_settings import *


desired_im_sz = (160, 160)
categories = ['sfo']

# Recordings used for validation and testing.
val_recordings = [('sfo', 'val')]
test_recordings = [('sfo', 'test')]

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        _, folders, _ = os.walk(c_dir).next()
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for category, folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'raw', category, folder)
            _, _, files = os.walk(im_dir).next()
            im_list += [os.path.join(im_dir, f) for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files)

        print('Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        if im_list:
            X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
            for i, im_file in enumerate(im_list):
                try:
                    im = imread(im_file)
                    X[i] = process_im(im, desired_im_sz)
                except IOError:
                    print('Could not read', im_file)

            hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
            hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))


# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im


if __name__ == '__main__':
    # download_data()
    # extract_data()
    process_data()
