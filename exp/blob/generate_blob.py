#!/usr/bin/env python

########################################################################################################################
# Synthetic blob dataset:
# 32x32 binary images, each displaying a single blob.  An image has two continuous descriptors in range [0,1]: one
# represents closedness score (how topologically closed the blob is), and the other represents symmetry score (how
# left-right symmetric the blob is).
# Two binary classification tasks are defined: "closed vs. non-closed" and "symmetric vs. non-symmetric".  For each
# task, the images with highest/lowest closedness or symmetry scores are taken as positive/negative examples.
########################################################################################################################

import os
import sys
import shutil
import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.measure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

IM_FORMAT = '.png'
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
if os.path.exists(os.path.join(SCRIPT_PATH, 'data')):
    print 'Data folder exists - creation skipped'
    sys.exit()
else:
    print 'Data folder does not exist - creating...'
np.random.seed(0)
SETS = [{'name': 'blob_train_image_data', 'size': int(25e3)},
        {'name': 'blob_test_image_data', 'size': int(5e3)}]
POS_NEG_PROPORTION = 0.4  # Proportion of images with highest/lowest scores to be taken as positive/negative examples

# Image generation parameters
IM_SZ2 = 5                                          # log2 of image height/width
ELLP_MIN_HGT2 = 3.5                                 # log2 of minimal ellipse height
WARP_SIN_MAX_FREQ = 3 * (2 * np.pi / (2 ** IM_SZ2)) # Maximal frequency of warping sine
WARP_SIN_MAX_AMP = 4                                # Maximal amplitude of warping sine
REG_MIN_SZ2 = 4                                     # log2 of minimal region size
DROP_PROB_MIN = 0.1                                 # Minimal probability of pixel dropping
DROP_PROB_MAX = 0.33                                # Maximal probability of pixel dropping

# Global variables
X, Y = np.meshgrid(range(2 ** IM_SZ2), range(2 ** IM_SZ2))

# Loop through sets
for s in xrange(len(SETS)):
    set_name = SETS[s]['name']
    set_size = SETS[s]['size']
    print set_name, 'creation begun'
    os.makedirs(os.path.join(SCRIPT_PATH, 'data', set_name))
    I_list = []
    # Generation loop - each iteration renders an image
    I_i = 0
    while I_i < set_size:
        if (I_i % 1000) == 0:
            print 'Image {}/{}'.format(I_i,set_size)
        # Create ellipse with random height
        ellp_a = (2 ** IM_SZ2) / 2
        ellp_b = int((2 ** np.random.uniform(ELLP_MIN_HGT2, IM_SZ2)) / 2)
        I = (((X - (2 ** IM_SZ2 / 2 - 0.5)) / ellp_a) ** 2 + ((Y - (2 ** IM_SZ2 / 2 - 0.5)) / ellp_b) ** 2 < 1)\
            .astype(float)
        # Warp according to random sine
        warp_sin = np.random.uniform(0, WARP_SIN_MAX_AMP) * np.sin(
            np.random.uniform(0, WARP_SIN_MAX_FREQ) * np.arange(2 ** IM_SZ2))
        for i in range(2 ** IM_SZ2):
            I[:, i] = np.roll(I[:, i], warp_sin[i].astype(int))
        # Rotate by random degree
        I = scipy.ndimage.interpolation.rotate(I, np.random.uniform(0, 180.0), reshape=False)
        # Scale down to random region size and place in random tile
        reg_sz2 = np.random.randint(REG_MIN_SZ2, IM_SZ2 + 1)
        reg_loc_x = (2 ** reg_sz2) * np.random.randint(0, 2 ** (IM_SZ2 - reg_sz2))
        reg_loc_y = (2 ** reg_sz2) * np.random.randint(0, 2 ** (IM_SZ2 - reg_sz2))
        I_scl = scipy.ndimage.interpolation.zoom(I, 2 ** (reg_sz2 - IM_SZ2))
        I[:] = 0.0
        I[reg_loc_x:reg_loc_x + 2 ** reg_sz2, reg_loc_y:reg_loc_y + 2 ** reg_sz2] = I_scl
        # Drop pixels randomly
        drop_prob = np.random.uniform(DROP_PROB_MIN, DROP_PROB_MAX)
        drop_mask_sym = np.random.binomial(n=1, p=0.5) > 0.5
        if drop_mask_sym:
            drop_mask = np.random.binomial(n=1, p=drop_prob, size=(2 ** reg_sz2, 2 ** (reg_sz2 - 1)))
            drop_mask = np.concatenate((drop_mask, np.fliplr(drop_mask)), axis=1)
        else:
            drop_mask = np.random.binomial(n=1, p=drop_prob, size=(2 ** reg_sz2, 2 ** reg_sz2))
        I[reg_loc_x:(reg_loc_x + 2 ** reg_sz2), reg_loc_y:(reg_loc_y + 2 ** reg_sz2)] *= (1 - drop_mask)
        # Binarize
        I = I > 0.1
        # Reiterate if image is empty, otherwise proceed
        if not np.any(I):
            continue
        else:
            I_i += 1
        # Measure closedness
        I_cls = scipy.ndimage.morphology.binary_closing(np.pad(I,(1,),'constant',constant_values=False))\
                [1:(2 ** IM_SZ2 + 1), 1:(2 ** IM_SZ2 + 1)]
        score_cls = np.sum(I).astype(float) / np.sum(I_cls).astype(float)
        # Measure symmetry
        I_reg = I[reg_loc_x:(reg_loc_x + 2 ** reg_sz2), reg_loc_y:(reg_loc_y + 2 ** reg_sz2)]
        score_sym = np.sum(np.logical_and(I_reg, np.fliplr(I_reg))).astype(float) / np.sum(I_reg).astype(float)
        # Save image and update list
        filename = set_name + str(I_i) + IM_FORMAT
        scipy.misc.imsave(os.path.join(SCRIPT_PATH, 'data', set_name, filename),I)
        I_list.append({'filename': filename, 'score_cls': score_cls, 'score_sym': score_sym})
    # Save image list to CSV file
    with open(os.path.join(SCRIPT_PATH, 'data', set_name, 'list.csv'), 'w') as csv_file:
        fieldnames = ['filename', 'score_cls', 'score_sym']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
        csv_writer.writeheader()
        for I_i in xrange(len(I_list)):
            csv_writer.writerow(I_list[I_i])
    # Plot and save score histograms
    for score_type in ['cls', 'sym']:
        plt.figure()
        plt.hist([I_list[I_i]['score_{}'.format(score_type)] for I_i in xrange(len(I_list))], bins=20)
        plt.title('{} -- score_{}'.format(set_name, score_type))
        plt.savefig(os.path.join(SCRIPT_PATH, 'data', set_name, 'score_{}_hist'.format(score_type) + IM_FORMAT))
    # Index positive/negative examples for classification
    for score_type in ['cls', 'sym']:
        order = sorted(range(len(I_list)), key=lambda i: I_list[i]['score_{}'.format(score_type)])
        for o in xrange(0, int(len(I_list) * POS_NEG_PROPORTION)):
            I_list[order[o]]['label_{}'.format(score_type)] = 0
        for o in xrange(len(I_list) - int(len(I_list) * POS_NEG_PROPORTION), len(I_list)):
            I_list[order[o]]['label_{}'.format(score_type)] = 1
        with open(os.path.join(SCRIPT_PATH, 'data', set_name, 'index_{}.txt'.format(score_type)), 'w') as index:
            for I_i in xrange(len(I_list)):
                if 'label_{}'.format(score_type) in I_list[I_i]:
                    index.write('{} {}\n'.format(I_list[I_i]['filename'], I_list[I_i]['label_{}'.format(score_type)]))
    # Set creation complete
    print set_name, 'creation complete'

print '#######'
print 'Data creation done!'
