#!/usr/bin/env python

"""
    
"""

import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from lap import lapjv
from scipy.spatial.distance import squareform, cdist

from rsub import *
from matplotlib import pyplot as plt

from cnn import load_model

# --
# Helpers

def load_img(fname):
    return np.array(Image.open(fname))[...,:3]

# --
# IO

model = load_model()

# Load images
fnames = sorted(glob('data/frames/*'))
imgs   = np.stack([load_img(fname) for fname in fnames])
imgs   = imgs[...,0]

# Load ground truth
gt  = pd.DataFrame(np.load('data/gt.npy'))
gt.columns = ('t', 'x', 'y', 'idx')

assert (imgs[(gt.t, gt.x, gt.y)] > 0).all()

# --
# Run tracker

hist = {}

# First frame
pred = model.inference(img)
pred = pred.softmax(dim=1)[0, 1]
dets = list(zip(*np.where(pred > 0.5)))
last_dets = dets

# Subsequent frames
for frame_idx, img in enumerate(imgs[1:]):
    print(f'frame_idx={frame_idx}')
    pred = model.inference(img)
    pred = pred.softmax(dim=1)[0, 1]
    dets = list(zip(*np.where(pred > 0.5)))
    
    dist = cdist(last_dets, dets, metric='euclidean')
    
    _, cols, _ = lapjv(dist, extend_cost=True)
    
    for det_idx, ld in enumerate(last_dets):
        key = (frame_idx, ld[0], ld[1])
        if cols[det_idx] != -1:
            match = dets[cols[det_idx]]
            val   = (frame_idx + 1, match[0], match[1])
            hist[key] = val
        else:
            hist[key] = None
    
    last_dets = dets

# --
# Plot a track

track = []
k = list(hist.keys())[2]
while True:
    track.append(k)
    k = hist[k]
    if k is None: break

z = np.row_stack(track)
_ = plt.plot(z[:, 1], z[:, 2])
show_plot()
