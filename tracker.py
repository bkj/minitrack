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

from train_detector import load_model

# --
# Helpers

def load_img(fname):
    return np.array(Image.open(fname))[...,:3]

# --
# IO

model = load_model()

# Load images
fnames = sorted(glob('frames/*'))
imgs   = np.stack([load_img(fname) for fname in fnames])
imgs   = imgs[...,0]

# Load ground truth
gt  = pd.DataFrame(np.load('gt.npy'))
gt.columns = ('t', 'x', 'y', 'idx')

assert (imgs[(gt.t, gt.x, gt.y)] > 0).all()

out = model.inference(imgs[0])
out = out.softmax(dim=1)[0,1]
out = out.detach()

_ = plt.figure(figsize=(10, 10))
_ = plt.imshow(imgs[0], cmap='gray')
_ = show_plot()

_ = plt.figure(figsize=(10, 10))
_ = plt.imshow(out > 0.5)
show_plot()

(out > 0.5).nonzero()
gt[gt.t == 0].sort_values(['x', 'y'])

# --
# Fake detections

for i in range(n_timesteps - 1):
    xy_1  = xy[t == i]
    xy_2  = xy[t == i + 1]
    idx_1 = idx[t == i]
    idx_2 = idx[t == i + 1]
    
    dist       = cdist(xy_1, xy_2, metric='euclidean')
    _, cols, _ = lapjv(dist, extend_cost=True)
    
    print((idx_1 == idx_2[cols]).mean())

