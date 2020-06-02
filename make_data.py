#!/usr/bin/env python

"""
    make_data.py
"""

import numpy as np
from joblib import Parallel, delayed

from rsub import *
from matplotlib import pyplot as plt

_ = np.random.seed(123)

# --
# Generate data

n_traj = 32

xys = []
for _ in range(n_traj):
    rad    = np.random.uniform(0.5, 2)
    offset = np.random.uniform(-rad, rad, 2)
    speed  = int(np.random.uniform(100, 200))
    
    xy = np.column_stack([
        np.cos(np.linspace(0, 2 * np.pi, speed)),
        np.sin(np.linspace(0, 2 * np.pi, speed)),
    ])
    
    xy = xy * rad + offset
    xy = (xy * 400).astype(np.int) + 400
    
    if np.random.uniform() > 0.5:
        xy = xy[::-1]
    
    xys.append(xy)

# --
# Save frames

def plot_frame(frame, n_noise=10):
    rng = np.random.RandomState(frame)
    
    fig = plt.figure(figsize=(5, 5), dpi=160, frameon=False)
    ax  = fig.add_axes([0, 0, 1, 1])
    
    for idx, xy in enumerate(xys):
        _ = ax.scatter(xy[frame,1], 800 - xy[frame,0], s=20, c='white', marker="x")
    
    noise = rng.choice(800, (n_noise, 2))
    _     = ax.scatter(noise[:,1], 800 - noise[:,0], s=20, c='white', marker="s")
    
    _ = ax.set_facecolor('black')
    _ = ax.set_xlim(0, 800)
    _ = ax.set_ylim(0, 800)
    _ = fig.savefig(f'frames/frame.{frame:04d}.png')
    _ = plt.close()


min_traj = min([xy.shape[0] for xy in xys])
xys      = [xy[:min_traj] for xy in xys]

jobs     = [delayed(plot_frame)(frame) for frame in range(min_traj)]
_        = Parallel(backend='multiprocessing', n_jobs=8, verbose=True)(jobs)

# --
# Save ground truth

gt = [np.column_stack([np.arange(xy.shape[0]), xy, np.repeat(i, xy.shape[0])]) for i, xy in enumerate(xys)]
gt = np.row_stack(gt)
sel = (
    (gt[:,1] < 800) &
    (gt[:,1] >= 0)  &
    (gt[:,2] < 800) &
    (gt[:,2] >= 0)
)
gt = gt[sel]
np.save('gt.npy', gt)

