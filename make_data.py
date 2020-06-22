#!/usr/bin/env python

"""
    make_data.py
"""

import os
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

# --
# Parameters

n_traj = 32
dim    = 800
n_jobs = 8
seed   = 123

_ = np.random.seed(seed)

# --
# Generate data

os.makedirs('data/frames', exist_ok=True)

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
    xy = (xy * dim // 2).astype(np.int) + dim // 2
    
    if np.random.uniform() > 0.5:
        xy = xy[::-1]
    
    xys.append(xy)

# --
# Save frames

def plot_frame(frame, n_noise=10):
    rng = np.random.RandomState(frame)
    
    fig = plt.figure(figsize=(5, 5), dpi=dim // 5, frameon=False)
    ax  = fig.add_axes([0, 0, 1, 1])
    
    for idx, xy in enumerate(xys):
        _ = ax.scatter(xy[frame,1], dim - xy[frame,0], s=20, c='white', marker="x")
    
    noise = rng.choice(dim, (n_noise, 2))
    _     = ax.scatter(noise[:,1], dim - noise[:,0], s=20, c='white', marker="s")
    
    _ = ax.set_facecolor('black')
    _ = ax.set_xlim(0, dim)
    _ = ax.set_ylim(0, dim)
    _ = fig.savefig(f'data/frames/frame.{frame:04d}.png')
    _ = plt.close()


min_traj = min([xy.shape[0] for xy in xys])
xys      = [xy[:min_traj] for xy in xys]

jobs     = [delayed(plot_frame)(frame) for frame in range(min_traj)]
_        = Parallel(backend='multiprocessing', n_jobs=n_jobs, verbose=True)(jobs)

# --
# Save ground truth

gt = [np.column_stack([np.arange(xy.shape[0]), xy, np.repeat(i, xy.shape[0])]) for i, xy in enumerate(xys)]
gt = np.row_stack(gt)
gt = gt[(
    (gt[:,1] < dim) &
    (gt[:,1] >= 0)  &
    (gt[:,2] < dim) &
    (gt[:,2] >= 0)
)]
np.save('data/gt.npy', gt)

