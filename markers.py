#!/usr/bin/env python

"""
    markers.py
"""

from rsub import *
from matplotlib import pyplot as plt

# --
# Positive marker

fig = plt.figure(figsize=(0.19375, 0.19375), dpi=160, frameon=False)
ax  = fig.add_axes([0, 0, 1, 1])

_ = ax.scatter(400, 400, s=20, c='white', marker="x")

_ = ax.set_facecolor('black')
_ = ax.set_xlim(0, 800)
_ = ax.set_ylim(0, 800)
_ = fig.savefig(f'pos.png')
_ = plt.close()

# --
# Negative marker

fig = plt.figure(figsize=(0.19375, 0.19375), dpi=160, frameon=False)
ax  = fig.add_axes([0, 0, 1, 1])

_ = ax.scatter(400, 400, s=20, c='white', marker="s")

_ = ax.set_facecolor('black')
_ = ax.set_xlim(0, 800)
_ = ax.set_ylim(0, 800)
_ = fig.savefig(f'neg.png')
_ = plt.close()

# --
# Empty marker

fig = plt.figure(figsize=(0.19375, 0.19375), dpi=160, frameon=False)
ax  = fig.add_axes([0, 0, 1, 1])

_ = ax.scatter(400, 400, s=20, c='black', marker="s")

_ = ax.set_facecolor('black')
_ = ax.set_xlim(0, 800)
_ = ax.set_ylim(0, 800)
_ = fig.savefig(f'nul.png')
_ = plt.close()
