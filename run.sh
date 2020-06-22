#!/bin/bash

# run.sh

conda activate minitrack_env

# --
# Generate synthetic data

python make_data.py

# --
# Visualize

ffmpeg -r 10 -i data/frames/frame.%04d.png -vcodec mpeg4 -y data/example.mp4

# --
# Train detector

python train_detector.py

# --
# Test tracker (incomplete)

python tracker.py