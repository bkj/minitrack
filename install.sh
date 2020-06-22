#!/bin/bash

# install.sh

conda create -y -n minitrack_env python=3.7
conda activate minitrack_env

pip install numpy==1.18.4
pip install lap
pip install Pillow
pip install joblib
pip install matplotlib
pip install git+https://github.com/bkj/rsub
pip install pandas

conda install -y -c pytorch pytorch==1.4.0 torchvision