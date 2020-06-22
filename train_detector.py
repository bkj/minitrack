#!/usr/bin/env python

"""
    train_detector.py
    
    !! Should train on more realistic data -- training like this means the network can't handle occlusions at all
"""

import json
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F

from cnn import SimpleDetector

if __name__ == "__main__":

    # --
    # IO

    pos = np.array(Image.open('data/pos.png'))
    neg = np.array(Image.open('data/neg.png'))
    nul = np.array(Image.open('data/nul.png'))
    
    d = pos.shape[0]
    center = d // 2 + 1
    
    X = torch.Tensor(np.stack([pos, neg, nul]))
    X = X.permute(0, 3, 1, 2)[:,:1]
    X = X / 255

    y = torch.zeros_like(X).long()
    y[0, 0, center, center] = 1

    X = X.repeat((256, 1, 1, 1))
    y = y.repeat((256, 1, 1, 1))

    y = y.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()

    # --
    # Train model

    model = SimpleDetector()
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(32):
        out  = model(X)
        out  = out.permute(0, 2, 3, 1).reshape(-1, 2)
        
        loss = F.cross_entropy(out, y, weight=torch.FloatTensor([1, d ** 2]))
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        p = out.argmax(dim=1) == 1
        recall    = (p & (y == 1)).float().sum() / (y == 1).float().sum()
        precision = (y[p] == 1).float().mean()
        print(json.dumps({"precision" : float(precision), "recall" : float(recall)}))

    torch.save(model.state_dict(), 'models/detector.pth')
