#!/usr/bin/env python

"""
    train_detector.py
    
    !! Should train on more realistic data -- training like this means the network can't handle occlusions at all
"""

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F

class SimpleDetector(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()
        d = 31
        
        self.conv1 = nn.Conv2d(1, n_channels, kernel_size=d, padding=d // 2)
        self.head  = nn.Conv2d(n_channels, 2, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.head(x)
        return x
    
    def inference(self, x):
        assert len(x.shape) == 2
        
        x = x[None, None]
        
        if x.dtype == np.float:
            x = x / 255
        
        x = torch.FloatTensor(x)
        with torch.no_grad():
            return self(x)


def load_model(weight_path='detector.pth'):
    model = SimpleDetector()
    model.load_state_dict(torch.load(weight_path))
    model = model.eval()
    return model


if __name__ == "__main__":

    # --
    # IO

    pos = np.array(Image.open('pos.png'))
    neg = np.array(Image.open('neg.png'))
    nul = np.array(Image.open('nul.png'))
    
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
        print(float(precision), float(recall))

    torch.save(model.state_dict(), 'detector.pth')
