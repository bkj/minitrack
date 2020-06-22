#!/usr/bin/env python

"""
    cnn.py
"""

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


def load_model(weight_path='models/detector.pth'):
    model = SimpleDetector()
    model.load_state_dict(torch.load(weight_path))
    model = model.eval()
    return model
