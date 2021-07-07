import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class FightingDDA:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(lr=0.25e-3, self.model.parameters())