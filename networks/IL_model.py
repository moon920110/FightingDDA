import os
import time

from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.network import DDANetwork
from dataloader.data_generator import GameLogDataset


class FightingDDA:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = DDANetwork()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.25e-3)

    def train(self):
        dataset = GameLogDataset()
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        self.model.train()
        for epc in range(self.args.epoch):
            total_loss = 0.

            for i, (features, labels) in enumerate(data_loader):

