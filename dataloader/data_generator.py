import os
import random
from glob import glob

from tqdm import tqdm
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from const.consts import *


class GameLogDataset(Dataset):
    def __init__(self, root_path='D:\\DDA\\dataset', player='P1'):
        self.data_path = os.path.join(root_path, 'features')
        self.label_path = os.path.join(root_path, 'labels')
        if player == 'both':
            self.data_files = glob(os.path.join(self.data_path, '*.csv'))
            self.label_files = glob(os.path.join(self.label_path, '*.csv'))
        else:
            self.data_files = glob(os.path.join(self.data_path, f'{player}_*.csv'))
            self.label_files = glob(os.path.join(self.label_path, f'{player}_*.csv'))

        self.samples = self._make_samples()

    def _make_samples(self):
        samples = []
        for data, label in tqdm(zip(self.data_files, self.label_files), total=len(self.data_files)):
            features = pd.read_csv(data, index_col=False)
            keys = pd.read_csv(label, index_col=False)
            features['self_x'] = self._transform_position(features['self_x'])
            features['oppo_x'] = self._transform_position(features['oppo_x'])
            features['players_x_distance'] = self._transform_position(features['players_x_distance'])
            features['self_y'] = self._transform_position(features['self_y'], 'height')
            features['oppo_y'] = self._transform_position(features['oppo_y'], 'height')
            np_features = features.to_numpy()
            np_keys = keys.to_numpy()
            for f, k in zip(np_features, np_keys):
                samples.append((f, k))

        return samples

    def _transform_position(self, data, direction='width'):
        if direction == 'width':
            return (WINDOW_WIDTH - data) / WINDOW_WIDTH * 100
        else:
            return WINDOW_HEIGHT - data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


if __name__ == "__main__":
    tt = GameLogDataset()