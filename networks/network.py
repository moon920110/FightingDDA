import os
import torch
import torch.nn as nn


class DDANetwork(nn.Module):
    def __init__(self):
        super(DDANetwork, self).__init__()

        # common features
        self.common_features = nn.Sequential(
            nn.Linear(5, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # self features
        self.self_features = nn.Sequential(
            nn.Linear(11, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # self attack features
        self.self_attack_features = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # self projectile features
        self.self_projectile_features = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # oppo features
        self.oppo_features = nn.Sequential(
            nn.Linear(11, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # oppo attack features
        self.oppo_attack_features = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # oppo projectile features
        self.oppo_projectile_features = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # core memory
        self.core = nn.LSTM(input_size=832, hidden_size=128, num_layers=2)

        # out
        self.act = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def forward(self, common_f, self_f, self_att_f, self_proj_f, opp_f, opp_att_f, opp_proj_f):
        c = self.common_features(common_f)
        s = self.self_features(self_f)
        sa = self.self_attack_features(self_att_f)
        sp = self.self_projectile_features(self_proj_f)
        o = self.oppo_features(opp_f)
        oa = self.oppo_attack_features(opp_att_f)
        op = self.oppo_projectile_features(opp_proj_f)

        f = torch.cat([c, s, sa, sp, o, oa, op], dim=-1)
        core, _ = self.core(f.unsqueeze(0))
        action = self.act(core[:, -1, :])

        return action
