import os
import json
from glob import glob
from utils.features_extractor import FightingFeaturesExtractor

import pandas as pd
import numpy as np
from tqdm import tqdm


def extract_logs(path="D:\\DDA\\GAME_LOG\\league_replay", save_path='D:\\DDA\\dataset', extract_player='both'):
    feature_path = os.path.join(save_path, 'features')
    label_path = os.path.join(save_path, 'labels')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(feature_path):
        os.mkdir(os.path.join(save_path, 'features'))
    if not os.path.exists(label_path):
        os.mkdir(os.path.join(save_path, 'labels'))

    files = glob(os.path.join(path, '*.json'))
    fe = FightingFeaturesExtractor([])

    for file in tqdm(files):
        with open(file, 'r') as f:
            save_file_name = file.split('\\')[-1]

            data = json.load(f)
            fe.set_game_data(data)
            extracted_p1_features = []
            extracted_p1_labels = []
            extracted_p2_features = []
            extracted_p2_labels = []

            for round_data in data['rounds']:
                if len(round_data) == 0:
                    continue
                for frame_data in round_data:
                    fe.set_frame_data(frame_data)
                    current_frame = fe.get_feature('current_frame')
                    elapsed_milli_time = fe.get_feature('elapsed_milli_time')

                    p1_features, p1_labels = get_features(fe, 'P1', current_frame, elapsed_milli_time)
                    p2_features, p2_labels = get_features(fe, 'P2', current_frame, elapsed_milli_time)

                    extracted_p1_features.append(p1_features)
                    extracted_p1_labels.append(p1_labels)
                    extracted_p2_features.append(p2_features)
                    extracted_p2_labels.append(p2_labels)

            if extract_player == 'both' or extract_player == 'p1':
                p1_features_df = pd.DataFrame(extracted_p1_features)
                p1_features_df.to_csv(os.path.join(save_path, f'features/1P_{save_file_name}.csv'), index=False)
                p1_labels_df = pd.DataFrame(extracted_p1_labels)
                p1_labels_df.to_csv(os.path.join(save_path, f'labels/1P_{save_file_name}.csv'), index=False)
            if extract_player == 'both' or extract_player == 'p2':
                p2_features_df = pd.DataFrame(extracted_p2_features)
                p2_features_df.to_csv(os.path.join(save_path, f'features/2P_{save_file_name}.csv'), index=False)
                p2_labels_df = pd.DataFrame(extracted_p2_labels)
                p2_labels_df.to_csv(os.path.join(save_path, f'labels/2P_{save_file_name}.csv'), index=False)


def get_features(fe, player, current_frame, elapsed_milli_time):
    """
    Inputs:
        current_frame
        elapsed_milli_time
        self_x, self_y (each is normalized to [0, 1])
        oppo_x, oppo_y (each is normalized to [0, 1])
        players_distance_x, y()
        hp_diff()
        self_front
        self_state_id
        self_action_id
        self_speed_x
        self_speed_y
        self_remaining_frames
        self_energy
        self_available_action_mask()
        self_hp
        self_attack {
            speed_x, y
            damage
            guard_damage
            attack_type
            impact_x, y
            attack_distance_from_oppo_x, y()
        }
        self_proj_num()
        self_closest_proj_to_oppo() [
            speed_x, y
            damage
            guard_damage
            attack_type
            impact_x, y
            proj_distance_from_oppo_x, y()
        ]
        oppo_front
        oppo_state_id
        oppo_action_id
        oppo_speed_x
        oppo_speed_y
        oppo_remaining_frames
        oppo_energy
        oppo_available_action_mask()
        oppo_hp
        oppo_attack {
            speed_x, y
            damage
            guard_damage
            attack_type
            impact_x, y
            attack_distance_from_self_x, y()
        }
        oppo_proj_num()
        oppo_closest_proj_to_self() [
            speed_x, y
            damage
            guard_damage
            attack_type
            impact_x, y
            proj_distance_from_self_x, y()
        ]
        is_opponent_approaching()
        is_self_approaching()

    Labels:
        key_a, b, c, u, d, r, l
    """
    features = {}
    labels = {}
    opponent = 'P2' if player == 'P1' else 'P1'

    # common features
    features['current_frame'] = current_frame
    features['elapsed_milli_time'] = elapsed_milli_time
    features['players_x_distance'] = fe.get_special('players_x_distance()', player)
    features['players_y_distance'] = fe.get_special('players_y_distance()', player)
    features['hp_diff'] = fe.get_special('hp_diff()', player)

    # self featurs
    features['self_front'] = fe.get_feature(f'{player}.front')
    features['self_state_id'] = fe.get_feature(f'{player}.state_id')
    features['self_action_id'] = fe.get_feature(f'{player}.action_id')
    features['self_x'] = (fe.get_feature(f'{player}.left') + fe.get_feature(f'{player}.right')) / 2
    features['self_y'] = (fe.get_feature(f'{player}.top') + fe.get_feature(f'{player}.bottom')) / 2
    features['self_speed_x'] = fe.get_feature(f'{player}.speed_x')
    features['self_speed_y'] = fe.get_feature(f'{player}.speed_y')
    features['self_remaining_frames'] = fe.get_feature(f'{player}.remaining_frames')
    features['self_energy'] = fe.get_feature(f'{player}.energy')
    features['self_hp'] = fe.get_feature(f'{player}.hp')
    features['is_self_approaching'] = fe.get_special('opponent_is_approaching()', opponent)
    # self_available_action_mask =

    # self attack features
    features['self_att_type'] = fe.get_feature(f'{player}.attack.attack_type_id')  # if att_type is None or att_type == 0
    if features['self_att_type'] is not None and features['self_att_type'] != 0:
        features['self_att_speed_x'] = fe.get_feature(f'{player}.attack.speed_x')
        features['self_att_speed_y'] = fe.get_feature(f'{player}.attack.speed_y')
        features['self_att_damage'] = fe.get_feature(f'{player}.attack.hit_damage')
        features['self_att_guard_damage'] = fe.get_feature(f'{player}.attack.guard_damage')
        features['self_att_impact_x'] = fe.get_feature(f'{player}.attack.impact_x')
        features['self_att_impact_y'] = fe.get_feature(f'{player}.attack.impact_y')
        features['self_att_distance_from_oppo_x'] = -fe.get_special('attack_x_distance()', opponent)  # bad for far distance
        features['self_att_distance_from_oppo_y'] = -fe.get_special('attack_y_distance()', opponent)
    else:
        features['self_att_type'] = 0
        features['self_att_speed_x'] = 0
        features['self_att_speed_y'] = 0
        features['self_att_damage'] = 0
        features['self_att_guard_damage'] = 0
        features['self_att_impact_x'] = 0
        features['self_att_impact_y'] = 0
        features['self_att_distance_from_oppo_x'] = 0
        features['self_att_distance_from_oppo_y'] = 0

    # self projectile features
    features['self_proj_num'], features['self_closest_proj_to_oppo_type'], \
        features['self_closest_proj_to_oppo_speed_x'], features['self_closest_proj_to_oppo_speed_y'], \
        features['self_closest_proj_to_oppo_damage'], features['self_closest_proj_to_oppo_guard_damage'], \
        features['self_closest_proj_to_oppo_impact_x'], features['self_closest_proj_to_oppo_impact_y'], \
        features['self_closest_proj_to_oppo_distance_x'], features['self_closest_proj_to_oppo_distance_y'] \
        = fe.get_special('closest_projectile_info()', opponent)

    # opponent features
    features['oppo_front'] = fe.get_feature(f'{opponent}.front')
    features['oppo_state_id'] = fe.get_feature(f'{opponent}.state_id')
    features['oppo_action_id'] = fe.get_feature(f'{opponent}.action_id')
    features['oppo_x'] = (fe.get_feature(f'{opponent}.left') + fe.get_feature(f'{opponent}.right')) / 2
    features['oppo_y'] = (fe.get_feature(f'{opponent}.top') + fe.get_feature(f'{opponent}.bottom')) / 2
    features['oppo_speed_x'] = fe.get_feature(f'{opponent}.speed_x')
    features['oppo_speed_y'] = fe.get_feature(f'{opponent}.speed_y')
    features['oppo_remaining_frames'] = fe.get_feature(f'{opponent}.remaining_frames')
    features['oppo_energy'] = fe.get_feature(f'{opponent}.energy')
    features['oppo_hp'] = fe.get_feature(f'{opponent}.hp')
    features['is_oppo_approaching'] = fe.get_special('opponent_is_approaching()', player)
    # oppo_available_action_make =

    # opponent attack features
    features['oppo_att_type'] = fe.get_feature(f'{opponent}.attack.attack_type_id')
    if features['oppo_att_type'] is not None and features['oppo_att_type'] != 0:
        features['oppo_att_speed_x'] = fe.get_feature(f'{opponent}.attack.speed_x')
        features['oppo_att_speed_y'] = fe.get_feature(f'{opponent}.attack.speed_y')
        features['oppo_att_damage'] = fe.get_feature(f'{opponent}.attack.hit_damage')
        features['oppo_att_guard_damage'] = fe.get_feature(f'{opponent}.attack.guard_damage')
        features['oppo_att_impact_x'] = fe.get_feature(f'{opponent}.attack.impact_x')
        features['oppo_att_impact_y'] = fe.get_feature(f'{opponent}.attack.impact_y')
        features['oppo_att_distance_from_self_x'] = fe.get_special('attack_x_distance()', player)
        features['oppo_att_distance_from_self_y'] = fe.get_special('attack_y_distance()', player)
    else:
        features['oppo_att_type'] = 0
        features['oppo_att_speed_x'] = 0
        features['oppo_att_speed_y'] = 0
        features['oppo_att_damage'] = 0
        features['oppo_att_guard_damage'] = 0
        features['oppo_att_impact_x'] = 0
        features['oppo_att_impact_y'] = 0
        features['oppo_att_distance_from_self_x'] = 0
        features['oppo_att_distance_from_self_y'] = 0

    # opponent projectile features
    features['oppo_proj_num'], features['oppo_closest_proj_to_self_type'], \
        features['oppo_closest_proj_to_self_speed_x'], features['oppo_closest_proj_to_self_speed_y'], \
        features['oppo_closest_proj_to_self_damage'], features['oppo_closest_proj_to_self_guard_damage'], \
        features['oppo_closest_proj_to_self_impact_x'], features['oppo_closest_proj_to_self_impact_y'], \
        features['oppo_closest_proj_to_self_distance_x'], features['oppo_closest_proj_to_self_distance_y'] \
        = fe.get_special('closest_projectile_info()', player)

    # Labels
    labels['current_frame'] = current_frame
    labels['elapsed_milli_time'] = elapsed_milli_time
    labels['key_a'] = fe.get_feature('P1.key_a')  # boolean
    labels['key_b'] = fe.get_feature('P1.key_b')
    labels['key_c'] = fe.get_feature('P1.key_c')
    labels['key_u'] = fe.get_feature('P1.key_up')
    labels['key_d'] = fe.get_feature('P1.key_down')
    labels['key_r'] = fe.get_feature('P1.key_right')
    labels['key_l'] = fe.get_feature('P1.key_left')

    return features, labels


def get_compressed_features(df, filename):
    compressed_features = {}
    compressed_features['game'] = filename
    total_frame_cnt = len(df)

    for i in range(5):
        cnt = df[(192*i <= df['self_x']) & (df['self_x'] < 192*(i+1))].shape[0]
        compressed_features[f'self_time_spent_in_bin{i}'] = cnt / total_frame_cnt

        cnt = df[(192 * i <= df['oppo_x']) & (df['oppo_x'] < 192 * (i + 1))].shape[0]
        compressed_features[f'oppo_time_spent_in_bin{i}'] = cnt / total_frame_cnt

    compressed_features['close_distance_ratio'] = df[df['players_x_distance'] < 100].shape[0] / total_frame_cnt
    compressed_features['avg_distance'] = df['players_x_distance'].mean()

    approaching_cnt = df[df['is_self_approaching'] == 1].shape[0]
    moving_away_cnt = df[df['is_self_approaching'] == -1].shape[0]
    if approaching_cnt > 0 or moving_away_cnt > 0:
        compressed_features['self_approaching_ratio'] = approaching_cnt / (approaching_cnt + moving_away_cnt)
        compressed_features['self_moving_away_ratio'] = moving_away_cnt / (approaching_cnt + moving_away_cnt)
    else:
        compressed_features['self_approaching_ratio'] = 0.
        compressed_features['self_moving_away_ratio'] = 0.
    approaching_cnt = df[df['is_oppo_approaching'] == 1].shape[0]
    moving_away_cnt = df[df['is_oppo_approaching'] == -1].shape[0]
    if approaching_cnt > 0 or moving_away_cnt > 0:
        compressed_features['oppo_approaching_ratio'] = approaching_cnt / (approaching_cnt + moving_away_cnt)
        compressed_features['oppo_moving_away_ratio'] = moving_away_cnt / (approaching_cnt + moving_away_cnt)
    else:
        compressed_features['oppo_approaching_ratio'] = 0.
        compressed_features['oppo_moving_away_ratio'] = 0.

    compressed_features['self_avg_approaching_speed'] \
        = df[df['is_self_approaching'] == 1]['self_speed_x'].abs().mean() \
            if df[df['is_self_approaching'] == 1].shape[0] > 0 else 0.
    compressed_features['self_avg_moving_away_speed'] \
        = -df[df['is_self_approaching'] == -1]['self_speed_x'].abs().mean() \
            if df[df['is_self_approaching'] == -1].shape[0] > 0 else 0.
    compressed_features['oppo_avg_approaching_speed'] \
        = df[df['is_oppo_approaching'] == 1]['oppo_speed_x'].abs().mean() \
            if df[df['is_oppo_approaching'] == 1].shape[0] > 0 else 0.
    compressed_features['oppo_avg_moving_away_speed'] \
        = -df[df['is_oppo_approaching'] == -1]['oppo_speed_x'].abs().mean() \
            if df[df['is_oppo_approaching'] == -1].shape[0] > 0 else 0.

    for i in range(55):
        compressed_features[f'self_action{i}_ratio'] = df[df['self_action_id'] == i].shape[0] / total_frame_cnt
        compressed_features[f'oppo_action{i}_ratio'] = df[df['oppo_action_id'] == i].shape[0] / total_frame_cnt
    for i in range(4):
        compressed_features[f'self_state{i}_ratio'] = df[df['self_state_id'] == i].shape[0] / total_frame_cnt
        compressed_features[f'oppo_state{i}_ratio'] = df[df['oppo_state_id'] == i].shape[0] / total_frame_cnt
    self_attacks = df[df['self_att_type'] > 0]
    oppo_attacks = df[df['oppo_att_type'] > 0]
    self_attack_cnt = self_attacks.shape[0]
    oppo_attack_cnt = oppo_attacks.shape[0]
    for i in range(1, 5):
        if self_attack_cnt > 0:
            compressed_features[f'self_attack_type{i}_ratio'] = df[df['self_att_type'] == i].shape[0] / self_attack_cnt
        else:
            compressed_features[f'self_attack_type{i}_ratio'] = 0.
        if oppo_attack_cnt > 0:
            compressed_features[f'oppo_attack_type{i}_ratio'] = df[df['oppo_att_type'] == i].shape[0] / oppo_attack_cnt
        else:
            compressed_features[f'oppo_attack_type{i}_ratio'] = 0.

    compressed_features['self_attack_ratio'] = self_attack_cnt / total_frame_cnt
    compressed_features['oppo_attack_ratio'] = oppo_attack_cnt / total_frame_cnt

    compressed_features['self_attack_avg_damage'] \
        = self_attacks['self_att_damage'].mean() if self_attack_cnt > 0 else 0.
    compressed_features['oppo_attack_avg_damage'] \
        = oppo_attacks['oppo_att_damage'].mean() if oppo_attack_cnt > 0 else 0.

    self_projs = df[df['self_closest_proj_to_oppo_type'] > 0]
    oppo_projs = df[df['oppo_closest_proj_to_self_type'] > 0]
    self_projs_cnt = self_projs.shape[0]
    oppo_projs_cnt = oppo_projs.shape[0]
    for i in range(1, 5):
        if self_projs_cnt > 0:
            compressed_features[f'self_projectiles_type{i}_ratio'] \
                = df[df['self_closest_proj_to_oppo_type'] == i].shape[0] / self_projs_cnt
        else:
            compressed_features[f'self_projectiles_type{i}_ratio'] = 0.
        if oppo_projs_cnt > 0:
            compressed_features[f'oppo_projectiles_type{i}_ratio'] \
                = df[df['oppo_closest_proj_to_self_type'] == i].shape[0] / oppo_projs_cnt
        else:
            compressed_features[f'oppo_projectiles_type{i}_ratio'] = 0.
    compressed_features['self_projectiles_ratio'] = self_projs_cnt / total_frame_cnt
    compressed_features['oppo_projectiles_ratio'] = oppo_projs_cnt / total_frame_cnt

    compressed_features['self_projectiles_avg_damage'] \
        = self_projs['self_closest_proj_to_oppo_damage'].mean() \
        if self_projs_cnt > 0 else 0.
    compressed_features['oppo_projectiles_avg_damage'] \
        = oppo_projs['oppo_closest_proj_to_self_damage'].mean() \
        if oppo_projs_cnt > 0 else 0.

    self_proj_nums = df[df['self_proj_num'] > 0]
    oppo_proj_nums = df[df['oppo_proj_num'] > 0]
    compressed_features['self_avg_projectiles_num'] = self_proj_nums['self_proj_num'].mean() \
        if self_proj_nums.shape[0] > 0 else 0.
    compressed_features['oppo_avg_projectiles_num'] = oppo_proj_nums['oppo_proj_num'].mean() \
        if oppo_proj_nums.shape[0] > 0 else 0.


    indices = df.index
    hp_diff_by_frame = df.diff()

    # be hit data && guard data
    self_be_hit_condition = hp_diff_by_frame['self_hp'] < 0
    self_be_hit_indices = indices[self_be_hit_condition]
    be_hit_cnt = 0
    guard_cnt = 0
    for idx in self_be_hit_indices:
        oppo_attack_dmg = df.iloc[idx-1]['oppo_att_damage']
        oppo_proj_dmg = df.iloc[idx-1]['oppo_closest_proj_to_self_damage']

        if -1 * hp_diff_by_frame.iloc[idx]['self_hp'] == oppo_attack_dmg \
                or hp_diff_by_frame.iloc[idx]['self_hp'] == oppo_proj_dmg:
            be_hit_cnt += 1
        else:
            guard_cnt += 1

    # hit data && blocked data
    self_hit_condition = hp_diff_by_frame['oppo_hp'] < 0
    self_hit_indices = indices[self_hit_condition]
    hit_cnt = 0
    blocked_cnt = 0
    for idx in self_hit_indices:
        self_attack_dmg = df.iloc[idx-1]['self_att_damage']
        self_proj_dmg = df.iloc[idx-1]['self_closest_proj_to_oppo_damage']

        if -1 * hp_diff_by_frame.iloc[idx]['oppo_hp'] == self_attack_dmg \
                or hp_diff_by_frame.iloc[idx]['oppo_hp'] == self_proj_dmg:
            hit_cnt += 1
        else:
            blocked_cnt += 1

    compressed_features['self_be_hit_per_second'] = be_hit_cnt / total_frame_cnt * 60
    compressed_features['self_hit_per_second'] = hit_cnt / total_frame_cnt * 60
    compressed_features['self_guard_per_second'] = guard_cnt / total_frame_cnt * 60
    compressed_features['self_blocked_per_second'] = blocked_cnt / total_frame_cnt * 60

    def zero_crossing(data):
        return np.where(np.diff(np.sign(np.array(data))))[0]
    zero_crossing_cnt = len(zero_crossing(df['hp_diff']))

    compressed_features['avg_hp_diff'] = df['hp_diff'].mean()
    compressed_features['self_hp_sup_ratio'] = len(df[df['hp_diff'] > 0]) / total_frame_cnt
    compressed_features['oppo_hp_sup_ratio'] = len(df[df['hp_diff'] < 0]) / total_frame_cnt
    compressed_features['avg_hp_zero_crossing'] = zero_crossing_cnt / total_frame_cnt * 60
    compressed_features['avg_self_hp_reducing_speed'] = hp_diff_by_frame['self_hp'].sum() / total_frame_cnt * 60
    compressed_features['avg_oppo_hp_reducing_speed'] = hp_diff_by_frame['oppo_hp'].sum() / total_frame_cnt * 60
    compressed_features['avg_self_energy_gaining_speed'] \
        = hp_diff_by_frame[hp_diff_by_frame['self_energy'] > 0]['self_energy'].sum() / total_frame_cnt * 60
    compressed_features['avg_oppo_energy_gaining_speed'] \
        = hp_diff_by_frame[hp_diff_by_frame['oppo_energy'] > 0]['oppo_energy'].sum() / total_frame_cnt * 60
    compressed_features['avg_self_energy_reducing_speed'] \
        = hp_diff_by_frame[hp_diff_by_frame['self_energy'] < 0]['self_energy'].sum() / total_frame_cnt * 60
    compressed_features['avg_oppo_energy_reducing_speed'] \
        = hp_diff_by_frame[hp_diff_by_frame['oppo_energy'] < 0]['oppo_energy'].sum() / total_frame_cnt * 60

    return compressed_features


def compress_features(path='D:\\DDA\\dataset\\user_data'):
    dirs = os.listdir(path)

    def skip_logic(idx):
        if idx == 0:
            return False
        if (idx-1) % 15 == 0:
            return False
        return True

    for dir in tqdm(dirs):
        files = glob(os.path.join(path, dir, "features\\1P*.csv"))
        save_path = os.path.join("D:\\DDA\\dataset\\compressed", dir + '.csv')
        compressed_list = []
        for file in files:
            # df = pd.read_csv(file, skiprows=lambda x: skip_logic(x))
            df = pd.read_csv(file)
            compressed_list.append(get_compressed_features(df, file.split("\\")[-1].replace("1P_", "")))
            # compressed_features = pd.DataFrame.from_dict(compressed_features, orient='index').T
        compressed_list = pd.DataFrame(compressed_list)
        compressed_list.to_csv(save_path)


if __name__ == '__main__':
    root_dir = 'D:\\Research\\Game\\DDA\\data\\GAME_LOG\\replay'
    dir_list = os.listdir(root_dir)
    #
    for d in dir_list:
        extract_logs(os.path.join(root_dir, d), os.path.join(f'D:\\Research\\Game\\DDA\\data\\dataset\\final_user_test_data\\{d}'), 'p1')
    # compress_features()

