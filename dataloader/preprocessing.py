import os
import json
from glob import glob
from utils.features_extractor import FightingFeaturesExtractor
from const.consts import *

import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_logs(path="D:\\DDA\\GAME_LOG\\league_replay", save_path='D:\\DDA\\dataset'):
    files = glob(os.path.join(path, '*.json'))
    fe = FightingFeaturesExtractor([])
    game_num = 0

    for file in tqdm(files):
        with open(file, 'r') as f:
            data = json.load(f)
            fe.set_game_data(data)

            for round_data in data['rounds']:
                extracted_p1_features = []
                extracted_p1_labels = []
                extracted_p2_features = []
                extracted_p2_labels = []
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

                p1_features_df = pd.DataFrame(extracted_p1_features)
                p1_features_df.to_csv(os.path.join(save_path, f'features/P1_{game_num}.csv'), index=False)
                p1_labels_df = pd.DataFrame(extracted_p1_labels)
                p1_labels_df.to_csv(os.path.join(save_path, f'labels/P1_{game_num}.csv'), index=False)
                p2_features_df = pd.DataFrame(extracted_p2_features)
                p2_features_df.to_csv(os.path.join(save_path, f'features/P2_{game_num}.csv'), index=False)
                p2_labels_df = pd.DataFrame(extracted_p2_labels)
                p2_labels_df.to_csv(os.path.join(save_path, f'labels/P2_{game_num}.csv'), index=False)

                game_num += 1


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

    features['current_frame'] = current_frame
    features['elapsed_milli_time'] = elapsed_milli_time
    features['players_x_distance'] = fe.get_special('players_x_distance()', player)
    features['players_y_distance'] = fe.get_special('players_y_distance()', player)
    features['hp_diff'] = fe.get_special('hp_diff()', player)

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
    features['self_proj_num'], features['self_closest_proj_to_oppo_type'], \
        features['self_closest_proj_to_oppo_speed_x'], features['self_closest_proj_to_oppo_speed_y'], \
        features['self_closest_proj_to_oppo_damage'], features['self_closest_proj_to_oppo_guard_damage'], \
        features['self_closest_proj_to_oppo_impact_x'], features['self_closest_proj_to_oppo_impact_y'], \
        features['self_closest_proj_to_oppo_distance_x'], features['self_closest_proj_to_oppo_distance_y'] \
        = fe.get_special('closest_projectile_info()', opponent)

    features['oppo_front'] = fe.get_feature(f'{opponent}.front')
    features['oppo_state_id'] = fe.get_feature(f'{opponent}.state_id')
    features['oppo_action_id'] = fe.get_feature(f'{opponent}.action_id')
    features['oppo_x'] = (fe.get_feature(f'{opponent}.left') + fe.get_feature(f'{opponent}.right')) / 2
    features['oppo_y'] = (fe.get_feature(f'{opponent}.top') + fe.get_feature(f'{opponent}.bottom')) / 2
    features['oppo_speed_x'] = fe.get_feature(f'{opponent}.speed_x')
    features['oppo_speed_y'] = fe.get_feature(f'{opponent}.speed_y')
    features['oppo_remaining_frames'] = fe.get_feature(f'{opponent}.remaining_frames')
    features['oppo_energy'] = fe.get_feature(f'{opponent}.energy')
    features['self_hp'] = fe.get_feature(f'{opponent}.hp')
    features['is_oppo_approaching'] = fe.get_special('opponent_is_approaching()', player)
    # oppo_available_action_make =
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
    features['oppo_proj_num'], features['oppo_closest_proj_to_self_type'], \
        features['oppo_closest_proj_to_self_speed_x'], features['oppo_closest_proj_to_self_speed_y'], \
        features['oppo_closest_proj_to_self_damage'], features['oppo_closest_proj_to_self_guard_damage'], \
        features['oppo_closest_proj_to_self_impact_x'], features['oppo_closest_proj_to_self_impact_y'], \
        features['oppo_closest_proj_to_self_distance_x'], features['oppo_closest_proj_to_self_distance_y'] \
        = fe.get_special('closest_projectile_info()', player)

    ### Labels
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

extract_logs()
# with open(path, 'r') as f:
#     data = json.load(f)
#     print(json.dumps(data, indent=4))
#
# # Construct a new object by passing a list of the desired features (this is used later to retrieve the features in bulk).
# fe = FightingFeaturesExtractor(['P1.attack.hit_area.bottom', 'players_distance()'])
#
# # Inform the features extractor of the game data.
# fe.set_game_data(data)
#
# # Iterate through the rounds and the frames.
# for round_data in data['rounds']:
#     for frame_data in round_data:
#
#         # Let's print only some random frames in this demo
#         import random
#         if random.random() > 0.0002:
#             continue
#
#         # Get HP
#         print('P2 HP: %s' % fe.get_hp(frame_data, 'P2'))
#
#         # Cache the frame data in the features extractor
#         fe.set_frame_data(frame_data)
#
#         # Get a feature: use a name with dotted notation and call the method get_feature().
#         print('P1 front: ', fe.get_feature('P1.front'))
#         print('P1 state: : %s' % fe.get_feature('P1.state'))
#         print('P2 remaining frames: %s' % fe.get_feature('P2.remaining_frames'))
#         print('P2 hit-area bottom: %s' % fe.get_feature('P2.bottom'))
#         print('P1 attack hit-area top: %s' % fe.get_feature('P1.attack.hit_area.top')) # Could be None if there is no attack
#         print('P2 first projectile impact x: %s' % fe.get_feature('P2.projectiles[0].impact_x')) # Could be None if there are not projectiles
#
#         # Get number of projectiles of P2
#         print('P2 no. of projectiles: %s' % fe.get_feature('P2.projectiles.count'))
#         if fe.get_feature('P2.projectiles.count') > 0:
#             print('P2 first projectile hit-area left: %s' % fe.get_feature('P2.projectiles[0].hit_area.left'))
#
#
#         # Get some special features (the second parameter passed to get_special indicates who is the "player")
#         print('Distance between players hit-boxes: %s' % fe.get_special('players_x_distance()', 'P1'))
#         print('Difference in X of players: %s' % fe.get_special('players_x_diff()', 'P1'))
#         print('Difference in Y of players: %s' % fe.get_special('players_y_diff()', 'P1'))
#         print('Sign of player y speed: %s' % fe.get_special('player_is_falling()', 'P1'))
#         print('Sign of opponent y speed: %s' % fe.get_special('opponent_is_falling()', 'P1'))
#         print('Is opponent getting closer: %s' % fe.get_special('opponent_is_approaching()', 'P1'))
#         print('Is opponent attacking: %s' % fe.get_special('opponent_is_attacking()', 'P1'))
#         print('Distance between the player hit-area and the hit-area of the closest opponent attack or projectile: %s' % fe.get_special('closest_threat_x_distance()', 'P1'))
#         print('Distance between the player hit-area and the hit-area of the opponent attack: %s' % fe.get_special('attack_x_distance()', 'P1'))
#         print('Distance between the player hit-area and the hit-area of the closest opponent projectile: %s ' % fe.get_special('closest_projectile_x_distance()', 'P1'))
#         print('Can opponent perform action: %s' % fe.get_special('opponent_is_busy()', 'P1'))
#
#         print()
#
# print()
#
# # Get information about features (can be useful when you want to normalize continuous features or convert enum features to one-hot representations, etc).
# print('Info of feature "P1.state": %s' % fe.get_feature_info('P1.state'))
# print('Info of feature "P1.action": %s' % fe.get_feature_info('P1.action'))
# print('Info of feature "P2.remaining_frames": %s' % fe.get_feature_info('P2.remaining_frames'))
# print('Info of feature "P2.projectiles[0].impact_x": %s' % fe.get_feature_info('P2.projectiles[0].impact_x'))
# print('Info of feature "players_x_distance()": %s' % fe.get_feature_info('players_x_distance()'))
# print('Info of feature "players_x_diff()": %s' % fe.get_feature_info('players_x_diff()'))
# print('Info of feature "player_is_falling()": %s' % fe.get_feature_info('player_is_falling()'))
# print('Info of feature "closest_threat_x_distance()": %s' % fe.get_feature_info('closest_threat_x_distance()'))
# print('Info of feature "attack_x_distance()": %s' % fe.get_feature_info('attack_x_distance()'))
# print('Info of feature "closest_projectile_x_distance()": %s' % fe.get_feature_info('closest_projectile_x_distance()'))
#
#
#
# # --- OUTPUT ---
#
# # P2 HP: 480
# # P1 state: : CROUCH
# # P2 remaining frames: 42
# # P2 hit-area bottom: 640
# # P1 attack hit-area top: None
# # P2 first projectile impact x: None
# # P2 no. of projectiles: 0
# # Distance between players hit-boxes: 2
# # Difference in X of players: 3
# # Difference in Y of players: 0
# # Sign of player y speed: 0
# # Sign of opponent y speed: 0
# # Is opponent getting closer: False
# # Is opponent attacking: False
# # Distance between the player hit-area and the hit-area of the closest opponent attack or projectile: None
# # Distance between the player hit-area and the hit-area of the opponent attack: None
# # Distance between the player hit-area and the hit-area of the closest opponent projectile: None
# # Can opponent perform action: True
# #
# # P2 HP: 64
# # P1 state: : AIR
# # P2 remaining frames: 32
# # P2 hit-area bottom: 640
# # P1 attack hit-area top: 297
# # P2 first projectile impact x: 10
# # P2 no. of projectiles: 1
# # P2 first projectile hit-area left: 624
# # Distance between players hit-boxes: 2
# # Difference in X of players: 3
# # Difference in Y of players: -3
# # Sign of player y speed: 1
# # Sign of opponent y speed: 0
# # Is opponent getting closer: False
# # Is opponent attacking: False
# # Distance between the player hit-area and the hit-area of the closest opponent attack or projectile: None
# # Distance between the player hit-area and the hit-area of the opponent attack: None
# # Distance between the player hit-area and the hit-area of the closest opponent projectile: None
# # Can opponent perform action: True
# #
# #
# # Info of feature "P1.state": {'iterable': False, 'nullable': False, 'type': 'enum', 'possible_values': ['STAND', 'CROUCH', 'AIR', 'DOWN']}
# # Info of feature "P1.action": {'iterable': False, 'nullable': False, 'type': 'enum', 'possible_values': ['NEUTRAL', 'STAND', 'FORWARD_WALK', 'DASH', 'BACK_STEP', 'CROUCH', 'JUMP', 'FOR_JUMP', 'BACK_JUMP', 'AIR', 'STAND_GUARD', 'CROUCH_GUARD', 'AIR_GUARD', 'STAND_GUARD_RECOV', 'CROUCH_GUARD_RECOV', 'AIR_GUARD_RECOV', 'STAND_RECOV', 'CROUCH_RECOV', 'AIR_RECOV', 'CHANGE_DOWN', 'DOWN', 'RISE', 'LANDING', 'THROW_A', 'THROW_B', 'THROW_HIT', 'THROW_SUFFER', 'STAND_A', 'STAND_B', 'CROUCH_A', 'CROUCH_B', 'AIR_A', 'AIR_B', 'AIR_DA', 'AIR_DB', 'STAND_FA', 'STAND_FB', 'CROUCH_FA', 'CROUCH_FB', 'AIR_FA', 'AIR_FB', 'AIR_UA', 'AIR_UB', 'STAND_D_DF_FA', 'STAND_D_DF_FB', 'STAND_F_D_DFA', 'STAND_F_D_DFB', 'STAND_D_DB_BA', 'STAND_D_DB_BB', 'AIR_D_DF_FA', 'AIR_D_DF_FB', 'AIR_F_D_DFA', 'AIR_F_D_DFB', 'AIR_D_DB_BA', 'AIR_D_DB_BB', 'STAND_D_DF_FC']}
# # Info of feature "P2.remaining_frames": {'iterable': False, 'nullable': False, 'type': <class 'int'>, 'min': 0, 'max': 3615}
# # Info of feature "P2.projectiles[0].impact_x": {'iterable': False, 'nullable': False, 'type': <class 'int'>, 'min': 0, 'max': 30}
# # Info of feature "players_x_distance()": {'iterable': False, 'nullable': False, 'type': <class 'int'>, 'min': 0, 'max': 960}
# # Info of feature "players_x_diff()": {'iterable': False, 'nullable': False, 'type': <class 'int'>, 'min': 0, 'max': 960}
# # Info of feature "player_is_falling()": {'iterable': False, 'nullable': False, 'type': 'enum', 'possible_values': [-1, 0, 1]}
# # Info of feature "closest_threat_x_distance()": {'iterable': False, 'nullable': True, 'type': <class 'int'>, 'min': 0, 'max': 960}
# # Info of feature "attack_x_distance()": {'iterable': False, 'nullable': True, 'type': <class 'int'>, 'min': 0, 'max': 960}
# # Info of feature "closest_projectile_x_distance()": {'iterable': False, 'nullable': True, 'type': <class 'int'>, 'min': 0, 'max': 960}
#
