WINDOW_HEIGHT = 640
WINDOW_WIDTH = 960

BIN_WIDTH = 192

# Features name
FEATURE_NAMES = {
    'common_features': [
        'current_frame',
        'elapsed_milli_time',
        'players_x_distance',
        'players_y_distance',
        'health_diff',
    ],
    'self_features': [
        'self_front',
        'self_state_id',
        'self_action_id',
        'self_x',
        'self_y',
        'self_speed_x',
        'self_speed_y',
        'self_remaining_frames',
        'self_energy',
        'self_hp'
        'is_self_approaching',
    ],
    'self_attack_features': [
        'self_att_type',
        'self_att_speed_x',
        'self_att_speed_y',
        'self_att_damage',
        'self_att_guard_damage',
        'self_att_impact_x',
        'self_att_impact_y',
        'self_att_distance_from_oppo_x',
        'self_att_distance_from_oppo_y',
    ],
    'self_projectile_features': [
        'self_proj_num',
        'self_closest_proj_to_oppo_type',
        'self_closest_proj_to_oppo_speed_x',
        'self_closest_proj_to_oppo_speed_y',
        'self_closest_proj_to_oppo_damage',
        'self_closest_proj_to_oppo_guard_damage',
        'self_closest_proj_to_oppo_impact_x',
        'self_closest_proj_to_oppo_impact_y',
        'self_closest_proj_to_oppo_distance_x',
        'self_closest_proj_to_oppo_distance_y',
    ],
    'opponent_features': [
        'oppo_front',
        'oppo_state_id',
        'oppo_action_id',
        'oppo_x',
        'oppo_y',
        'oppo_speed_x',
        'oppo_speed_y',
        'oppo_remaining_frames',
        'oppo_energy',
        'oppo_hp'
        'is_oppo_approaching',
    ],
    'opponent_attack_features': [
        'oppo_att_type',
        'oppo_att_speed_x',
        'oppo_att_speed_y',
        'oppo_att_damage',
        'oppo_att_guard_damage',
        'oppo_att_impact_x',
        'oppo_att_impact_y',
        'oppo_att_distance_from_self_x',
        'oppo_att_distance_from_self_y',
    ],
    'opponent_projectile_features': [
        'oppo_proj_num',
        'oppo_closest_proj_to_self_type',
        'oppo_closest_proj_to_self_speed_x',
        'oppo_closest_proj_to_self_speed_y',
        'oppo_closest_proj_to_self_damage',
        'oppo_closest_proj_to_self_guard_damage',
        'oppo_closest_proj_to_self_impact_x',
        'oppo_closest_proj_to_self_impact_y',
        'oppo_closest_proj_to_self_distance_x',
        'oppo_closest_proj_to_self_distance_y',
    ],
    'actions': [
        'key_a',
        'key_b',
        'key_c',
        'key_up',
        'key_down',
        'key_left',
        'key_right',
    ],
}

