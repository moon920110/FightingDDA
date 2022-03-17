from enum import Enum


class Action(Enum):
    # state
    NEUTRAL = 'NEUTRAL'
    STAND = 'STAND'

    # move
    FORWARD_WALK = 'FORWARD_WALK'
    DASH = 'DASH'
    BACK_STEP = 'BACK_STEP'
    CROUCH = 'CROUCH'
    JUMP = 'JUMP'
    FOR_JUMP = 'FOR_JUMP'
    BACK_JUMP = 'BACK_JUMP'

    # state
    AIR = 'AIR'

    # guard
    STAND_GUARD = 'STAND_GUARD'
    CROUCH_GUARD = 'CROUCH_GUARD'
    AIR_GUARD = 'AIR_GUARD'

    # out of control
    STAND_GUARD_RECOV = 'STAND_GUARD_RECOV'
    CROUCH_GUARD_RECOV = 'CROUCH_GUARD_RECOV'
    AIR_GUARD_RECOV = 'AIR_GUARD_RECOV'
    STAND_RECOV = 'STAND_RECOV'
    CROUCH_RECOV = 'CROUCH_RECOV'
    AIR_RECOV = 'AIR_RECOV'

    # state
    CHANGE_DOWN = 'CHANGE_DOWN'
    DOWN = 'DOWN'
    RISE = 'RISE'
    LANDING = 'LANDING'

    # normal attack
    THROW_A = 'THROW_A'
    THROW_B = 'THROW_B'
    THROW_HIT = 'THROW_HIT'
    THROW_SUFFER = 'THROW_SUFFER'

    # normal attack
    STAND_A = 'STAND_A'
    STAND_B = 'STAND_B'
    CROUCH_A = 'CROUCH_A'
    CROUCH_B = 'CROUCH_B'
    AIR_A = 'AIR_A'
    AIR_B = 'AIR_B'
    AIR_DA = 'AIR_DA'
    AIR_DB = 'AIR_DB'
    STAND_FA = 'STAND_FA'
    STAND_FB = 'STAND_FB'
    CROUCH_FA = 'CROUCH_FA'
    CROUCH_FB = 'CROUCH_FB'
    AIR_FA = 'AIR_FA'
    AIR_FB = 'AIR_FB'
    AIR_UA = 'AIR_UA'
    AIR_UB = 'AIR_UB'

    # special attack
    STAND_D_DF_FA = 'STAND_D_DF_FA'
    STAND_D_DF_FB = 'STAND_D_DF_FB'
    STAND_F_D_DFA = 'STAND_F_D_DFA'
    STAND_F_D_DFB = 'STAND_F_D_DFB'
    STAND_D_DB_BA = 'STAND_D_DB_BA'
    STAND_D_DB_BB = 'STAND_D_DB_BB'
    AIR_D_DF_FA = 'AIR_D_DF_FA'
    AIR_D_DF_FB = 'AIR_D_DF_FB'
    AIR_F_D_DFA = 'AIR_F_D_DFA'
    AIR_F_D_DFB = 'AIR_F_D_DFB'
    AIR_D_DB_BA = 'AIR_D_DB_BA'
    AIR_D_DB_BB = 'AIR_D_DB_BB'
    STAND_D_DF_FC = 'STAND_D_DF_FC'

    @staticmethod
    def to_group(action_num):
        if action_num < 2 or action_num == 9 or 19 <= action_num < 23:  # state
            return 0
        elif 2 <= action_num < 9:  # move
            return 1
        elif 13 <= action_num < 19:  # recovery
            return 2
        elif 23 <= action_num < 43:  # normal attack
            return 3
        elif 43 <= action_num:  # special attack
            return 4
        else:  # guard
            return 5

    @staticmethod
    def group_to_string(group_num):
        if group_num == 0:
            return 'STATE'
        elif group_num == 1:
            return 'MOVE'
        elif group_num == 2:
            return 'OUT_OF_CONTROL'
        elif group_num == 3:
            return 'NORMAL_ATTACK'
        elif group_num == 4:
            return 'SPECIAL_ATTACK'
        else:
            return 'GUARD'

    @staticmethod
    def to_ordinal(action):
        if action == Action.NEUTRAL: return 0
        if action == Action.STAND: return 1
        if action == Action.FORWARD_WALK: return 2
        if action == Action.DASH: return 3
        if action == Action.BACK_STEP: return 4
        if action == Action.CROUCH: return 5
        if action == Action.JUMP: return 6
        if action == Action.FOR_JUMP: return 7
        if action == Action.BACK_JUMP: return 8
        if action == Action.AIR: return 9
        if action == Action.STAND_GUARD: return 10
        if action == Action.CROUCH_GUARD: return 11
        if action == Action.AIR_GUARD: return 12
        if action == Action.STAND_GUARD_RECOV: return 13
        if action == Action.CROUCH_GUARD_RECOV: return 14
        if action == Action.AIR_GUARD_RECOV: return 15
        if action == Action.STAND_RECOV: return 16
        if action == Action.CROUCH_RECOV: return 17
        if action == Action.AIR_RECOV: return 18
        if action == Action.CHANGE_DOWN: return 19
        if action == Action.DOWN: return 20
        if action == Action.RISE: return 21
        if action == Action.LANDING: return 22

        if action == Action.THROW_A: return 23
        if action == Action.THROW_B: return 24
        if action == Action.THROW_HIT: return 25
        if action == Action.THROW_SUFFER: return 26

        if action == Action.STAND_A: return 27
        if action == Action.STAND_B: return 28
        if action == Action.CROUCH_A: return 29
        if action == Action.CROUCH_B: return 30
        if action == Action.AIR_A: return 31
        if action == Action.AIR_B: return 32
        if action == Action.AIR_DA: return 33
        if action == Action.AIR_DB: return 34
        if action == Action.STAND_FA: return 35
        if action == Action.STAND_FB: return 36
        if action == Action.CROUCH_FA: return 37
        if action == Action.CROUCH_FB: return 38
        if action == Action.AIR_FA: return 39
        if action == Action.AIR_FB: return 40
        if action == Action.AIR_UA: return 41
        if action == Action.AIR_UB: return 42

        if action == Action.STAND_D_DF_FA: return 43
        if action == Action.STAND_D_DF_FB: return 44
        if action == Action.STAND_F_D_DFA: return 45
        if action == Action.STAND_F_D_DFB: return 46
        if action == Action.STAND_D_DB_BA: return 47
        if action == Action.STAND_D_DB_BB: return 48
        if action == Action.AIR_D_DF_FA: return 49
        if action == Action.AIR_D_DF_FB: return 50
        if action == Action.AIR_F_D_DFA: return 51
        if action == Action.AIR_F_D_DFB: return 52
        if action == Action.AIR_D_DB_BA: return 53
        if action == Action.AIR_D_DB_BB: return 54

        if action == Action.STAND_D_DF_FC: return 55

USELESS_ACTIONS = [
    Action.STAND,
    Action.AIR,
    Action.STAND_GUARD_RECOV,
    Action.CROUCH_GUARD_RECOV,
    Action.AIR_GUARD_RECOV,
    Action.STAND_RECOV,
    Action.CROUCH_RECOV,
    Action.AIR_RECOV,
    Action.CHANGE_DOWN,
    Action.DOWN,
    Action.RISE,
    Action.LANDING,
    Action.THROW_HIT,
    Action.THROW_SUFFER
]

AIR_ACTIONS = [
    Action.AIR_GUARD,
    Action.AIR_A,
    Action.AIR_B,
    Action.AIR_DA,
    Action.AIR_DB,
    Action.AIR_FA,
    Action.AIR_FB,
    Action.AIR_UA,
    Action.AIR_UB,
    Action.AIR_D_DF_FA,
    Action.AIR_D_DF_FB,
    Action.AIR_F_D_DFA,
    Action.AIR_F_D_DFB,
    Action.AIR_D_DB_BA,
    Action.AIR_D_DB_BB
]

GROUND_ACTIONS = [
    Action.STAND_D_DB_BA,
    Action.BACK_STEP,
    Action.FORWARD_WALK,
    Action.DASH,
    Action.JUMP,
    Action.FOR_JUMP,
    Action.BACK_JUMP,
    Action.STAND_GUARD,
    Action.CROUCH_GUARD,
    Action.THROW_A,
    Action.THROW_B,
    Action.STAND_A,
    Action.STAND_B,
    Action.CROUCH_A,
    Action.CROUCH_B,
    Action.STAND_FA,
    Action.STAND_FB,
    Action.CROUCH_FA,
    Action.CROUCH_FB,
    Action.STAND_D_DF_FA,
    Action.STAND_D_DF_FB,
    Action.STAND_F_D_DFA,
    Action.STAND_F_D_DFB,
    Action.STAND_D_DB_BB
]

ALL_ACTIONS = [a.value for a in Action]
ALL_USEFUL_ACTIONS = [a.value for a in Action if a not in USELESS_ACTIONS]
AIR_ACTIONS = [a.value for a in AIR_ACTIONS]
GROUND_ACTIONS = [a.value for a in GROUND_ACTIONS]
