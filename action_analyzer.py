import pandas as pd
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from const.action import Action, ALL_ACTIONS

def get_action_dist():
    root_path = "D:\\Research\\Game\\DDA\\data\\dataset\\final_user_test_data"
    save_path = "D:\\Research\\Game\\DDA\\data\\analysis"
    dirs = os.listdir(root_path)

    player_action_one_hot = np.zeros((20, 12, 6))
    agent_action_one_hot = np.zeros((20, 12, 6))
    for d in dirs:
        files = glob(os.path.join(root_path, d, "features", "*.csv"))
        player = int(d.replace('p', ""))
        if player == 21:
            player = 10
        for file in files:
            chunks = file.split('_')
            agent = ""
            for chunk in chunks:
                if "Agent" in chunk:
                    agent = int(chunk.replace("Agent", ""))

            df = pd.read_csv(file)
            oppo_upper = df[df['hp_diff'] < -40]
            self_upper = df[df['hp_diff'] > 40]
            same = df[(-40 <= df['hp_diff']) & (df['hp_diff'] <= 40)]
            oppo_upper_agent_actions = oppo_upper['oppo_action_id']
            self_upper_agent_actions = self_upper['oppo_action_id']
            same_agent_actions = same['oppo_action_id']
            for agent_action in same_agent_actions:
                group = Action.to_group(agent_action)
                agent_action_one_hot[player-1, agent-1, group] += 1
            # save_file_name = file.split('\\')[-1] + '_player_action.png'
            # plt.hist(df['self_action_id'], bins=55)
            # plt.savefig(os.path.join(save_path, save_file_name))
            # save_file_name = file.split('\\')[-1] + '_agent_action.png'
            # plt.figure()
            # plt.hist(df['oppo_action_id'], bins=55)
            # plt.savefig(os.path.join(save_path, save_file_name))
            # plt.show()

    # np.save('player_action.npy', player_action_one_hot)
    np.save('data/same_agent_action_group.npy', agent_action_one_hot)


def get_action_dist_per_agent():
    agent_action = np.load('data/agent_action_groups_per_pred_state.npy')
    print(agent_action.shape)
    action_dist = agent_action.mean(axis=0)
    labels = ['move', 'guard', 'out of control', 'normal attack', 'command attack', 'special attack']
    x = np.array([i for i in range(0, 6*2, 2)])
    x2 = np.array([i for i in range(1, 6*2, 2)])
    com = [2, 3, 4, 6, 8, 10, 11, 12]

    baseline_same_acts = action_dist[0, :]
    baseline = {'agent1':
        {
            'player_upper': [],
            'draw': [],
            'agent_upper': [],
        }
    }
    # same_freq_acts = baseline_same_acts.argsort()[-10:][::-1]
    # same_freq_acts = same_freq_acts[baseline_same_acts[same_freq_acts] != 0]
    # su_freq_acts = baseline_self_upper_acts.argsort()[-10:][::-1]
    # su_freq_acts = su_freq_acts[baseline_self_upper_acts[su_freq_acts] != 0]
    # ou_freq_acts = baseline_oppo_upper_acts.argsort()[-10:][::-1]
    # ou_freq_acts = ou_freq_acts[baseline_oppo_upper_acts[ou_freq_acts] != 0]
    idx = 0
    for sa, sua, oua in zip(baseline_same_acts, baseline_self_upper_acts, baseline_oppo_upper_acts):
        baseline['agent1']['draw'].append(Action.group_to_string(idx) + f'({sa/baseline_same_acts.sum()*100:.2f}%)')
        baseline['agent1']['player_upper'].append(Action.group_to_string(idx) + f'({sua/baseline_self_upper_acts.sum()*100:.2f}%)')
        baseline['agent1']['agent_upper'].append(Action.group_to_string(idx) + f'({oua/baseline_oppo_upper_acts.sum()*100:.2f}%)')
        idx += 1
    # for i in com:
    #     plt.bar(x, baseline)
    #     d = action_dist[i-1, :] / action_dist[i-1].sum()
    #     # plt.pie(d, labels=labels, autopct='%.1f%%')
    #     plt.bar(x2, d)
    #     plt.title(f'agent1 vs agent{i}')
    #     plt.savefig(os.path.join("D:\\Research\\Game\\DDA\\data\\analysis", f"same_agent{i}_pie.png"))
    #     plt.show()
    df = pd.DataFrame(baseline)
    df.to_csv(os.path.join("D:\\Research\\Game\\DDA\\data\\analysis", f"baseline_action_dist2.csv"))


def get_action_dist_per_agent2():
    same_agent_action = np.load('data/same_agent_action_group.npy')
    oppo_upper_agent_action = np.load('data/oppo_upper_agent_action_group.npy')
    self_upper_agent_action = np.load('data/self_upper_agent_action_group.npy')
    # print(agent_action)
    same_action_dist = same_agent_action.sum(axis=0)
    oppo_upper_action_dist = oppo_upper_agent_action.sum(axis=0)
    self_upper_action_dist = self_upper_agent_action.sum(axis=0)
    labels = ['move', 'guard', 'out of control', 'normal attack', 'command attack', 'special attack']
    x = np.array([i for i in range(0, 6*2, 2)])
    x2 = np.array([i for i in range(1, 6*2, 2)])
    com = [2, 3, 4, 6, 8, 10, 11, 12]

    baseline_same_acts = same_action_dist[0, :]
    baseline_self_upper_acts = self_upper_action_dist[0, :]
    baseline_oppo_upper_acts = oppo_upper_action_dist[0, :]
    baseline = {'agent1':
        {
            'player_upper': [],
            'draw': [],
            'agent_upper': [],
        }
    }
    # same_freq_acts = baseline_same_acts.argsort()[-10:][::-1]
    # same_freq_acts = same_freq_acts[baseline_same_acts[same_freq_acts] != 0]
    # su_freq_acts = baseline_self_upper_acts.argsort()[-10:][::-1]
    # su_freq_acts = su_freq_acts[baseline_self_upper_acts[su_freq_acts] != 0]
    # ou_freq_acts = baseline_oppo_upper_acts.argsort()[-10:][::-1]
    # ou_freq_acts = ou_freq_acts[baseline_oppo_upper_acts[ou_freq_acts] != 0]
    print(baseline_same_acts)
    idx = 0
    for sa, sua, oua in zip(baseline_same_acts, baseline_self_upper_acts, baseline_oppo_upper_acts):
        baseline['agent1']['draw'].append(Action.group_to_string(idx) + f'({sa/baseline_same_acts.sum()*100:.2f}%)')
        baseline['agent1']['player_upper'].append(Action.group_to_string(idx) + f'({sua/baseline_self_upper_acts.sum()*100:.2f}%)')
        baseline['agent1']['agent_upper'].append(Action.group_to_string(idx) + f'({oua/baseline_oppo_upper_acts.sum()*100:.2f}%)')
        idx += 1
    # for i in com:
    #     plt.bar(x, baseline)
    #     d = action_dist[i-1, :] / action_dist[i-1].sum()
    #     # plt.pie(d, labels=labels, autopct='%.1f%%')
    #     plt.bar(x2, d)
    #     plt.title(f'agent1 vs agent{i}')
    #     plt.savefig(os.path.join("D:\\Research\\Game\\DDA\\data\\analysis", f"same_agent{i}_pie.png"))
    #     plt.show()
    df = pd.DataFrame(baseline)
    df.to_csv(os.path.join("D:\\Research\\Game\\DDA\\data\\analysis", f"baseline_action_dist2.csv"))


if __name__ == '__main__':
    # get_action_dist()
    get_action_dist_per_agent2()
