from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from const.action import Action, ALL_ACTIONS
from joblib import load
from collections import deque
from tqdm import tqdm
from datetime import datetime
import seaborn as sns


def get_action_dist():
    root_path = "D:\\Research\\Game\\DDA\\data\\dataset\\final_user_test_data"
    model_root_path = "D:\\Research\\Game\\DDA\\data\\pre-collected_user_data_for_training\\5sec\\new"
    save_path = "D:\\Research\\Game\\DDA\\data\\analysis"
    dirs = os.listdir(root_path)

    challenge_model = load(os.path.join(model_root_path, "Challenge_Lgb.pkl"))
    competence_model = load(os.path.join(model_root_path, "Competence_Randomforest.pkl"))
    valence_model = load(os.path.join(model_root_path, "Valence_Randomforest.pkl"))
    flow_model = load(os.path.join(model_root_path, "Immersion_Randomforest.pkl"))

    model_zoo = {
        2: challenge_model,
        4: competence_model,
        6: valence_model,
        8: flow_model,
    }
    state_dict = {
        2: 'Challenge',
        4: 'Competence',
        6: 'Valence',
        8: 'Flow',
    }
    player_action_one_hot = np.zeros((20, 12, 20, 6))
    agent_action_one_hot = np.zeros((20, 12, 20, 6))  # player, agent, predict_prob, actions
    result_dict = {
        'partc': [],
        'Agent': [],
        'time': [],
        'frame': [],
        'state': [],
        'state_value': [],
        'Action': [],
        'Action Group': [],
        'Player Action': [],
        'Player Action Group': [],
    }
    model_input = deque(maxlen=20)
    for d in tqdm(dirs, desc='users'):
        files = glob(os.path.join(root_path, d, "features", "*.csv"))
        player = int(d.replace('p', ""))
        if player == 21:
            player = 10
        for file in files:
            chunks = file.split('_')
            agent = int(chunks[-2].replace("Agent", ""))
            time = chunks[-1][:-12]
            time = datetime.strptime(time, "%Y.%m.%d-%H.%M")

            if agent not in model_zoo.keys():
                continue

            model = model_zoo[agent]
            df = pd.read_csv(file).iloc[:, 2:]
            for i in range(20):
                model_input.append(df.iloc[0].to_numpy())

            for line in range(len(df)):
                row = df.iloc[line]
                model_input.append(row.to_numpy())
                model_input_np = np.array(model_input).reshape(1, -1)
                if agent > 9:
                    result1 = model[0].predict_proba(model_input_np)[0] * 100
                    result2 = model[1].predict_proba(model_input_np)[0] * 100
                    result = result1[1] + result2[1]
                else:
                    result = model.predict_proba(model_input_np)[0] * 100
                    result = result[1]
                result_n = result // 10

                agent_action = row['oppo_action_id']
                group = Action.to_group(agent_action)
                if line % 15 == 0:
                    result_dict['partc'].append(f'p{player}')
                    result_dict['Agent'].append(f'Agent{agent}')
                    result_dict['time'].append(time)
                    result_dict['frame'].append(line)
                    result_dict['state'].append(state_dict[agent])
                    result_dict['state_value'].append(result)
                    result_dict['Action'].append(agent_action)
                    result_dict['Action Group'].append(group)
                # agent_action_one_hot[player-1, agent-1, int(result_n), group] += 1
    df = pd.DataFrame(result_dict)
    df.to_csv('D:\\Research\\Game\\DDA\\exp_data\\State&Action-Predict-15frame-3.csv')
    # np.save('player_action.npy', player_action_one_hot)
    # np.save('agent_action_groups_per_pred_state.npy', agent_action_one_hot)


def get_action_dist_per_agent():
    path = "D:\\Research\\Game\\DDA\\exp_data\\"
    df = pd.read_csv(os.path.join(path, "State&Action-Predict-15frame-4.csv"))
    df = df[(df['Agent'] == 'Agent1') | (df['Agent'] == 'Agent2') | (df['Agent'] == 'Agent4')
            | (df['Agent'] == 'Agent6') | (df['Agent'] == 'Agent8')]

    new_df = df.groupby(["Agent", "partc", "Action Group"]).count().iloc[:, 0]
    mean_action_group_cnt_per_agent = new_df.groupby(["Agent", "Action Group"]).mean()
    std_action_group_cnt_per_agent = new_df.groupby(["Agent", "Action Group"]).std()
    total_action_group_cnt_per_agent = new_df.groupby(["Agent", "Action Group"]).sum()
    ndf = pd.concat([mean_action_group_cnt_per_agent, std_action_group_cnt_per_agent, total_action_group_cnt_per_agent], axis=1)
    ndf.columns = ["mean", "std", "total"]
    new_df = pd.DataFrame(new_df)
    new_df.columns = ['count']

    # ndf.to_csv(os.path.join(path, "action_dist.csv"))
    # new_df.to_csv(os.path.join(path, "action_count_dist.csv"))


def analyze_action_statistics():
    path = "D:\\Research\\Game\\DDA\\exp_data\\"

    df = pd.read_csv(os.path.join(path, "action_count_dist.csv"))
    # sns.barplot(data=df, x="Agent", y="total", hue="Action Group")
    # df = df[df['Action Group'] == 0]
    # fig, axs = plt.subplots(ncols=6)
    actions = [0, 1, 2, 3, 4, 5]
    action_group_name = ['NO_OP', 'MOVE', 'OUT_OF_CONTROL', 'NORMAL_ATTACK', 'SPECIAL_ATTACK', 'GUARD']

    for i, a in enumerate(actions):
        sub_df = df[df['Action Group'] == a]
        ax = sns.boxplot(data=sub_df, x='Action Group', y='count', hue='Agent')
        ax.get_legend().remove()
        ax.set_xlabel(None)
        ax.set(ylim=[0, 250])
        ax.set_xticks([])
        ax.set_xlabel(action_group_name[i])
        ax.set_ylabel(f'# of {action_group_name[i]}')
        plt.savefig(os.path.join(path, 'newfolder', f'{action_group_name[i]}.png'))
        plt.show()
    # plt.subplots_adjust(wspace=1)



def get_action_dist_per_agent2():
    agent_action = np.load('data/agent_action_groups_per_pred_state.npy')
    # print(agent_action)
    agents = agent_action.sum(axis=0)
    labels = ['move', 'guard', 'out of control', 'normal attack', 'command attack', 'special attack']
    x = np.array([i for i in range(0, 6*2, 2)])
    x2 = np.array([i for i in range(1, 6*2, 2)])
    com = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    results = {}
    # baseline = action_dist[0, :] / action_dist[0].sum()
    for i in com:
        agent = agents[i-1]
        sub_results = {'low':{}, 'high':{}}

        # cumulate count by low/high
        low_probs = np.zeros([55,])
        high_probs = np.zeros([55,])
        if i < 10:
            low_probs += agent[:5].sum(axis=0)
            high_probs += agent[5:].sum(axis=0)
        else:
            low_probs += agent[:10].sum(axis=0)
            high_probs += agent[10:].sum(axis=0)

        action_names = []
        group_cnt = [0, 0, 0, 0, 0, 0]
        for act_idx, act in enumerate(low_probs):
            group_cnt[Action.to_group(act_idx+1)] += act
        for group_idx, group in enumerate(group_cnt):
            sub_results['low'][Action.group_to_string(group_idx)] = group / sum(group_cnt) * 100
        sub_results['low']['group_cnt'] = sum(group_cnt)
            # action_names.append(Action.group_to_string(group_idx) + f'({group / sum(group_cnt) * 100:.2f}%)')
        # sub_results['low'] = action_names

        action_names = {}
        group_cnt = [0, 0, 0, 0, 0, 0]
        for act_idx, act in enumerate(high_probs):
            group_cnt[Action.to_group(act_idx+1)] += act
        for group_idx, group in enumerate(group_cnt):
            sub_results['high'][Action.group_to_string(group_idx)] = group / sum(group_cnt) * 100
        sub_results['high']['group_cnt'] = sum(group_cnt)
            # action_names.append(Action.group_to_string(group_idx) + f'({group / sum(group_cnt) * 100:.2f}%)')
        # sub_results['high'] = action_names

        # for j, prob in enumerate(agent):
        #     if not prob.any():
        #         frequent_actions = None
        #         # print('agent:', i, 'prob:', j, 'actions:', frequent_actions)
        #         sub_results[j*10] = [frequent_actions]
        #     else:
        #         # frequent_actions = prob.argsort()[-10:][::-1]
        #         # frequent_actions = frequent_actions[prob[frequent_actions] != 0]
        #         action_names = []
        #         group_cnt = [0, 0, 0, 0, 0, 0]
        #         for act_idx, act in enumerate(prob):
        #             group_cnt[Action.to_group(act_idx+1)] += act
        #         for group_idx, group in enumerate(group_cnt):
        #             action_names.append(Action.group_to_string(group_idx) + f'({group / prob.sum() * 100:.2f}%)')
        #         # print('agent:', i, 'prob:', j, 'actions:', action_names)
        #         sub_results[j*10] = action_names
        results[f'agent{i}'] = sub_results
    df = pd.DataFrame.from_dict(results).T
    # print(df)
            # print(prob)
    #     plt.bar(x, baseline)
    #     d = action_dist[i-1, :] / action_dist[i-1].sum()
    #     # plt.pie(d, labels=labels, autopct='%.1f%%')
    #     plt.bar(x2, d)
    #     plt.title(f'agent1 vs agent{i}')
    #     plt.savefig(os.path.join("D:\\Research\\Game\\DDA\\data\\analysis", f"same_agent{i}_pie.png"))
    #     plt.show()
    # df = pd.DataFrame(action_dist)
    # df.to_csv(os.path.join("D:\\Research\\Game\\DDA\\data\\analysis", f"agent_action_groups_per_pred_state_low_high.csv"))


if __name__ == '__main__':
    # get_action_dist()
    # get_action_dist_per_agent()
    analyze_action_statistics()