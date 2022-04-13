from glob import glob
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns
from const.action import Action


def geq_refactor():
    geq_path = "D:\\Research\\Game\\DDA\\data\\pre-collected_user_data_for_training\\total_survey.csv"
    df = pd.read_csv(geq_path)
    level = []
    game_order = []
    for _, row in df.iterrows():
        isna = row.isna()
        if not isna['Level1']:
            level.append(1)
            game_order.append(row['Level1'])
        elif not isna['Level2']:
            level.append(2)
            game_order.append(row['Level2'])
        elif not isna['Level3']:
            level.append(3)
            game_order.append(row['Level3'])
        elif not isna['Level4']:
            level.append(4)
            game_order.append(row['Level4'])
    df['Level'] = level
    df['game_order'] = game_order
    df = df.drop(['Level1', 'Level2', 'Level3', 'Level4'], axis=1)
    print(df)

    df.to_csv("D:\\Research\\Game\\DDA\\data\\pre-collected_user_data_for_training\\total_survey_refactor.csv")


def geq_personal_analysis():
    geq_path = "D:\\Research\\Game\\DDA\\data\\pre-collected_user_data_for_training\\total_survey_refactor.csv"
    df = pd.read_csv(geq_path)
    df = df.drop(['Q3', 'Q7'], axis=1)
    df['PX'] = df.iloc[:, 2:-2].mean(axis=1)
    print(df)


def geq_analysis():
    geq_path = "D:\\Research\\Game\\DDA\\data\\pre-collected_user_data_for_training\\labeling.csv"
    df = pd.read_csv(geq_path)
    df['CH_Level'] = 5 - df['CH_Level']
    df['Bo_Level'] = 5 - df['Bo_Level']
    df['An_Level'] = 5 - df['An_Level']

    df['PX'] = df.iloc[:, 3:].mean(axis=1)
    lv1_px = df[df['Level'] == 1]['PX'].mean()
    lv2_px = df[df['Level'] == 2]['PX'].mean()
    lv3_px = df[df['Level'] == 3]['PX'].mean()
    lv4_px = df[df['Level'] == 4]['PX'].mean()

    df['rank'] = df.groupby('UserNum')['PX'].rank()
    # print(lv1_px, lv2_px, lv3_px, lv4_px)
    # print(df[df['Level'] == 1]['PX'].tolist())
    new_df = pd.DataFrame({
        'Lv1': df[df['Level'] == 1]['PX'].tolist(),
        'Lv2': df[df['Level'] == 2]['PX'].tolist(),
        'Lv3': df[df['Level'] == 3]['PX'].tolist(),
        'Lv4': df[df['Level'] == 4]['PX'].tolist(),
    })
    # print(new_df)
    # ax = sns.displot(new_df, kind='kde', bw_adjust=0.5, fill=True)
    ax = sns.boxplot(x='Level', y='PX', data=df, hue='Level')
    # ax.set(xlabel='PX', ylabel='ratio')

    plt.savefig("D:\\Research\\Game\\DDA\\data\\pre-collected_user_data_for_training\\px_box.png")
    plt.show()


def parse_user_actions():
    root_path = "D:\\Research\\Game\\DDA\\exp_data\\"
    game_log_path = "D:\\Research\\Game\\DDA\\data\\dataset\\final_user_test_data\\"
    state_df = pd.read_csv(os.path.join(root_path, 'State&Action-Predict-15frame-2.csv'))
    state_df['Player Action'] = 1
    state_df['Player Action Group'] = 1
    partcs = [f'p{i}' for i in range(1, 21)]
    agents = [f'Agent{i}' for i in range(1, 13)]
    for partc in partcs:
        for agent in agents:
            print(partc, agent)
            times = state_df[(state_df['partc'] == partc) & (state_df['Agent'] == agent)]['time'].unique()
            for time in times:
                date_obj = datetime.strptime(time, '%Y-%m-%d %H:%M')
                converted_time = datetime.strftime(date_obj, '%Y.%m.%d-%H.%M')

                data_path = glob(os.path.join(game_log_path, partc, 'features', f'*{agent}_{converted_time}*.csv'))
                # print(partc, agent, converted_time)
                player_log = pd.read_csv(data_path[0])
                frames = state_df[(state_df['partc'] == partc) & (state_df['Agent'] == agent) & (state_df['time'] == time)]['frame'].unique()

                # state_df[(state_df['partc'] == partc)
                #          & (state_df['Agent'] == agent)
                #          & (state_df['time'] == time), 'Player Action'] \
                #         = player_log[player_log['current_frame'].isin(frames)]['self_action_id']
                for frame in frames:
                    state_df.loc[(state_df['partc'] == partc)
                             & (state_df['Agent'] == agent)
                             & (state_df['time'] == time)
                             & (state_df['frame'] == frame), 'Player Action'] \
                        = int(player_log.loc[(player_log['current_frame'] == frame), 'self_action_id'])
                    state_df.loc[(state_df['partc'] == partc)
                             & (state_df['Agent'] == agent)
                             & (state_df['time'] == time)
                             & (state_df['frame'] == frame), 'Player Action Group'] \
                        = Action.to_group(int(player_log.loc[(player_log['current_frame'] == frame), 'self_action_id']))
    state_df.to_csv(os.path.join(root_path, 'State&Action-Predict-15frame-4.csv'))

def analyze_user_state():
    root_path = "D:\\Research\\Game\\DDA\\exp_data\\player_experience(GEQ)3.csv"
    save_path = "D:\\Research\\Game\\DDA\\exp_data\\"
    df = pd.read_csv(root_path)
    df = df[(df['game'] != 3) & (df['game'] != 5) & (df['game'] != 7) & (df['game'] != 9) & (df['game'] != 10)]

    target_states = [1, 2, 4, 6, 8]
    result = {
        'challenge': {},
        'competence': {},
        'valence': {},
        'flow': {},
        'GX': {},
    }
    for i in range(1, 21):
        user_data = df[df['userNum'] == i]
        result['challenge'][f'P{i}'] = {}
        result['competence'][f'P{i}'] = {}
        result['valence'][f'P{i}'] = {}
        result['flow'][f'P{i}'] = {}
        result['GX'][f'P{i}'] = {}
        for state in target_states:
            game_data = user_data[user_data['game'] == state]
            challenge = game_data['challenge'].mean()
            competence = game_data['competence'].mean()
            valence = game_data['valence'].mean()
            flow = game_data['Flow'].mean()
            gx = game_data['gameExperience'].mean()
            result['challenge'][f'P{i}'][f'agent{state}'] = challenge
            result['competence'][f'P{i}'][f'agent{state}'] = competence
            result['valence'][f'P{i}'][f'agent{state}'] = valence
            result['flow'][f'P{i}'][f'agent{state}'] = flow
            result['GX'][f'P{i}'][f'agent{state}'] = gx
        result['challenge'][f'P{i}']['mean'] = user_data['challenge'].mean()
        result['competence'][f'P{i}']['mean'] = user_data['competence'].mean()
        result['valence'][f'P{i}']['mean'] = user_data['valence'].mean()
        result['flow'][f'P{i}']['mean'] = user_data['Flow'].mean()
        result['GX'][f'P{i}']['mean'] = user_data['gameExperience'].mean()

    states = result.keys()
    final = []
    conv = {
        'agent1': 'baseline',
        'agent2': 'Ch-u',
        'agent4': 'Co-u',
        'agent6': 'Va-u',
        'agent8': 'Fl-u',
        'mean': 'mean',
    }
    for state in states:
        df = pd.DataFrame(result[state]).T
        df.to_csv(os.path.join(save_path, state + '_2.csv'))
    #     players = result[state].keys()
    #     for player in players:
    #         games = result[state][player].keys()
    #         for game in games:
    #             final.append([player, conv[game], state, result[state][player][game]])
    #
    # column = ['player', 'game', 'state', 'value']
    # final_df = pd.DataFrame(final, columns=column)
    # final_df.to_csv(os.path.join(save_path, 'player_personal_state_analyze.csv'))


def draw_box_plot():
    root_path = "D:\\Research\\Game\\DDA\\exp_data\\"
    total_state = pd.read_csv(os.path.join(root_path, 'player_personal_state_analyze.csv'))
    games = ['Ch-u', 'Co-u', 'Va-u', 'Fl-u']
    states = ['challenge', 'competence', 'valence', 'flow']

    fig, axs = plt.subplots(ncols=4)
    colors = [
        ['#FFE400', '#FF2424', '#8C8C8C'],
        ['#FFE400', '#0054FF', '#8C8C8C'],
        ['#FFE400', '#2FED28', '#8C8C8C'],
        ['#FFE400', '#FF24FF', '#8C8C8C'],
    ]
    hatches = ['\\\\', '..', '//']

    for i, game in enumerate(games):
        a = total_state[(total_state['game'] == 'baseline')
                        | (total_state['game'] == game)
                        | (total_state['game'] == 'mean')]
        a = a[a['state'] == states[i]]
        print(a)
        sns.set_palette(sns.color_palette(colors[i]))
        ax = sns.boxplot(x='state', y='value', hue='game', data=a, ax=axs[i])
        ax.get_legend().remove()
        ax.set_xlabel(game)
        ax.set_ylabel(None)
        ax.set(ylim=(1, 7))
        for j, patch in enumerate(ax.artists):
            patch.set_hatch(hatches[j])
            patch.set_edgecolor('#BDBDBD')
        # plt.savefig(os.path.join(root_path, 'box_plot', f'{game}.png'))
    plt.subplots_adjust(wspace=0.3)
    plt.show()


def draw_box_plot3():
    root_path = "D:\\Research\\Game\\DDA\\exp_data\\"
    challenge = pd.read_csv(os.path.join(root_path, 'challenge_2.csv'))
    competence = pd.read_csv(os.path.join(root_path, 'competence_2.csv'))
    valence = pd.read_csv(os.path.join(root_path, 'valence_2.csv'))
    flow = pd.read_csv(os.path.join(root_path, 'flow_2.csv'))
    gx = pd.read_csv(os.path.join(root_path, 'GX_2.csv'))

    states = ['competence', 'valence', 'flow']
    state_map = {
        'challenge': challenge,
        'competence': competence,
        'valence': valence,
        'flow': flow,
    }
    game_map = {
        'challenge': 'agent2',
        'competence': 'agent4',
        'valence': 'agent6',
        'flow': 'agent8',
    }

    result = []
    for state in states:
        game = state_map[state]
        state_std = game[game_map[state]].std() / 2
        gx_std = gx[game_map[state]].std()

        for i in range(1, 21):
            p_i_state = float(game[game['player'] == f'P{i}'][game_map[state]])
            p_i_gx = float(gx[gx['player'] == f'P{i}'][game_map[state]])
            for j in range(i, 21):
                p_j_state = float(game[game['player'] == f'P{j}'][game_map[state]])
                p_j_gx = float(gx[gx['player'] == f'P{j}'][game_map[state]])

                if abs(p_i_state - p_j_state) <= state_std and abs(p_i_gx - p_j_gx) > gx_std:
                    print(state, f'P{i}: {p_i_state}|{p_i_gx}', f'P{j}: {p_j_state}|{p_j_gx}', f'{state_std}|{gx_std}')
                    result.append([state, f'P{i}',p_i_state, p_i_gx, f'P{j}', p_j_state, p_j_gx, state_std, gx_std])
    header = ['state', 'player1', 'player1_state_value', 'player1_gx_value',
              'player2', 'player2_state_value', 'player2_gx_value',
              'std_state_value', 'std_gx_value']
    df = pd.DataFrame(result, columns=header)
    df.to_csv(os.path.join(root_path, 'sim_diff_list.csv'))


def draw_box_plot2():
    root_path = "D:\\Research\\Game\\DDA\\exp_data\\"
    total_state = pd.read_csv(os.path.join(root_path, 'player_experience_per_state.csv'))
    baseline = total_state[total_state['game'] == 1]
    states = total_state[(total_state['game'] == 2)
                         | (total_state['game'] == 4)
                         | (total_state['game'] == 6)
                         | (total_state['game'] == 8)
                         | (total_state['game'] == 11)
                         | (total_state['game'] == 12)]

    games = {
        1: 'baseline',
        2: 'Ch-u',
        4: 'Co-u',
        6: 'Va-u',
        8: 'Fl-u',
        11: 'Ch-u_Co-u',
        12: 'Va-u_Fl-u',
    }
    game_names = []
    for i in range(len(states)):
       game_names.append(games[int(states.iloc[i]['game'])])
    states['game_name'] = game_names
    tmp_base = []
    compare_games = ['Ch-u', 'Co-u', 'Va-u', 'Fl-u', 'Ch-u_Co-u', 'Va-u_Fl-u']
    for row in baseline.itertuples(index=False):
        tmp_base.extend([list(row)]*6)
    compare_games *= 40
    new_base = pd.DataFrame(tmp_base, columns=baseline.columns)
    new_base['game_name'] = compare_games
    new_total = pd.concat([states, new_base])

    # fig = plt.figure(figsize=(15, 13))
    fig, axs = plt.subplots(ncols=6)
    compare_games = ['Ch-u', 'Co-u', 'Va-u', 'Fl-u', 'Ch-u_Co-u', 'Va-u_Fl-u']
    colors = [
        ['#FFF612', '#FF2424'],
        ['#FFF612', '#0054FF'],
        ['#FFF612', '#2FED28'],
        ['#FFF612', '#FF24FF'],
        ['#FFF612', '#5F00FF'],
        ['#FFF612', '#805B56'],
    ]
    for i, compare_game in enumerate(compare_games):
        game = new_total[new_total['game_name'] == compare_game]
        sns.set_palette(sns.color_palette(colors[i]))
        ax = sns.boxplot(x='game_name', y='gameExperience', hue='game', data=game, ax=axs[i])
        ax.get_legend().remove()
        ax.set_xlabel(None)
        ax.set(ylim=(1, 7))
    plt.subplots_adjust(wspace=1.3)
    plt.show()


from matplotlib.patches import PathPatch
def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """
    # iterating through Axes instances
    for ax in g.axes:
        # iterating through axes artists:
        for c in ax.get_children():
            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)
                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new
                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def draw_line_plot():
    root_path = "D:\\Research\\Game\\DDA\\exp_data\\"
    # df = pd.read_csv(os.path.join(root_path, 'state_with_action_analysis.csv'))
    df = pd.read_csv(os.path.join(root_path, 'State&Action-Predict-15frame-4.csv'))
    partcs = {
        'p18': 'Agent4',
        'p6': 'Agent4',
        'p15': 'Agent6',
        'p19': 'Agent6',
        'p5': 'Agent8',
        'p12': 'Agent8',
    }
    game_map = {
        'Agent4': 'Competence',
        'Agent6': 'Valence',
        'Agent8': 'Flow',
    }

    color_codes = ['#0070C0', '#00B050', '#7F7F7F', '#ED7D31', '#FF0000', '#B17ED8']
    state_colors = {'Challenge': '#9E480E',
                    'Competence': '#FFC000',
                    'Valence': '#70AD47',
                    'Flow': '#ED7D31'}
    cmap = colors.ListedColormap(color_codes)
    for partc, agent in partcs.items():
        part_df = df[(df['partc'] == partc) & (df['Agent'] == agent)]
        target_state = game_map[agent]
        times = part_df['time'].unique()
        time1 = datetime.strptime(times[0], "%Y-%m-%d %H:%M")
        time2 = datetime.strptime(times[1], "%Y-%m-%d %H:%M")
        if time1 < time2:
            part_df = part_df.replace(times[0], 1)
            part_df = part_df.replace(times[1], 2)
        else:
            part_df = part_df.replace(times[0], 2)
            part_df = part_df.replace(times[1], 1)

        for i in range(2):
            fig, axs = plt.subplots(nrows=3)
            part_sess_df = part_df[part_df['time'] == i+1]
            # part_sess_df['Action Group'] = part_sess_df['Action Group'].replace(1, 10)
            # part_sess_df['Action Group'] = part_sess_df['Action Group'].replace(2, 1)
            # part_sess_df['Action Group'] = part_sess_df['Action Group'].replace(10, 2)
            # part_sess_df['Player Action Group'] = part_sess_df['Player Action Group'].replace(1, 10)
            # part_sess_df['Player Action Group'] = part_sess_df['Player Action Group'].replace(2, 1)
            # part_sess_df['Player Action Group'] = part_sess_df['Player Action Group'].replace(10, 2)

            act_ratio = part_sess_df.groupby(['Action Group'])['Action Group'].count() / len(part_sess_df) * 100
            # act_ratio['Action Group'] = act_ratio['Action Group'].replace()
            act_ratio = act_ratio.cumsum()
            # act_ratio = []
            # for j in range(6):
            #     act_cnt = part_sess_df[part_sess_df['Action Group'] == j]
            #     act_ratio.append(len(act_cnt) / len(part_sess_df) * 100)
            heatmap_x = np.linspace(0, part_sess_df['frame'].max())
            extent = [heatmap_x[0] - (heatmap_x[1] - heatmap_x[0])/2.,
                      heatmap_x[-1]+(heatmap_x[1] - heatmap_x[0])/2.,
                      0, 1]
            part_sess_df[target_state] = part_sess_df[target_state]#.rolling(7).sum()
            sns.lineplot(x='frame', y=target_state, data=part_sess_df, ax=axs[0], color=state_colors[target_state])
            # print(part_sess_df)
            axs[0].set(ylim=(0.2, 1.))
            axs[1].imshow(part_sess_df['Action Group'].to_numpy()[np.newaxis, :], aspect=150., extent=extent, cmap=cmap)
            axs[1].set_yticks([])
            axs[2].imshow(part_sess_df['Player Action Group'].to_numpy()[np.newaxis, :], aspect=150., extent=extent, cmap=cmap)
            axs[2].set_yticks([])

            plt.savefig(os.path.join(root_path, 'line_graph', f'{partc}_sess{i+1}_{target_state}.png'))
            plt.show()


if __name__ == '__main__':
    # analyze_user_state()
    # draw_line_plot()
    # parse_user_actions()
    # geq_analysis()
    geq_personal_analysis()
