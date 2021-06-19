import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
_COLORS = ['red', 'orange', 'yellow', 'limegreen', 'lightseagreen', 'skyblue', 'cornflowerblue', 'purple', 'pink']
def plot_sample_trials(trials):
    """Plots 4 sample trials from the dataframe
    :param trials:
    :return:
    """
    n_trials = 10
    trials = trials.sample(n_trials)
    fig, axs = plt.subplots(2, int(n_trials/2), figsize=(3 * (n_trials/2), 5), subplot_kw=dict(projection="polar"))
    for ax, (idx, trial) in zip(axs.flat, trials.iterrows()):
        plot_trial(ax, trial)
    plt.tight_layout()
    plt.show()
    return fig


def plot_dist_attempts(trials):
    fig, ax = plt.subplots(1, 2, figsize=(3*2, 4))
    dist_theta = []
    dist_theta_attempts = [[] for attempt in range(trials['n_attempts'].max()-1)]
    for idx, trial in trials.iterrows():
        dist_theta.append(2 * np.pi * trial['dist_theta']/trial['n_items'])
        for attempt in range(len(trial['dist_theta'])):
            dist_theta_attempts[attempt].append(2 * np.pi * trial['dist_theta'][attempt]/trial['n_items'])
    dist_theta = [np.abs(dist) for sublist in dist_theta for dist in sublist]
    ax[0].hist(dist_theta)
    ax[0].axvline(np.mean(dist_theta), color='red')
    ax[0].set_xlabel('abs. distance between consecutive attempts (rad)')
    ax[0].set_ylabel('freq')

    dist_theta_attempts = [np.abs(np.asarray(sublist)) for sublist in dist_theta_attempts]
    print(dist_theta_attempts)
    ax[1].violinplot(dist_theta_attempts, np.arange(1, trials['n_attempts'].max()), showmeans=True)
    ax[1].set_xlabel('attempt')
    ax[1].set_ylabel('absolute dist. between adjacent thetas (rad)')
    plt.tight_layout()
    plt.show()
    return fig

def plot_attempts(trials):
    """Plots attempts as a function of ratio"""
    trials = trials.copy()
    fig, ax = plt.subplots(1, 4, figsize=(3 * 4, 4))
    y_min, y_max = [0, trials['n_attempts'].max()]
    for a in ax[1:]:
        a.set_ylim([y_min, y_max])

    ax[0].hist(trials['n_attempts'])
    ax[0].axvline(trials['n_attempts'].mean())
    ax[0].set_xlabel('no. of attempts')
    ax[0].set_ylabel('no. of trials')

    ax[2].plot(trials.groupby('frac_target')['n_attempts'].mean())
    ax[2].set_xlabel('target ratio')
    ax[2].set_ylabel('no. of attempts')

    ax[1].plot(trials.groupby('n_items')['n_attempts'].mean())
    ax[1].set_xlabel('no. of items')
    ax[1].set_ylabel('no. of attempts')

    ax[3].plot(trials.groupby('n_colors')['n_attempts'].mean())
    ax[3].set_xlabel('no. of colors')
    ax[3].set_ylabel('no. of attempts')

    plt.tight_layout()
    return fig

def plot_trial(axs, trial):
    theta = np.linspace(0, 2 * np.pi, endpoint=False, num = trial['n_items'])
    occ_circle = trial['occ_circle']
    map_circle = trial['map_circle']
    target_start = (np.flatnonzero(map_circle == trial['target_color'])[0] + trial['rot']) % trial['n_items']
    target_com = (target_start + (trial['n_target']-1)/2) % trial['n_items']
    theta = theta - target_com*2*np.pi/trial['n_items']
    occ_circle_colors = [_COLORS[color] for color in occ_circle]
    max_p = 0
    angles = np.linspace(0, 2 * np.pi, endpoint=False, num=trial['n_items'])[
                 trial['theta']] - target_com * 2 * np.pi / trial['n_items']
    angle_colors = cm.get_cmap('gray')(np.linspace(0, 1, num=len(angles), endpoint=True))
    for i, p in enumerate(trial['posterior']):
        axs.plot(theta, p + 0.1, zorder=1)
        axs.scatter(angles[i], p.max() + 0.1, color=angle_colors[i], edgecolors='black', zorder=10)
        max_p = np.maximum(max_p, p.max())
    axs.scatter(theta, np.ones(trial['n_items']) * (max_p + 0.2), color=occ_circle_colors, s=50)
    axs.set_rmax(max_p + .3)
    axs.set_rticks([])
    axs.set_rmin(0)


def plot_attempts_histogram(trials):
    p_target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_colors = [2, 3]
    fig, axs = plt.subplots(len(n_colors), len(p_target), figsize=(3 * len(p_target), 4 * len(n_colors)),
                            sharey='row')
    trials = trials.copy()
    trials = trials[trials['n_colors'].isin(n_colors)]
    trials = trials[trials['frac_target'].isin(p_target)]

    for i, n_c in enumerate(n_colors):
        col_trials = trials[trials['n_colors'] == n_c]
        grouped = col_trials.groupby('frac_target')['n_attempts']
        max_attempts = col_trials['n_attempts'].max()
        for ax, (p, group) in zip(axs[i], grouped):
            ax.hist(group, density=True, bins=np.arange(max_attempts)-0.5, rwidth=0.75)
            ax.set_title(f'p = {p}')
            ax.axhline(p, linestyle='--')
            ax.set_xlabel('attempts')
            ax.set_ylabel('% of trials')
            ax.set_xticks(np.arange(1, max_attempts, 2))
    return fig


