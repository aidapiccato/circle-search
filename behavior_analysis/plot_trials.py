import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import pandas as pd
import seaborn as sns

_COLORS = ['red', 'orange', 'yellow', 'limegreen', 'lightseagreen', 'skyblue', 'cornflowerblue', 'purple', 'pink']



def plot_sample_multi_region_trials(trials):
    trials = trials['trials']
    n_trials = 6
    trials = trials[trials['multi_region']]
    fig, axs = plt.subplots(2, int(n_trials / 2), figsize=(3 * (n_trials / 2), 5), subplot_kw=dict(projection="polar"))
    for ax, (idx, trial) in zip(axs.flat, trials.iterrows()):
        plot_trial(ax, trial)
    plt.tight_layout()
    plt.show()
    return fig


def plot_sample_trials(trials):
    """Plots 4 sample trials from the dataframe
    :param trials:
    :return:

    """
    trials = trials['trials']
    n_trials = 6
    trials = trials.sample(n_trials)
    fig, axs = plt.subplots(2, int(n_trials / 2), figsize=(3 * (n_trials / 2), 5), subplot_kw=dict(projection="polar"))
    for ax, (idx, trial) in zip(axs.flat, trials.iterrows()):
        plot_trial(ax, trial)
    plt.tight_layout()
    plt.show()
    return fig


def plot_dist_attempts(trials):
    trials = trials['trials']
    fig, ax = plt.subplots(1, 2, figsize=(3 * 2, 4))
    dist_theta = []
    dist_theta_attempts = [[] for attempt in range(trials['n_attempts'].max() - 1)]
    for idx, trial in trials.iterrows():
        dist_theta.append(2 * np.pi * trial['dist_theta'] / trial['n_items'])
        for attempt in range(len(trial['dist_theta'])):
            dist_theta_attempts[attempt].append(2 * np.pi * trial['dist_theta'][attempt] / trial['n_items'])
    dist_theta = [np.abs(dist) for sublist in dist_theta for dist in sublist]
    ax[0].hist(dist_theta)
    ax[0].axvline(np.mean(dist_theta), color='red')
    ax[0].set_xlabel('abs. distance between consecutive attempts (rad)')
    ax[0].set_ylabel('freq')

    dist_theta_attempts = [np.abs(np.asarray(sublist)) for sublist in dist_theta_attempts]
    ax[1].violinplot(dist_theta_attempts, np.arange(1, trials['n_attempts'].max()), showmeans=True)
    ax[1].set_xlabel('attempt')
    ax[1].set_ylabel('absolute dist. between adjacent thetas (rad)')
    plt.tight_layout()
    plt.show()
    return fig


def plot_attempts(trials):
    trials = trials['trials']
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
    theta = np.linspace(0, 2 * np.pi, endpoint=False, num=trial['n_items'])
    occ_circle = trial['occ_circle'][0]
    map_circle = trial['map_circle'][0]
    target_start = (np.flatnonzero(map_circle == trial['target_color'])[0] + trial['rot']) % trial['n_items']
    target_com = (target_start + (trial['n_target'] - 1) / 2) % trial['n_items']
    theta = theta - target_com * 2 * np.pi / trial['n_items']
    occ_circle_colors = [_COLORS[color] for color in occ_circle]
    max_p = 0
    min_p = 1.0
    angles = np.linspace(0, 2 * np.pi, endpoint=False, num=trial['n_items'])[
                 trial['theta']] - target_com * 2 * np.pi / trial['n_items']
    angle_colors = cm.get_cmap('viridis')(np.linspace(0.1, 1, num=len(angles), endpoint=True))
    if trial['posterior'] != 0:
        for i, p in enumerate(trial['posterior']):
            axs.plot(theta, p + 0.1, zorder=1, color=angle_colors[i])
            axs.scatter(angles[i], p.max() + 0.1, color=angle_colors[i], edgecolors='black', zorder=10)
            max_p = np.maximum(max_p, p.max())
            min_p = np.minimum(min_p, p.min())
    else:
        axs.scatter(angles, np.repeat(0.15, len(angles)), color=angle_colors, s=80, edgecolor='black')
    axs.scatter(theta, np.ones(trial['n_items']) * (max_p + 0.1), color=occ_circle_colors, s=50)
    axs.set_rmax(max_p + .2)
    axs.set_rmin(min_p - 0.2)
    axs.set_rticks([])
    axs.set_rmin(0)


def plot_statistics(trials):
    trials = trials['trials']
    fig, axs = plt.subplots(1, 3)

    axs[0].hist(trials['n_target'] / trials['n_items'], rwidth=0.75)
    axs[0].set_title('n_target/n_items')

    axs[1].hist(trials['frac_target'], rwidth=0.75)
    axs[1].set_title('frac_target')

    # axs[2].hist(trials['color_entropy'], rwidth=0.75)
    # axs[2].set_title('color_entropy')

    three_col_trials = trials[trials['n_colors'] == 3]
    denom = (three_col_trials['n_items'] - three_col_trials['n_target'])
    three_col_trials['other_colors_dist'] = np.vstack(
        [np.asarray(trial['color_dist'])[trial['other_colors'].astype(int)]
         for idx, trial in three_col_trials.iterrows()])[:, 0] / denom
    three_col_trials['true_frac_target'] = three_col_trials['n_target'] / three_col_trials['n_items']
    three_col_trials['true_frac_target_bin'] = pd.qcut(three_col_trials['true_frac_target'], q=4, labels=False)
    sns.stripplot(x='true_frac_target_bin', y='other_colors_dist', data=three_col_trials, ax=axs[2])
    plt.tight_layout()
    plt.show()
    return fig


def plot_attempts_histogram(trials):
    trials = trials['trials']
    p_target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_colors = [2, 3]
    fig, axs = plt.subplots(len(n_colors), len(p_target) - 1, figsize=(3 * len(p_target), 4 * len(n_colors)),
                            sharey='row')
    trials = trials.copy()
    trials = trials[trials['n_colors'].isin(n_colors)]

    for i, n_c in enumerate(n_colors):
        col_trials = trials[trials['n_colors'] == n_c]

        for ax, bin_idx in zip(axs[i], range(1, len(p_target))):
            group = col_trials[col_trials.frac_target.between(p_target[bin_idx - 1], p_target[
                bin_idx])][
                'n_attempts']
            max_attempts = group.max()
            ax.hist(group, density=True, bins=np.arange(1, max_attempts + 2) - 0.5, rwidth=0.75)
            ax.set_title(r'target size ratio $\in (%.1f, %.1f)$' % (p_target[bin_idx - 1], p_target[bin_idx]))
            ax.axvline(group.mean(), linestyle='--', color='red')
            ax.axhline(np.mean([p_target[bin_idx - 1], p_target[bin_idx]]), color='blue', linestyle='--')
            ax.set_xlabel('# attempts')
            ax.set_ylabel('% of trials')
            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # textstr = '\n'.join((
            #     r'$\mu=%.2f$' % (group.mean(),),
            #     r'$N = %d$' % (len(group),),
            # ))
            # ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            #         verticalalignment='top', bbox=props)

        axs[i][0].text(-0.18, 0.5 * (0.25 + 0.75), f'# colors = {n_c}',
                       horizontalalignment='right',
                       verticalalignment='center',
                       rotation='vertical',
                       transform=axs[i][0].transAxes)
    plt.tight_layout()
    plt.show()
    return fig


def plot_attempts_histogram_source_size(trials):
    trials = trials['attempts']
    trials['source_size'] = trials['source_size'] / trials['n_items']
    source_size_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_colors = [2, 3]
    fig, axs = plt.subplots(len(n_colors), len(source_size_bins) - 1,
                            figsize=(3 * len(source_size_bins), 4 * len(n_colors)),
                            sharey='row')
    trials = trials.copy()
    trials = trials[trials['n_colors'].isin(n_colors)]
    trials = trials[trials['n_attempts'] > 1]
    trials = trials[trials['remaining_attempts'] > 0]
    for i, n_c in enumerate(n_colors):
        col_trials = trials[trials['n_colors'] == n_c]
        for ax, bin_idx in zip(axs[i], range(1, len(source_size_bins))):
            group = \
                col_trials[col_trials.source_size.between(source_size_bins[bin_idx - 1], source_size_bins[bin_idx])][
                    'remaining_attempts']
            max_attempts = group.max()
            ax.hist(group, density=True, bins=np.arange(1, max_attempts + 2) - 0.5, rwidth=0.75)
            ax.set_title(r'source size ratio $\in (%.1f, %.1f)$' % (source_size_bins[bin_idx - 1], source_size_bins[
                bin_idx]))
            ax.axvline(group.mean(), linestyle='--', color='red')
            ax.set_xlabel('# remaining attempts')
            ax.set_ylabel('% of trials')
        axs[i][0].text(-0.18, 0.5 * (0.25 + 0.75), f'# colors = {n_c}',
                       horizontalalignment='right',
                       verticalalignment='center',
                       rotation='vertical',
                       transform=axs[i][0].transAxes)

    plt.tight_layout()
    plt.show()
    return fig


def plot_source_size_entropy(trials):
    trials = trials['attempts']
    trials = trials[trials['remaining_attempts'] > 0]
    trials = trials[trials['n_attempts'] > 1]
    trials['source_size'] = trials['source_size'] / trials['n_items']
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(x='source_size', y='posterior_entropies_norm', data=trials, ax=ax)
    bins = np.linspace(start=0, stop=1, num=20)
    trials['source_size_bin'] = pd.cut(trials['source_size'], bins=bins, labels=bins[1:])
    sns.lineplot(x='source_size_bin', y='posterior_entropies_norm', data=trials, estimator='mean', ax=ax)
    ax.set_xlabel('source size ratio')
    ax.set_ylabel('posterior entropy')
    plt.tight_layout()
    plt.show()
    return fig



def plot_source_size_remaining_attempts(trials):
    trials = trials['attempts']
    fig, ax = plt.subplots(1, 1)
    trials = trials[trials['remaining_attempts'] > 0]
    trials = trials[trials['n_attempts'] > 1]
    sns.scatterplot(x='source_size', y='remaining_attempts', data=trials, ax=ax, facecolor='white', edgecolor='black')
    bins = np.linspace(start=0, stop=1, num=20)
    trials['source_size_bin'] = pd.cut(trials['source_size'], bins=bins, labels=bins[1:])
    sns.lineplot(x='source_size_bin', y='remaining_attempts', data=trials, estimator='mean', ax=ax,
                 err_style='band', hue='n_colors')
    ax.set_xlabel('source size ratio')
    ax.set_ylabel('# remaining attempts')
    plt.tight_layout()
    plt.show()
    return fig


def plot_attempts_entropy(trials):
    trials = trials['attempts']
    trials = trials[trials['remaining_attempts'] > 0]
    trials = trials[trials['n_attempts'] > 1]
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(x='posterior_entropies_norm', y='remaining_attempts', data=trials, ax=ax,
                    hue='n_colors', facecolor='white', edgecolor='black')
    bins = np.linspace(start=0, stop=1, num=20)
    trials['posterior_entropies_norm_bin'] = pd.cut(trials['posterior_entropies_norm'], bins=bins, labels=bins[1:])
    sns.lineplot(x="posterior_entropies_norm_bin", y="remaining_attempts", data=trials, estimator="mean", ax=ax,
                 hue='n_colors')
    ax.set_xlabel('posterior entropy (bits)')
    ax.set_ylabel('# remaining attempts')
    plt.tight_layout()
    plt.show()
    return fig


def plot_attempts_entropy_histogram(trials):
    trials = trials['attempts']
    entropy_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_colors = [2, 3]
    fig, axs = plt.subplots(len(n_colors), len(entropy_bins) - 1,
                            figsize=(3 * len(entropy_bins), 4 * len(n_colors)),
                            sharey='row')
    trials = trials.copy()
    trials = trials[trials['n_colors'].isin(n_colors)]
    trials = trials[trials['n_attempts'] > 1]
    trials = trials[trials['remaining_attempts'] >= 1]
    for i, n_c in enumerate(n_colors):
        col_trials = trials[trials['n_colors'] == n_c]
        for ax, bin_idx in zip(axs[i], range(1, len(entropy_bins))):
            group = \
                col_trials[col_trials.posterior_entropies_norm.between(entropy_bins[bin_idx - 1], entropy_bins[
                    bin_idx])][
                    'remaining_attempts']
            max_attempts = group.max()
            ax.hist(group, density=True, bins=np.arange(1, max_attempts + 2) - 0.5, rwidth=0.75)
            ax.set_title(r'posterior entropy $\in (%.1f, %.1f)$' % (entropy_bins[bin_idx - 1], entropy_bins[bin_idx]))
            ax.axvline(group.mean(), linestyle='--', color='red')
            ax.set_xlabel('# remaining attempts')
            ax.set_ylabel('% of trials')
        axs[i][0].text(-0.18, 0.5 * (0.25 + 0.75), f'# colors = {n_c}',
                       horizontalalignment='right',
                       verticalalignment='center',
                       rotation='vertical',
                       transform=axs[i][0].transAxes)

    plt.tight_layout()
    plt.show()
    return fig


def plot_attempts_source_size(trials):
    trials = trials['trials']
    fig, axs = plt.subplots(1, 1)
    trials = trials[trials['remaining_attempts'] > 0]
    trials = trials[trials['n_attempts'] > 1]
    source_sizes = np.hstack([np.asarray(trial['source_size']) / trial['n_items'] for idx, trial in trials.iterrows()])
    # source_sizes_binned = pd.qcut(source_sizes)
    remaining_attempts = np.hstack(trials['remaining_attempts'])
    non_neg = np.flatnonzero(remaining_attempts > 0)
    source_sizes = source_sizes[non_neg]
    remaining_attempts = remaining_attempts[non_neg]
    sns.scatterplot(source_sizes, remaining_attempts, ax=axs)
    axs.set_xlabel('source size')
    axs.set_ylabel('number of remaining attempts')

    plt.tight_layout()
    plt.show()
    return fig
