import numpy as np
import matplotlib.pyplot as plt

def plot_sample_trials(trials):
    """Plots 4 sample trials from the dataframe
    :param trials:
    :return:
    """
    n_trials = 4
    trials = trials.sample(n_trials)
    fig, axs = plt.subplots(2, n_trials, figsize=(3 * n_trials, 5))
    for ax, (idx, trial) in zip(axs.T, trials.iterrows()):
        plot_trial(ax, trial)
    plt.tight_layout()
    plt.show()
    return fig

def plot_attempts(trials):
    """Plots attempts as a function of ratio"""
    trials = trials.copy()
    trials['r_color'] = trials['n_target']/trials['n_items']
    fig, ax = plt.subplots(1, 3, figsize=(3 * 3, 4))

    ax[0].hist(trials['n_attempts'])
    ax[0].set_xlabel('no. of attempts')
    ax[0].set_ylabel('no. of trials')

    ax[2].plot(trials.groupby('r_color')['n_attempts'].mean())
    ax[2].set_xlabel('color ratio')
    ax[2].set_ylabel('no. of attempts')

    ax[1].plot(trials.groupby('n_items')['n_attempts'].mean())
    ax[1].set_xlabel('no. of items')
    ax[1].set_ylabel('no. of attempts')

    plt.tight_layout()
    return fig

def plot_trial(axs, trial):
    trial['occ_circle'].plot(axs[0])
    axs[0].set_title(f'target={trial["target_color"]}')
    for i, p in enumerate(trial['posterior']):
        axs[1].scatter(trial["theta"][i], np.amax(p), color='red', zorder=10)
        axs[1].plot(np.arange(len(p)), p, '-o', label=f'{i + 1}', zorder=0)
    axs[1].set_title('posterior')
    axs[1].legend(title='attempt')

