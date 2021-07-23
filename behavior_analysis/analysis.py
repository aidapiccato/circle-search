import numpy as np
from scipy.stats import entropy
import pandas as pd
def get_features(trials):
    # ratio of target items to
    com_target = []
    dist_theta = []
    dist_target = []
    color_dists = []
    entropies = []
    other_colors = []
    source_size = []
    posterior_entropies = []
    multi_region = []
    posterior_entropies_norm = []
    remaining_attempts = []
    trials['n_target'] = [len(np.flatnonzero(trial['occ_circle'] == trial['target_color'])) for _, trial in
                          trials.iterrows()]

    add_posterior = 'posterior' in trials.columns
    for idx, trial in trials.iterrows():
        sizes = []
        rem_attempts = []
        for idx, attempt in enumerate(trial['theta']):
            sizes.append(len(np.flatnonzero(trial['map_circle'][0] == trial['map_circle'][0][attempt])))
            rem_attempts.append(trial['n_attempts'] - idx - 1)
        remaining_attempts.append(rem_attempts)
        source_size.append(sizes)
        target_start = (np.flatnonzero(trial['map_circle'][0] == trial['target_color'])[0] + trial['rot']) % trial[
            'n_items']
        color_dist = []
        other_colors.append(np.arange(trial['n_colors'])[np.flatnonzero(np.arange(trial['n_colors']) != trial[
            'target_color'])])
        for color in range(trial['n_colors']):
            color_dist.append(len(np.flatnonzero(trial['occ_circle'][0] == color)))
        post_entropies = []
        if add_posterior:
            for p in trial['posterior']:
                post_entropies.append(entropy(p))
            max_entropy = entropy(np.ones(trial['n_items'])/trial['n_items']) # TODO: Entropy should also decrease over
            # each attempt as a result of disqualifying number of views.
            posterior_entropies_norm.append(post_entropies/max_entropy)
        target_com = (target_start + (trial['n_target'] - 1) / 2) % trial['n_items']
        color_dists.append(color_dist)
        dist_theta.append(np.diff(trial['theta']))
        com_target.append(target_com)
        if 'n_regions' in trials.columns:
            multi_region.append(trial['n_regions'] > trial['n_colors'])
        entropies.append(entropy(color_dist))
        posterior_entropies.append(post_entropies)
        dist_target.append(trial['theta'] - com_target[-1])
    if 'n_regions' in trials.columns:
        trials['multi_region'] = multi_region
    trials['com_target'] = com_target
    trials['remaining_attempts'] = remaining_attempts
    trials['color_entropy'] = entropies
    if add_posterior:
        trials['posterior_entropies'] = posterior_entropies
        trials['posterior_entropies_norm'] = posterior_entropies_norm
    else:
        trials['posterior'] = np.zeros(len(trials))
        trials['posterior_entropies'] = np.zeros(len(trials))
        trials['posterior_entropies_norm'] = np.zeros(len(trials))
    trials['other_colors'] = other_colors
    trials['color_dist'] = color_dists
    trials['dist_target'] = dist_target
    trials['dist_theta'] = dist_theta
    trials['source_size'] = source_size
    attempts_df = _get_attempts_dataframe(trials)
    return dict(trials=trials, attempts=attempts_df)

def _get_attempts_dataframe(trials):
    trials = trials.reset_index(drop=False)  # adding trial number column
    trials_expl = trials[['remaining_attempts', 'source_size', 'index', 'theta', 'color_entropy', 'posterior',
                          'posterior_entropies',  'posterior_entropies_norm']]
    trials_expl = trials_expl.set_index('index')[['source_size', 'theta', 'remaining_attempts', 'posterior',
                                                  'posterior_entropies', 'posterior_entropies_norm']].apply(
        pd.Series.explode).reset_index(drop=False)

    trials_expl = trials_expl.join(trials[['n_items', 'n_colors', 'target_color', 'n_attempts']], on='index')
    trials_expl['posterior_entropies_norm'] = trials_expl['posterior_entropies_norm'].astype(float)
    trials_expl['remaining_attempts'] = trials_expl['remaining_attempts'].astype(int)
    return trials_expl
