import numpy as np


def get_features(trials):
    # ratio of target items to
    com_target = []
    dist_theta = []
    dist_target = []
    trials['n_target'] = [len(np.flatnonzero(trial['occ_circle'] == trial['target_color'])) for _, trial in
                          trials.iterrows()]
    for idx, trial in trials.iterrows():
        target_start = (np.flatnonzero(trial['map_circle'] == trial['target_color'])[0] + trial['rot']) % trial[
            'n_items']
        target_com = (target_start + (trial['n_target'] - 1) / 2) % trial['n_items']
        dist_theta.append(np.diff(trial['theta']))
        com_target.append(target_com)
        dist_target.append(trial['theta'] - com_target[-1])
    trials['com_target'] = com_target
    trials['dist_target'] = dist_target
    trials['dist_theta'] = dist_theta
    return trials
