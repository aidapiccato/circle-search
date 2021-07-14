import os
import pickle
import fnmatch
import json
import pandas as pd
import numpy as np
from task import CircleSearch

_META_STATE_CONTACTING_SEARCH_TRANSITION = 1


def _get_trials_list(dir, filter='output'):
    # TODO: Make this a utils file
    all_trials = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, filter):
            with open(os.path.join(root, filename), 'rb') as f:
                all_trials += pickle.load(f)
    return all_trials


def _extract_kv(key, val, arr):
    """Recursively unpacks item to turn a single key value pair. Meant to unpack dictionary value entries
    """
    if type(val) is dict:
        for nested_key in val.keys():
            nested_val = val[nested_key]
            arr = _extract_kv(nested_key, nested_val, arr)
    else:
        arr.append((key, val))
    return arr


def _get_meta_state_dict(meta_states):
    trials = []

    for trial in meta_states:
        step_dict = {}
        for step in trial:
            for item in step:
                kv_arr = _extract_kv(item[0], item[1], [])
                for key, val in kv_arr:
                    if key not in step_dict:
                        step_dict[key] = []
                    step_dict[key].append(val)
        trials.append(step_dict)
    return trials


def _get_state_dict(states):
    trials = []
    for trial in states:
        state_dict = {}
        for step in trial:
            for sprite, sprite_attr in step:
                if sprite not in state_dict.keys():
                    state_dict[sprite] = []
                state_dict[sprite].append(sprite_attr)
        trials.append(state_dict)
    return trials


def _unpack_moog(trials):
    """Unpacking list of MOOG trials into state and step dictionaries
    :param trials: List of trial objects
    :return:
    """
    meta_states = []
    states = []
    for trial in trials:
        trial_meta_state, trial_state = [], []
        for step in trial:
            trial_meta_state.append(step[:5])
            trial_state.append(step[-1])
        meta_states.append(trial_meta_state)
        states.append(trial_state)
    meta_states_dict = _get_meta_state_dict(meta_states)
    states_dict = _get_state_dict(states)
    trials_dicts = []
    for meta_state, state in zip(meta_states_dict, states_dict):
        meta_state.update(state)
        trials_dicts.append(meta_state)
    return trials_dicts


def _get_attempt_thetas(trial):
    phase_transitions = np.diff(trial['phase'])
    contacting_to_search = np.flatnonzero(phase_transitions == _META_STATE_CONTACTING_SEARCH_TRANSITION).astype(int)
    thetas = []
    for idx in contacting_to_search:
        diff_thetas = (set(trial['contacted_fruits'][idx + 1])).difference(set(thetas))
        thetas += list(diff_thetas)
    return list(thetas)


def _get_attempt_times(trial):
    phase_transitions = np.diff(trial['phase'])
    return np.asarray(trial['time'])[
        np.flatnonzero(phase_transitions == _META_STATE_CONTACTING_SEARCH_TRANSITION).astype(int)]


def _get_circle_search_obj(trial):
    return CircleSearch(n_colors=trial['n_colors'][0][0], n_items=trial['n_items'][0][0],
                        n_attempts=trial['n_attempts'][0], map_circle=trial['map_circle'][0][0],
                        occ_circle=trial['occ_circle'][0][0], target_color=trial['target_color'][0])


def _get_trial_info(trial):
    keys = ['frac_target', 'n_colors', 'n_items', 'n_regions',  'target_color', 'map_circle', 'occ_circle', 'rot']
    trial_info = {}
    for key in keys:
        trial_info[key] = np.squeeze(trial[key])[0]
    return trial_info


def _get_dataframe_rows(trial_dicts):
    trials = []
    for trial in trial_dicts:
        attempt_times = _get_attempt_times(trial)
        attempt_thetas = _get_attempt_thetas(trial)
        trial_info = _get_trial_info(trial)
        trial_dict = {'attempt_times': attempt_times, 'n_attempts': len(attempt_times), 'theta': attempt_thetas}
        trial_dict.update(trial_info)
        trials.append(trial_dict)
    return trials

def get_trials_dataframe(dir):
    filename_filter = 'output'
    trials = _get_trials_list(dir, filename_filter)
    return pd.DataFrame(trials)
