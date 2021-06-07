import os
import pickle
import fnmatch
import pandas as pd


def get_trials(dir, filter='output'):
    all_trials = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, filter):
            with open(os.path.join(root, filename), 'rb') as f:
                all_trials = all_trials + pickle.load(f)
    return all_trials


def get_trials_dataframe(trials):
    return pd.DataFrame(trials)
