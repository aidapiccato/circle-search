"""Basic configuration file.

The get_config() method in this file returns a dictionary that can be passed to
python_utils.build_from_config.build_from_config().
"""

import task
import models
import numpy as np

def get_config():
    """Get config for main.py."""
    config = {

        'constructor': task.Driver,
        'kwargs': {
            'n_trials': 10,
            'task_constructor':  task.CircleSearch,
            'task_kwargs': {
                    'n_colors': 3,
                    'n_regions': 5,
                    'n_items': 16,
                    'frac_target': 0.25,
                    'n_attempts': 100,
            },
            'model_constructor': models.IdealObserver,
        },
    }

    return config
