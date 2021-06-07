"""Basic configuration file.

The get_config() method in this file returns a dictionary that can be passed to
python_utils.build_from_config.build_from_config().
"""

import task
import models

def get_config():
    """Get config for main.py."""
    config = {

        'constructor': task.Driver,
        'kwargs': {
            'n_trials': 10,
            'task_constructor':  task.CircleSearch,
            'task_kwargs': {
                    'n_colors': 2,
                    'n_items': 8,
                    'n_attempts': 10,
                    'p_large': 0.5
            },
            'model_constructor': models.IdealObserver,
        },
    }

    return config
