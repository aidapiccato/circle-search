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
            'n_trials': 100,
            'task_constructor':  task.CircleSearchAmbiguousFlat,
            'task_kwargs': {
                    'n_colors': 4,
                    'n_items': 4
            },
            'model_constructor': models.IdealObserverAngles,
        },
    }

    return config
