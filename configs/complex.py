"""Complex configuration file.

The get_config() method in this file returns a dictionary that can be passed to
python_utils.build_from_config.build_from_config().
"""

import models


def _get_submodule_config():
    config = {
        'constructor': models.SimplePrinter,
        'args': [1, 2, 3],
        'kwargs': {
            'a': 1,
            'b': '2',
            'c': [3, 4, 5],
            'd': {
                'constructor': models.identity_fn,
                'kwargs': {
                    'e': 6,
                    'f': 7,
                },
            },
            'sub_module': {
                'constructor': models.SimplePrinter,
                'kwargs': {
                    'x': 0, 'y': 1,
                }
            }
        },
    }
    return config


def get_config():
    """Get config for main.py."""

    config = {
        'constructor': models.ScalarLogger,
        'kwargs': {
            'sub_module': _get_submodule_config(),
            'a': 0.1,
            'b': 0.2,
            'c': 0.3,
        },
    }

    return config
