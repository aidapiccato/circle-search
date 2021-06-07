"""Basic configuration file.

The get_config() method in this file returns a dictionary that can be passed to
python_utils.build_from_config.build_from_config().
"""

import models


def get_config():
    """Get config for main.py."""

    config = {
        'constructor': models.SimplePrinter,
        'args': [1, 2, 3],
        'kwargs': {
            'a': 1,
            'b': '2',
            'c': [3, 4, 5],
            'sub_module': {
                'constructor': models.SimplePrinter,
                'kwargs': {
                    'x': 0,
                    'y': '1',
                },
            },
        },
    }

    return config
