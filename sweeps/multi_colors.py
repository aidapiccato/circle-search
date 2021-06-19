"""Generate config overrides for a job array over configs.simple.py.

This defines all of the parameter overrides for a sweep over config parameters.
See python_utils/configs/sweep.py for functions you can use to generate sweeps.

Warning: Do not add or alter any print statements in this file. The launch
script openmind_launch.sh relies on the printing behavior of this file (config
first, then serialized sweep elements) to parse these prints --- the print
statements are the only way this file communicates the sweep to the launch
script.
"""

import numpy as np
from python_utils.configs import sweep

_CONFIG_NAME = 'configs.multi_colors'


def _get_param_sweep():
    """Return the sweep we want to launch."""
    param_sweep = sweep.product(
        sweep.discrete(('kwargs', 'task_kwargs', 'n_colors'), [2, 3]),
        sweep.discrete(('kwargs', 'task_kwargs', 'frac_target'), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                  1.0]),
        sweep.discrete(('kwargs', 'task_kwargs', 'n_items'), [16, 32, 64, 128, 256]),
    )
    return param_sweep


def main():
    """Generate and write sweep of config overrides."""

    # Print the config name. It is important to print this out at the beginning
    # because openmind_launch.sh reads the first thing printed by this script as
    # the config name.
    print(_CONFIG_NAME)

    # Define the sweep we want to launch:
    param_sweep = _get_param_sweep()

    # This overrides 'log_dir' in the config to be a short string capturing the
    # parameter values so far defined in param_sweep.
    param_sweep = sweep.add_log_dir_sweep(param_sweep)

    # Note, it is absolutely fine to add more overrides to the sweep here that
    # you don't want to include in the log_dir (to keep the log_dir short). But
    # since the log_dir override has already been added be sure to not add any
    # additional elements to the sweep, i.e. make sure that anything added here
    # is a singleton sweep. For example, I often add parameters like how often
    # scalars/images should be logged, batch size, etc. that I don't want to
    # sweep over but want to override for array launches from the values in the
    # config. Here's an example:
    # param_sweep = sweep.product(
    #     param_sweep,
    #     sweep.discrete(('kwargs', 'b'), ['-4']),
    # )

    # Print one spec per line. It is important to print these out line by line,
    # because openmind_launch.sh relies on these prints, piping them into an
    # array that it uses to launch to job array.
    for json_spec in sweep.serialize_sweep_elements(param_sweep):
        print(json_spec)


if __name__ == '__main__':
    main()
