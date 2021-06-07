"""Main entry point to run this project.

This script receives a --config flag, then builds the config and calls its
.run() method. For your project you likely want to fork this file with little
modification. You may want to change the last few lines of this file  to only
build part of the config or call some method other than .run().
"""

from absl import flags
from absl import app

import importlib
from utils.analysis import get_trials, get_trials_dataframe

FLAGS = flags.FLAGS
flags.DEFINE_string('analysis_module', 'behavior_analysis.plot_trials',
                    'Name of behavioral analysis module containing functions to carry out')
flags.DEFINE_list('analysis_functions', '', 'Name of functions in module to carry out ')
flags.DEFINE_string(
    'trials_directory', 'logs',
    'Directory to recursively search for output files')
flags.DEFINE_string('output_filter', 'output', 'Filter used to select output data files')

def main(_):

    ############################################################################
    # Loading trials intro dataframe
    ############################################################################

    trials = get_trials_dataframe(get_trials(FLAGS.trials_directory))

    ############################################################################
    # Loading and running analysis functions
    ############################################################################

    mod = importlib.import_module(FLAGS.analysis_module)

    for func_str in FLAGS.analysis_functions:
        func = getattr(mod, func_str)
        fig = func(trials)
        fig.savefig(f'images/analysis/{func_str}')



if __name__ == '__main__':
    app.run(main)
