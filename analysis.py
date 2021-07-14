"""Main entry point to run this project.

This script receives a --config flag, then builds the config and calls its
.run() method. For your project you likely want to fork this file with little
modification. You may want to change the last few lines of this file  to only
build part of the config or call some method other than .run().
"""

from absl import flags
from absl import app

import importlib

FLAGS = flags.FLAGS
flags.DEFINE_string('unpack_module', 'behavior_analysis.unpack_moog', 'Name of module containing functions to unpack '
                                                                       'raw data streams')
flags.DEFINE_string('analysis_module', 'behavior_analysis.analysis', 'Name of behavioral analysis module containing '
                                                                     'functions to apply to dataframe')
flags.DEFINE_string('analysis_function', 'get_features', 'Name of functions in analysis module to carry out')
flags.DEFINE_string('plot_module', 'behavior_analysis.plot_trials',
                    'Name of behavioral analysis module containing functions to carry out')
flags.DEFINE_list('plot_functions', '', 'Name of functions in module to carry out ')
flags.DEFINE_string(
    'trials_directory', 'logs',
    'Directory to recursively search for output files')
flags.DEFINE_string('output_filter', 'output', 'Filter used to select output data files')


def main(_):
    ############################################################################
    # Loading trials intro data frame
    ############################################################################

    mod = importlib.import_module(FLAGS.unpack_module)
    get_trials_dataframe = getattr(mod, 'get_trials_dataframe')
    trials = get_trials_dataframe(FLAGS.trials_directory)
    
    ############################################################################
    # Loading and running analysis functions
    ############################################################################

    mod = importlib.import_module(FLAGS.analysis_module)
    func = getattr(mod, FLAGS.analysis_function)
    trials = func(trials)

    ############################################################################
    # Loading and running plotting functions
    ############################################################################

    mod = importlib.import_module(FLAGS.plot_module)

    for func_str in FLAGS.plot_functions:
        func = getattr(mod, func_str)
        fig = func(trials)
        fig.savefig(f'images/analysis/{func_str}')


if __name__ == '__main__':
    app.run(main)
