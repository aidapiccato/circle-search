"""Main entry point to run this project.

This script receives a --config flag, then builds the config and calls its
.run() method. For your project you likely want to fork this file with little
modification. You may want to change the last few lines of this file  to only
build part of the config or call some method other than .run().
"""

from absl import flags
from absl import app

import importlib
import logging
import os
import pickle

from python_utils.configs import build_from_config
from python_utils.configs import override_config

from task import plot_trials

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'configs.circle_search',
                    'Module name of task config to use.')
flags.DEFINE_string(
    'config_overrides', '',
    'JSON-serialized config overrides. This is typically not used locally, '
    'only when running sweeps on Openmind.')
flags.DEFINE_string('log_directory', 'logs', 'Prefix for the log directory.')
flags.DEFINE_string(
    'metadata', '',
    'Metadata to write to metadata.log file. Often used for slurm task ID.')

def main(_):

    ############################################################################
    # Load config
    ############################################################################

    config_module = importlib.import_module(FLAGS.config)
    config = config_module.get_config()
    logging.info(FLAGS.config_overrides)

    # Apply config overrides
    config = override_config.override_config_from_json(config,
                                                       FLAGS.config_overrides)

    ############################################################################
    # Create logging directory
    ############################################################################

    log_dir = FLAGS.log_directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # If log_dir is not empty, create a new enumerated sub-directory in it for
    # logging.
    list_log_dir = os.listdir(log_dir)
    existing_log_subdirs = [
        int(filename) for filename in list_log_dir if filename.isdigit()]
    if not existing_log_subdirs:
        existing_log_subdirs = [-1]
    new_log_subdir = str(max(existing_log_subdirs) + 1)
    log_dir = os.path.join(log_dir, new_log_subdir)
    os.mkdir(log_dir)

    logging.info('Log directory: {}'.format(log_dir))

    ############################################################################
    # Log config name, config overrides, config, and metadata
    ############################################################################

    def _log(log_filename, thing_to_log):
        f_name = os.path.join(log_dir, log_filename)
        logging.info('In file {} will be written:'.format(log_filename))
        logging.info(thing_to_log)
        f_name_open = open(f_name, 'w+')
        f_name_open.write(thing_to_log)

    _log('config_name.log', FLAGS.config)
    _log('config_overrides.log', FLAGS.config_overrides)
    _log('config.log', str(config))
    _log('metadata.log', FLAGS.metadata)

    ############################################################################
    # Build and run task
    ############################################################################

    task = build_from_config.build_from_config(config)
    task_output = task(log_dir=log_dir)
    with open(os.path.join(log_dir, "output"), 'wb') as f:
        pickle.dump(task_output, f)



if __name__ == '__main__':
    app.run(main)
