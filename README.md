# Template Project

This directory contains a template python project, including main run script,
configuration setup, and openmind launch script. This directory is intended to
be forked by you when you start a new project.

## Code Organization and Summary

### Configuration Files

The main run script is
[main.py](https://github.mit.edu/jazlab/python_utils/blob/master/template_project/main.py).
That can be run as is with the command `$ python3 main.py`. The optional
`--config` flag allows you to control which configuration file to run. See
[configs](https://github.mit.edu/jazlab/python_utils/blob/master/template_project/configs)
for a couple of examples.

A configuration file has a `get_config()` function that returns a big dictionary
containing a blueprint of the object-oriented code you want to run, including
all hyperparameters. The motivation for this design is to have all parameters in
one place and to have lazy construction of objects/functions so that parameters
can be overridden when running sweeps on Openmind. The configuration dictionary
is compiled into objects in `main.py`, and see
[build_from_config.py](https://github.mit.edu/jazlab/python_utils/blob/master/configs/build_from_config.py)
for details about how this is done.

### Overriding Configs for Hyperparameter Sweeps

Running sweeps over sets of hyperparameters is often a essential part of
research, typically done on Openmind. To do this at scale and minimize manual
overhead we must have an efficient way to specify the hyperparameters and values
to sweep over and launch job arrays to Openmind. Consequently, we must somehow
general sets of hyperparameters and pass those as flags to `main.py`. In this
framework that is done with the `--config_overrides` flag, which takes a string
encoding all aspects of the config to override and applies those overrides
before instantiating the objects/functions in the config.

The details of how the encoding is done in `--config_overrides` are not
important, because the user should never need to manually input
`--config_overrides`. (Nonetheless, details can be found in 
[override_config.py](https://github.mit.edu/jazlab/python_utils/blob/master/configs/override_config.py)).
Instead, to run a sweep the user writes a python file specifying the config
overrides. See
[sweeps](https://github.mit.edu/jazlab/python_utils/blob/master/template_project/sweeps)
for examples. This override construction is made easy painless by a set of tools
[sweep.py](https://github.mit.edu/jazlab/python_utils/blob/master/configs/sweep.py)
that allow sweeping a config node over a set of discrete values, taking products
of such discrete sweeps, and zipping/chaining sweeps. There is also a tool for
creating a directory name that summarized the hyperparameter values and can be
used for logging.

To launch a sweep on Openmind, run the script
[openmind_launch.sh](https://github.mit.edu/jazlab/python_utils/blob/master/template_project/openmind_launch.sh)
with argument a string path to your config overrides script. This launcher will
read the overrides from the overrides script and launch a job array for each
hyperparameter set provided by the overrides script. It also does a few other
things, like logging the config and the overrides script name and creating a
bash file with cancellation commands so you can easily terminate jobs.

The upshot of all of this is that if you fork this directory for your own
project, you should only have to modify the model code, config for your model
(and sweep files if you run sweeps on Openmind), and can reuse all of the
boilerplate code implemented here.

## Getting Started

To get started using this template, first fork this directory. You could use
`git clone`, but that would clone the entire `python_utils` codebase so you
would have to then remove everything except `template_project`. Consequently, it
may be easier to just copy the files here one by one.

Then, be sure to test it out before modifying anything. First, you must install
the `python_utils` library:
``` bash
pip install git+https://github.mit.edu/jazlab/python_utils
```
Then try running `main.py` and try launching a sweep on Openmind (see Openmind
background below for more details).

Then write your own project code! You could re-write your project into
modules.py or create new subdirectories with whatever code you want. In
parallel, create config files to run your code.

Through this whole process you will probably not need to touch `main.py`,
`openmind_launch.sh`, or `openmind_task.sh` much. However, there are a couple of
reasons you may want to touch those:
* You may want to change the very bottom part of `main.py` that builds and runs
the config. For example, you want want to call a function other than `.run()` to
run your code.
* You may want to change the `#SBATCH` parameters in `openmind_task.sh`.
Specifically, you will likely want to increase the time limit and may want to
add more parameters (e.g. for email alerts, number of CPUs/GPUs, etc.).
* You may need to run your code in a singularity image on Openmind if you depend
on special GPU images or something that cannot be pip installed into your
virtual environment on Openmind. That would entail changing the last few lines
of `openmind_task.sh`.

## Openmind Background

If you are new to running things on Openmind, here's a cheat sheet for getting
started:
* Read the
[Openmind cookbook](https://github.mit.edu/MGHPCC/OpenMind/wiki/Cookbook:-Getting-started).
* SSH into Openmind with `ssh your_username@openmind7.mit.edu`. For example, I
would run `ssh nwatters@openmind7.mit.edu`.
* Navigate to a directory of your choice. If you're going to want to use more
than a few gigs of storage, try using `/om`. For example, I might use
`/om/user/nwatters/python_utils/template_project`.
* Note: Try using an SSH interpreter in your code editor so that you can write
code on your local computer but have that mirrored on Openmind. For example,
[Here](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html)
are instructions for configuring and SSH interpreter on Pycharm. If you don't
use an SSH interpreter you will have to copy (e.g. with `scp` or `rsync`) your
code to Openmind whenever you want to run anything on Openmind.
* If you're going to be running any non-trivial code, be sure to get off the
head node by entering an interactive session with
`srun -n 1 -t 02:00:00  --pty bash` (that's for 2 hours, but feel free to change
the time limit). You can run `hostname` anytime to check whether you are in an
interactive session and which node you are using.
* Create a virtual environment so you can install packages there without
installing on all of Openmind. For example, `conda create -n my_env python=3.7`
will create a virtual environment named `my_env`, which you can then enter with
`conda activate my_env`. You may want to use the `--prefix` flag to
`conda create` to create the environment outside of `/home`. For example, I use
the command
`conda create --prefix /om/user/nwatters/venvs/python_utils python=3.7`. Note
that once created, the virtual environment will persist ad infinitum, so while
you will need to re-activate it each time you log into Openmind, you will not
need to re-create a new virtual environment.
* Now once you've activated the virtual environment you are ready to pip install
`python_utils` and start running code on Openmind.



