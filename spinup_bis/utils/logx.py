"""Some simple logging functionality, inspired by rllab's logging.

It's trimmed down of saving capabilities, version of Spinning Up logger.
"""

import atexit
import collections
import json
import os
import os.path as osp
import time

import matplotlib.pyplot as plt
import numpy as np

from spinup_bis.utils import mpi_tools
from spinup_bis.utils import serialization_utils


plt.style.use('seaborn-white')


COLOR2NUM = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = COLOR2NUM[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def flatten_config(config, prefix=''):
    flattened = {}
    for key, val in config.items():
        if isinstance(val, dict):
            flattened.update(flatten_config(val, prefix=prefix + key + '.'))
        else:
            flattened[prefix + key] = val
    return flattened


class Logger:
    """A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(self,
                 output_dir=None,
                 output_fname='progress.txt',
                 exp_name=None):
        """Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.
            output_fname (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress.txt``.
            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        if mpi_tools.proc_id() == 0:
            self.output_dir = output_dir or f'./out/{time.time():.0f}'
            if osp.exists(self.output_dir):
                print(f'Warning: Log dir {self.output_dir} already exists!'
                      ' Storing info there anyway.')
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(
                osp.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            print(colorize(
                f'Logging data to {self.output_file}.', 'green', bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    @staticmethod
    def log(msg, color='green'):
        """Print a colorized message to stdout."""
        if mpi_tools.proc_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, \
                f'Trying to introduce a new key {key}' \
                ' that you didn\'t include in the first iteration!'
        assert key not in self.log_current_row, \
            f'You already set {key} this iteration.' \
            ' Maybe you forgot to call dump_tabular()'
        self.log_current_row[key] = val

    def save_config(self, config):
        """Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = serialization_utils.convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if mpi_tools.proc_id() == 0:
            output = json.dumps(config_json, separators=(
                ',', ':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config:\n', color='cyan', bold=True))
            print(output)
            with open(osp.join(self.output_dir, 'config.json'), 'w') as out:
                out.write(output)

    def dump_tabular(self):
        """Write all of the diagnostics from the current iteration.

        Writes to stdout and to the output file:
        (path/to/output_directory/progress.txt).
        """
        if mpi_tools.proc_id() == 0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%' + f'{max_key_len}'
            fmt = '| ' + keystr + 's | %15s |'
            n_slashes = 22 + max_key_len
            print('-' * n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, '')
                # Format value string.
                if hasattr(val, '__float__'):
                    valstr = f'{val:8.3g}'
                elif isinstance(val, tuple):  # Histogram!
                    hist, edges = val

                    # Accumulate hist values.
                    hist_, edges_ = [], []
                    for i in range(len(hist) // 4):
                        # Average because hist is density.
                        hist_.append(sum(hist[4 * i:4 * (i + 1)]) / 4)
                        edges_.append(edges[4 * i])
                    edges_.append(edges[-1])

                    valstr = ', '.join([f'(>{e:5.3g}: {v:5.3g})'
                                        for v, e in zip(hist_, edges_[:-1])])
                    valstr += f', (<{edges_[-1]:5.3g})'
                else:
                    valstr = val
                print(fmt % (key, valstr))
                vals.append(val)
            print('-' * n_slashes, flush=True)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write('\t'.join(self.log_headers) + '\n')
                self.output_file.write('\t'.join(map(str, vals)) + '\n')
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = collections.defaultdict(list)

    def store(self, **kwargs):
        """Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            self.epoch_dict[k].append(v)

    def log_tabular(self,  # pylint: disable=arguments-differ
                    key,
                    val=None,
                    with_min_and_max=False,
                    average_only=False,
                    histogram=False):
        """Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.

            histogram (bool): If true, then log histogram of values.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            if not v:  # in case nothing has been stored under this key
                stats = (np.nan, np.nan, np.nan, np.nan)
            else:
                vals = np.concatenate(v) if isinstance(
                    v[0], np.ndarray) and len(v[0].shape) > 0 else v
                stats = mpi_tools.mpi_statistics_scalar(
                    vals, with_min_and_max=with_min_and_max)
            super().log_tabular(
                key if average_only else 'Average' + key, stats[0])
            if not average_only:
                super().log_tabular('Std' + key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max' + key, stats[3])
                super().log_tabular('Min' + key, stats[2])
            if histogram:
                hist, edges = mpi_tools.mpi_histogram(vals)
                super().log_tabular('Histogram' + key, (hist, edges))
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """Lets an algorithm ask the logger for mean/std of a diagnostic."""
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(
            v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return mpi_tools.mpi_statistics_scalar(vals)
