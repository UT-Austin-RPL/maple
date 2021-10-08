"""
Utility functions for writing and loading data.
"""
import json
import numpy as np
import os
import os.path as osp
from collections import defaultdict, namedtuple

from maple.pythonplusplus import nested_dict_to_dot_map_dict


Trial = namedtuple("Trial", ["data", "variant", "directory"])


def matches_dict(criteria_dict, test_dict):
    for k, v in criteria_dict.items():
        if k not in test_dict:
            return False
        else:
            if test_dict[k] != v:
                return False
    return True


class Experiment(object):
    """
    Represents an experiment, which consists of many Trials.
    """
    def __init__(self, base_dir, criteria=None):
        """
        :param base_dir: A path. Directory structure should be something like:
        ```
        base_dir/
            foo/
                bar/
                    arbtrarily_deep/
                        trial_one/
                            variant.json
                            progress.csv
                        trial_two/
                            variant.json
                            progress.csv
                    trial_three/
                        variant.json
                        progress.csv
                        ...
                    variant.json  # <-- base_dir/foo/bar has its own Trial
                    progress.csv
                variant.json  # <-- base_dir/foo has its own Trial
                progress.csv
            variant.json  # <-- base_dir has its own Trial
            progress.csv
        ```

        The important thing is that `variant.json` and `progress.csv` are
        in the same sub-directory for each Trial.
        :param criteria: A dictionary of allowable values for the given keys.
        """
        if criteria is None:
            criteria = {}
        self.trials = get_trials(base_dir, criteria=criteria)
        assert len(self.trials) > 0, "Nothing loaded."
        self.label = 'AverageReturn'

    def get_trials(self, criteria=None):
        """
        Return a list of Trials that match a criteria.
        :param criteria: A dictionary from key to value that must be matches
        in the trial's variant. e.g.
        ```
        >>> print(exp.trials)
        [
            (X, {'a': True, ...})
            (Y, {'a': False, ...})
            (Z, {'a': True, ...})
        ]
        >>> print(exp.get_trials({'a': True}))
        [
            (X, {'a': True, ...})
            (Z, {'a': True, ...})
        ]
        ```
        If a trial does not have the key, the trial is filtered out.
        :return:
        """
        if criteria is None:
            criteria = {}
        return [trial for trial in self.trials
                if matches_dict(criteria, trial.variant)]


def get_dirs(root):
    """
    Get a list of all the directories under this directory.
    """
    yield root
    for root, directories, filenames in os.walk(root):
        for directory in directories:
            yield os.path.join(root, directory)


def get_trials(base_dir, verbose=False, criteria=None, excluded_seeds=()):
    """
    Get a list of (data, variant, directory) tuples, loaded from
        - process.csv
        - variant.json
    files under this directory.
    :param base_dir: root directory
    :param criteria: dictionary of keys and values. Only load experiemnts
    that match this criteria.
    :return: List of tuples. Each tuple has:
        1. Progress data (nd.array)
        2. Variant dictionary
    """
    if criteria is None:
        criteria = {}

    trials = []
    # delimiter = ','
    delimiter = ','
    for dir_name in get_dirs(base_dir):
        variant_file_name = osp.join(dir_name, 'variant.json')
        if not os.path.exists(variant_file_name):
            continue

        with open(variant_file_name) as variant_file:
            variant = json.load(variant_file)
        variant = nested_dict_to_dot_map_dict(variant)

        if 'seed' in variant and int(variant['seed']) in excluded_seeds:
            continue

        if not matches_dict(criteria, variant):
            continue

        data_file_name = osp.join(dir_name, 'progress.csv')
        # Hack for iclr 2018 deadline
        if not os.path.exists(data_file_name) or os.stat(
                data_file_name).st_size == 0:
            data_file_name = osp.join(dir_name, 'log.txt')
            if not os.path.exists(data_file_name):
                continue
            delimiter = '\t'
        if verbose:
            print("Reading {}".format(data_file_name))
        num_lines = sum(1 for _ in open(data_file_name))
        if num_lines < 2:
            continue
        # print(delimiter)
        data = np.genfromtxt(
            data_file_name,
            delimiter=delimiter,
            dtype=None,
            names=True,
        )
        trials.append(Trial(data, variant, dir_name))
    return trials


def get_all_csv(base_dir, verbose=False):
    """
    Get a list of all csv data under a directory.
    :param base_dir: root directory
    """
    data = []
    delimiter = ','
    for dir_name in get_dirs(base_dir):
        for data_file_name in os.listdir(dir_name):
            if data_file_name.endswith(".csv"):
                full_path = os.path.join(dir_name, data_file_name)
                if verbose:
                    print("Reading {}".format(full_path))
                data.append(np.genfromtxt(
                    full_path, delimiter=delimiter, dtype=None, names=True
                ))
    return data


def get_unique_param_to_values(all_variants):
    variant_key_to_values = defaultdict(set)
    for variant in all_variants:
        for k, v in variant.items():
            if type(v) == list:
                v = str(v)
            variant_key_to_values[k].add(v)
    unique_key_to_values = {
        k: variant_key_to_values[k]
        for k in variant_key_to_values
        if len(variant_key_to_values[k]) > 1
    }
    return unique_key_to_values