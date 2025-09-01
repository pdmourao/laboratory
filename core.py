import os
from time import time
from storage import exp_finder, dict_split
import json
import numpy as np

# This class is supposed to handle running general experiments, saving them to files, etc
# And also facilitate parallel processing across samples
# So it's a runner (as in it runs simulations) and supposed to be fast, so, naturally, it's called UsainBolt.

class Experiment:

    # f is the experiment function to be done
    # args and kwargs are arguments of f which are meant to be recorded as the experiment's parameters
    def __init__(self, func, directory = None, *args, **kwargs):

        # function object ready to compute samples
        # the arguments that will be saved on files
        self.func = func

        self._args = {}
        self._kwargs = {}

        self.set_args(*args, **kwargs)

        # other attributes
        self._file_prefix = None
        self._entropy = np.random.SeedSequence().entropy

        # setting the directory also searches for an experiment with the given inputs
        # see setter
        self.dir = directory

    @property
    def dir(self):
        return self._directory

    @property
    def file_prefix(self):
        return self._file_prefix

    @property
    def entropy(self):
        return self._entropy

    def set_args(self, *args, **kwargs):
        self._args = args
        for key, value in kwargs.items():
            self._kwargs[key] = value

    # sets the directory and checks if there's an existing experiment with matching inputs
    # if kept at None, experiments can still be ran but they won't be saved to files nor can files be read
    @dir.setter
    def dir(self, directory):
        if directory is None:
            self._directory = None
            self._file_prefix = None
            self._entropy = np.random.SeedSequence().entropy
        else:
            self._directory = directory
            input_files = exp_finder(directory=directory, file_spec = self.func.__name__, *self._args, **self._kwargs)
            if len(input_files) > 1:
                print(f'Warning: {len(input_files)} experiment(s) found for given inputs.')
                print(f'{input_files[0]} will be used.')

            # finds existent experiment
            try:
                inputs_file = input_files[0]
                with open(inputs_file[:-3] + 'json', mode="r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    self._entropy = int(data['entropy'])
                print('Experiment found!')
                self._file_prefix = inputs_file[:-10]
            except IndexError:
                self._file_prefix = None
                self._entropy = np.random.SeedSequence().entropy
                print('Experiment not found for given inputs.')

    # call the class to create a new experiment, i.e. save the input files
    def create(self):
        if self._file_prefix is None:
            print('Creating new experiment...')
            kwargs_json, kwargs_num = dict_split(**self._kwargs)

            inputs_file = os.path.join(self._directory, f'{self.func.__name__}-{int(time())}_inputs.npz')
            self._file_prefix = inputs_file[:-10]

            with open(f'{inputs_file[:-3]}json', mode="w", encoding="utf-8") as json_file:
                kwargs_json['entropy'] = str(self._entropy)
                json.dump(kwargs_json, json_file)
            np.savez(inputs_file, *self._args, **kwargs_num)
        else:
            print('Experiment already set.')

    # run a specific sample
    def run(self, sample, *extra_args, save = True, **extra_kwargs):
        output = self.func(entropy = (self._entropy, sample), *self._args, *extra_args, **extra_kwargs, **self._kwargs)
        if sample not in self.samples_present() and self._file_prefix is not None and save:
            np.savez(self._file_prefix+f'sample{sample}', *output)
        return output

    # detect which samples are present
    def samples_present(self):
        assert self._file_prefix is not None, 'Method samples_present needs valid experiment.'
        samples = [int(file.split('_sample')[-1][:-4]) for file in os.listdir(self._directory) if
                                self._file_prefix + 'sample' in os.path.join(self._directory, file)]
        samples.sort()
        return samples

    def samples_missing(self, total):
        sample_list = [sample for sample in range(total) if sample not in self.samples_present()]
        print(f'{len(sample_list)} sample(s) missing out of {total}.')
        return sample_list

    # read a specific sample
    def read_sample(self, sample):
        assert self._file_prefix is not None, 'Method read_sample needs valid experiment.'
        filename = self._file_prefix + f'sample{sample}.npz'
        with np.load(filename) as data:
            output = tuple(data.values())
        return output

    # read existing samples
    def read(self, max_s = np.inf):
        output_list = [self.read_sample(sample) for sample in self.samples_present() if sample < max_s]
        print(f'{len(output_list)} sample(s) were found.')
        return map(np.array, zip(*output_list))

    # read and average
    def read_av(self):
        return tuple([np.mean(output, axis =0) for output in self.read()])