import os
import numpy as np
import json
from functools import reduce

# we use both json and npz files for inputs, depending on whether they are numeric, or string/bool, respectively
# the first two functions handle that separation
def isjson(obj):
    if isinstance(obj, bool):
        return True # catches bools
    else:
        try:
            return all(isinstance(elem, str) for elem in obj) # catches strings and lists of strings
        except TypeError:
            return False

def dict_split(*args, **kwargs):
    kwargs_json = {}
    kwargs_num = {}

    for key, value in kwargs.items():
        if isjson(value):
            kwargs_json[key] = value
        else:
            kwargs_num[key] = value

    for idx, non_kw_arg in enumerate(args):
        kwargs_num[f'arr_{idx}'] = non_kw_arg

    return kwargs_json, kwargs_num


def catalogue(directory, *args, file_spec ='', **kwargs):

    kwargs_json, kwargs_num = dict_split(*args, **kwargs)

    for file in os.listdir(directory):
        if 'inputs.npz' in file and file_spec in file:
            file_handle = file[:-11]
            npz_file_os = os.path.join(directory, file)
            json_file_os = os.path.join(directory, file[:-3] + 'json')

            verdict = True
            file_args = {}

            with open(json_file_os, mode="r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    file_args[key] = value
                for key, value in kwargs_json.items():
                    if data[key] != value and data[key] != list(value):  # to allow for different iterables
                        verdict = False

            with np.load(npz_file_os) as data:
                for key, value in data.items():
                    file_args[key] = value
                for key, value in kwargs_num.items():
                    if not np.array_equal(data[key], value):
                        verdict = False

            if verdict:
                print(f'\nExperiment: {file_handle}')
                n_samples = len([sample for sample in os.listdir(directory) if file_handle+'_sample' in sample])
                print(f'{n_samples} sample(s) found.')
                for key, value in file_args.items():
                    if isinstance(value, np.ndarray) and len(np.shape(value)) > 0:
                        if len(np.shape(value)) == 1:
                            print(f'{key} is an array of length {np.shape(value)[0]}')
                        else:
                            print(f'{key} is an array of shape {np.shape(value)}')
                    else:
                        print(f'{key} = {value}')


def exp_finder(directory, *args, file_spec ='', **kwargs):

    kwargs_json, kwargs_num = dict_split(*args, **kwargs)
    file_list = []

    for file in os.listdir(directory):
        if 'inputs.npz' in file and file_spec == file[:len(file_spec)]:
            npz_file_os = os.path.join(directory, file)
            json_file_os = os.path.join(directory, file[:-3] + 'json')

            with np.load(npz_file_os) as np_data:
                try:
                    np.testing.assert_equal(dict(np_data), kwargs_num)
                    with open(json_file_os, mode="r", encoding="utf-8") as json_file:
                        json_data = json.load(json_file)
                        json_data.pop('entropy')
                        verdict = json_data == kwargs_json
                except AssertionError:
                    verdict = False

            if verdict:
                file_list.append(npz_file_os)

    return file_list


def math_to_python(file, directory = None):
    if directory is None:
        fname = file
    else:
        fname = os.path.join(directory, file)
    with open(fname, 'rb') as f:
        depth = np.fromfile(f, dtype=np.dtype('int32'), count=1)[0]
        dims = np.fromfile(f, dtype=np.dtype('int32'), count=depth)
        data = np.transpose(np.reshape(np.fromfile(f, dtype=np.dtype('float64'),
                                                   count=reduce(lambda x, y: x * y, dims)), dims))
    return data