import os
import random
import config
import utils
import numpy as np

if __name__ == '__main__':
    utils.files.set_project_directory()
    utils.files.mother_iam_coder()

    npz_files = utils.files.find_files(f"{config.PATH_TRAIN_DATA}", '', '.npz')
    path_file = random.choice(npz_files)
    data = np.load(path_file)
    data_in = data['in_data']
    data_out = data['out_data']
    res = np.zeros(len(data_out[0]))

    print('in_data shape - ', data_in.shape)
    print('out_data shape - ', data_out.shape)
    print('num elements - ', len(data_out))
    for k in data_out:
        res[k.tolist().index(1)] += 1
    print(res)
