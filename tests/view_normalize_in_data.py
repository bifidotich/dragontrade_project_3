import random
import config
import research
import utils
import numpy as np

if __name__ == '__main__':
    utils.files.set_project_directory()
    utils.files.mother_iam_coder()

    th = 0.00001
    step = 100

    npz_files = utils.files.find_files(f"{config.PATH_TRAIN_DATA}", '', '.npz')
    path_file = random.choice(npz_files)
    data = np.load(path_file)
    in_data = data['in_data']
    print(in_data.shape)
    print(path_file)
    print('count', len(in_data))
    print('step', step)
    for i in range(100, len(in_data), step):
        data_test = in_data[i]
        if max(data_test[0][:, 0]) > th:
            print(data_test.shape)
            print('i', i, f'data_test > {th}')
            properties = [data_test[0][:, i] for i in range(len(config.SELECTED_COLUMNS))]
            for i, p in enumerate(properties):
                print(i, min(p), max(p))
            research.SimpleGraph(properties)
