import random
import config
import utils
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import dragonfly_model

syb = 'EURUSD'

if __name__ == '__main__':
    utils.files.set_project_directory()
    utils.files.mother_iam_coder()

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        tf.config.experimental.set_visible_devices([], 'GPU')

    # dragonfly_model.utils.export_model_file(source_dir=f"{config.PATH_TEST_MODEL}/_checkpoints",
    #                                         new_file_name=f"{config.PATH_TEST_MODEL}/model.h5")


    def count_numbers(arr):
        number_count = {}
        for num in arr:
            if num in number_count:
                number_count[num] += 1
            else:
                number_count[num] = 1
        return number_count


    npz_files = utils.files.find_files(f"{config.PATH_TRAIN_DATA}", '', '.npz')
    path_file = random.choice(npz_files)
    data = np.load(path_file)
    model = dragonfly_model.cnn.load(f'{config.PATH_TEMP_MODELS}/{syb}_REGRESS_2/model.h5')
    predictions = model.predict(data['in_data'])
    cur_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(data['out_data'], axis=1)
    count_class = data['out_data'].shape[-1]

    true_res = 0
    for i in tqdm(range(len(true_labels))):
        if cur_labels[i] == true_labels[i]:
            true_res += 1
    for i in random.choices(range(len(true_labels)), k=20):
        print('predictions\n', predictions[i])
        print('max predictions', np.max(predictions[i]))
        print('cur_labels', cur_labels[i])
        print('true_labels', true_labels[i])
        print('=' * 100)

    print('true_res', true_res, len(true_labels), round(true_res / len(true_labels), 3))
    print('count (cur_labels)', count_numbers(cur_labels))
    print('count (true_labels)', count_numbers(true_labels))
    print('=' * 100)

    distr_predict = {}
    distr_predict_true = {}
    distr_predict_false = {}
    true_res, false_res = [], []
    for i in range(len(true_labels)):

        round_predict = str(np.round(np.max(predictions[i]), 2))
        if round_predict not in distr_predict:
            distr_predict[round_predict] = 0
            distr_predict_true[round_predict] = 0
            distr_predict_false[round_predict] = 0
        distr_predict[round_predict] += 1

        if cur_labels[i] == true_labels[i]:
            true_res.append(np.max(predictions[i]))
            distr_predict_true[round_predict] += 1
        else:
            false_res.append(np.max(predictions[i]))
            distr_predict_false[round_predict] += 1
    try:
        print("true_res mean", np.mean(true_res))
        print("true_res median", np.median(true_res))
        print("true_res min", np.min(true_res))
        print("true_res max", np.max(true_res))
        print("false_res mean", np.mean(false_res))
        print("false_res median", np.median(false_res))
        print("false_res min", np.min(false_res))
        print("false_res max", np.max(false_res))
    except Exception:
        pass
    print('=' * 100)

    print('distribution no true predict class')
    print('true: predict')
    distr_errors = {}
    for main_class in range(count_class):
        for check_class in range(count_class):
            if check_class == main_class:
                continue
            iter_res = 0
            for i in range(len(true_labels)):
                if cur_labels[i] == check_class and true_labels[i] == main_class:
                    iter_res += 1
            distr_errors[f'{main_class}:{check_class}'] = iter_res
    distr_errors = sorted(distr_errors.items(), key=lambda item: item[1], reverse=True)
    print(distr_errors)
    print('=' * 100)

    distr_predict = sorted(distr_predict.items(), key=lambda item: item[1], reverse=True)
    distr_predict_true = sorted(distr_predict_true.items(), key=lambda item: item[1], reverse=True)
    distr_predict_false = sorted(distr_predict_false.items(), key=lambda item: item[1], reverse=True)
    print('distribution predict')
    print('distr_predict', distr_predict)
    print('distr_predict_true', distr_predict_true)
    print('distr_predict_false', distr_predict_false)
