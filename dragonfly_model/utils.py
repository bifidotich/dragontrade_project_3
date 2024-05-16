import os
import utils
import shutil
import numpy as np


def export_model_file(source_dir, new_file_name):
    files = os.listdir(source_dir)
    files = [f for f in files if os.path.isfile(os.path.join(source_dir, f))]
    if files:
        utils.files.track_dir(new_file_name)
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(source_dir, x)))
        source_path = os.path.join(source_dir, latest_file)
        shutil.copy(source_path, new_file_name)


def balance_data(data_input, data_output):
    data_input = np.array(data_input)
    data_output = np.array(data_output)
    num_classes = data_output.shape[1]

    class_indices = [[] for _ in range(num_classes)]

    for i in range(len(data_output)):
        class_index = np.argmax(data_output[i])
        class_indices[class_index].append(i)

    min_samples_per_class = min(len(indices) for indices in class_indices)

    balanced_indices = []
    for indices in class_indices:
        balanced_indices.extend(indices[:min_samples_per_class])

    balanced_data_input = data_input[balanced_indices]
    balanced_data_output = data_output[balanced_indices]

    return balanced_data_input, balanced_data_output


def augment_data(data_input, data_output, target_size=None, noise_stddev=0.001):
    data_input = np.array(data_input)
    data_output = np.array(data_output)
    num_classes = data_output.shape[1]

    class_indices = [[] for _ in range(num_classes)]

    for i in range(len(data_output)):
        class_index = np.argmax(data_output[i])
        class_indices[class_index].append(i)

    if target_size is None:
        target_size = np.mean([len(indices) for indices in class_indices])

    augmented_data_input = []
    augmented_data_output = []

    for indices in class_indices:
        class_samples = len(indices)
        if class_samples < target_size:
            random_indices = np.random.choice(indices, size=int(target_size - class_samples), replace=True)

            for index in random_indices:
                original_sample = data_input[index]
                noisy_sample = original_sample + noise_stddev * np.random.randn(*original_sample.shape)

                augmented_data_input.append(noisy_sample)
                augmented_data_output.append(data_output[index])

    augmented_data_input = np.array(augmented_data_input)
    augmented_data_output = np.array(augmented_data_output)

    combined_data_input = np.concatenate([data_input, augmented_data_input], axis=0)
    combined_data_output = np.concatenate([data_output, augmented_data_output], axis=0)

    return combined_data_input, combined_data_output


def flatten_data(data_input, data_output=None):
    data_input = data_input.squeeze()
    if data_output is not None:
        data_output = data_output.ravel()
    data_input = data_input.reshape(data_input.shape[0], -1)
    return data_input, data_output
