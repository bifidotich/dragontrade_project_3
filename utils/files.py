import os
import sys
import json
import time
import pickle
import shutil
import codecs
import random
import inspect
import functools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class ConsoleLogger:
    def __init__(self, filename):
        track_dir(filename)
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
        if self.stdout is not None:
            sys.stdout = self.stdout


def mother_iam_coder(ignored_directories=None):

    if ignored_directories is None:
        ignored_directories = []
    ignored_directories = ignored_directories + ["venv", ".git", ".idea", "back", "_back", "config", "_other",
                                                 "_older_project", "_older_code", "_logs", "_backups"]
    total_lines = 0
    directory = os.getcwd()

    for root, dirs, files in os.walk(directory):

        dirs[:] = [d for d in dirs if d not in ignored_directories]

        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                with codecs.open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                    lines = file.readlines()
                    total_lines += len(lines)
    print(f"string .py codes: {total_lines}")


def clogger(description=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            str_status = ' ' * 3
            fn = inspect.getfile(func).split('\\')[-1]
            start_time = datetime.now()
            dtime = start_time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                root = f'{args[0].__class__.__name__}.'
            except Exception as e:
                root = ''
            name = f'{root}{func.__name__}'
            t1 = f"({dtime}){str_status}[{fn}] {name}....."
            t2 = f" ↳  {description}....."
            print(t1)
            if description:
                print(t2)
            time.sleep(0.01)
            str_results = '.....completed'
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                str_status = ' E '
                str_results = f'.....done with exception: {e}'
                raise
            finally:
                etime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dir_time = timedelta(seconds=round((datetime.now() - start_time).total_seconds()))
                t_end = f"({etime}){str_status}{name}{str_results}  ({dir_time})"
                print(t_end)
            return result
        return wrapper
    return decorator


def retry(max_attempts, retry_delay_seconds, active=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    function_name = func.__name__
                    exception_description = str(e)
                    print(f"'{function_name}' raised an exception: {exception_description}")
                    attempts += 1
                    if attempts < max_attempts:
                        time.sleep(retry_delay_seconds)
                    else:
                        print(f"Reached {func.__name__} maximum number of attempts ({max_attempts}). Exiting...")
                    if not active:
                        raise
        return wrapper
    return decorator


def set_project_directory(start_dir=os.getcwd(), venv_name='venv'):
    current_dir = start_dir
    while not os.path.exists(os.path.join(current_dir, venv_name)):
        parent_dir = os.path.dirname(current_dir)
        if current_dir == parent_dir:
            return None
        current_dir = parent_dir
    os.chdir(current_dir)
    return current_dir


def update_log(string, path_file='temp/log.txt'):
    track_dir(path_file)
    if not os.path.exists(path_file):
        open(path_file, 'w+', encoding="utf-8")
    with open(path_file, "a", encoding="utf-8") as file:
        file.write(string)


def track_dir(path):
    absolute_path = os.path.abspath(path)
    directory = os.path.dirname(absolute_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_json(path_file):
    if not os.path.exists(path_file):
        return {}
    with open(path_file, "r", encoding='utf8') as file:
        return json.load(file)


def write_json(path_file, data):
    track_dir(path_file)
    with open(path_file, "w", encoding='utf8') as file:
        json.dump(data, file)


def update_json(path_file, key, value):
    info_file = read_json(path_file)
    info_file[key] = value
    write_json(path_file, info_file)


def save_pkl(data, file_path):
    track_dir(file_path)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def del_string_from_filenames(path, string_to_remove):
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path):
        for filename in files + dirs:
            full_path = os.path.join(root, filename)
            if string_to_remove in filename:
                new_name = filename.replace(string_to_remove, '')
                os.rename(full_path, os.path.join(root, new_name))


def find_files(path_directory, partial_name, extension):
    matching_files = []
    for root, _, files in os.walk(path_directory):
        for filename in files:
            if partial_name in filename and filename.endswith(extension):
                matching_files.append(os.path.join(root, filename))
    return matching_files


def distribute_files(path_directory, max_mb_per_folder):
    max_bytes_per_folder = max_mb_per_folder * 1024 * 1024
    output_dirs = []
    current_dir = None
    current_weight = 0
    for root, _, files in os.walk(path_directory):
        random.shuffle(files)
        for filename in files:
            file_path = os.path.join(root, filename)
            file_size = os.path.getsize(file_path)
            if current_dir is None or current_weight + file_size > max_bytes_per_folder:
                current_dir = os.path.join(path_directory, f'{len(output_dirs) + 1}')
                if not os.path.exists(current_dir):
                    os.makedirs(current_dir)
                output_dirs.append(current_dir)
                current_weight = 0
            shutil.move(file_path, os.path.join(current_dir, filename))
            current_weight += file_size
    return output_dirs


def clear_directory(directory, del_directory=False):
    if not (os.path.exists(directory) and os.path.isdir(directory)):
        return
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"{file_path}: {e}")
    if del_directory:
        shutil.rmtree(directory)


def out_dir_and_remove(path_directory):
    for root, _, files in os.walk(path_directory):
        for file in files:
            source_path = os.path.join(root, file)
            target_path = os.path.join(path_directory, file)
            shutil.move(source_path, target_path)
    for root, dirs, files in os.walk(path_directory, topdown=False):
        for _dir in dirs:
            dir_path = os.path.join(root, _dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)


def remove_files_smaller_mb(path_directory, size_in_mb):
    size_in_bytes = size_in_mb * 1024 * 1024
    if not os.path.exists(path_directory):
        print(f"'{path_directory}' not found")
        return

    for root, dirs, files in os.walk(path_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            if file_size < size_in_bytes:
                print(f"file delete {file_path} (size {file_size} b.)")
                os.remove(file_path)


def remove_files_below_average(directory_path, min_delta=0):
    files = os.listdir(directory_path)
    if len(files) < 1:
        return

    total_size = sum(os.path.getsize(os.path.join(directory_path, file)) for file in files)
    average_size = total_size / len(files)

    for file in files:
        file_path = os.path.join(directory_path, file)
        if os.path.getsize(file_path) < average_size - int(min_delta):
            os.remove(file_path)


def validation_csv(input_file, output_file, keep_fraction):
    df = pd.read_csv(input_file)
    num_rows_to_remove = int(len(df) * (1 - keep_fraction))
    random_rows_to_remove = random.sample(range(len(df)), num_rows_to_remove)
    df = df.drop(random_rows_to_remove)
    df.to_csv(output_file, index=False)


def backup_directory(source_directory, back_directory, action='copy'):
    if not os.path.exists(source_directory):
        print(f"'{source_directory}' not found.")
        return

    current_time = datetime.now()
    formatted_date = current_time.strftime("%Y%m%d_%H%M%S")
    new_directory_name = f"{os.path.basename(source_directory)}_{formatted_date}"

    new_directory_path = os.path.join(back_directory, new_directory_name)

    track_dir(new_directory_path)
    if action == 'move':
        shutil.move(source_directory, new_directory_path)
    elif action == 'copy':
        shutil.copytree(source_directory, new_directory_path)
    elif action == 'zip':
        shutil.make_archive(new_directory_path, 'zip', source_directory)
        shutil.rmtree(source_directory)


def combine_npy(directory, output_file_path):
    combined_array = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".npy"):
            file_path = os.path.join(directory, file_name)
            array = np.load(file_path)
            combined_array.append(array)
    result_array = np.concatenate(combined_array, axis=0)
    np.save(output_file_path, result_array)


def move_files(input_directory, output_directory, select=None, copy=False):
    files = [f for f in os.listdir(input_directory)]
    if select is not None:
        if not isinstance(select, list):
            select = [select]
        files = [f for f in files if len([c for c in select if c in f]) > 0]
    for file in files:
        source_path = os.path.join(input_directory, file)
        destination_path = os.path.join(output_directory, file)
        track_dir(destination_path)
        if copy:
            shutil.copy(source_path, destination_path)
        else:
            shutil.move(source_path, destination_path)


def concat_csv_files(directory_path, output_filename='combined_output.csv'):
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    dfs = []

    for file in file_list:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.fillna(0.0)
    combined_df.to_csv(output_filename, index=False)
