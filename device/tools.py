import multiprocessing
import os
from device import cpu, gpu, config


def generate_combinations_params_list(input_list):
    if not input_list:
        return [[]]
    first_element = input_list[0]
    rest_combinations = generate_combinations_params_list(input_list[1:])
    all_combinations = []
    if not isinstance(first_element, list):
        for combination in rest_combinations:
            all_combinations.append([first_element] + combination)
    else:
        for sublist_element in first_element:
            for combination in rest_combinations:
                all_combinations.append([sublist_element] + combination)
    return all_combinations


def cpu_work(method, combination_args, num_workers=None):
    combination_params = generate_combinations_params_list(combination_args)
    if num_workers is None:
        num_workers = int(os.cpu_count() * config.VAL_CORE)
    cpu.parallel_method(method=method,
                        params_list=combination_params,
                        num_processes=num_workers)


def cpu_map(method, combination_args, num_workers=None):
    combination_params = generate_combinations_params_list(combination_args)
    combination_params = list(map(lambda x: tuple(x), combination_params))
    if num_workers is None:
        num_workers = int(os.cpu_count() * config.VAL_CORE)
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(method, iterable=combination_params)


def gpu_multifit(method, combination_args, devices=None):
    combination_params = generate_combinations_params_list(combination_args)
    gpu.parallel_fit(method=method,
                     params_list=combination_params,
                     devices=devices)
