import multiprocessing
import tensorflow as tf
from device.worker import gpu_worker


def parallel_fit(method, params_list, devices=None):

    if devices is None:
        devices = range(0, len(tf.config.experimental.list_physical_devices('GPU')))
    if len(devices) < 1:
        raise RuntimeError('GPU devices not found')

    params_queue = multiprocessing.Queue()
    for params in params_list:
        params_queue.put(params)

    pool = multiprocessing.Pool(processes=len(devices))
    workers = []
    for process_index in devices:
        worker_process = multiprocessing.Process(target=gpu_worker,
                                                 args=(params_queue, method, process_index))
        worker_process.start()
        workers.append(worker_process)
    for worker_process in workers:
        worker_process.join()

    pool.close()
    pool.join()
