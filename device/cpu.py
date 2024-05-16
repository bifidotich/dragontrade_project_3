import multiprocessing
from device.worker import worker


def parallel_method(method, params_list, num_processes=10):

    params_queue = multiprocessing.Queue()
    for params in params_list:
        params_queue.put(params)

    pool = multiprocessing.Pool(processes=num_processes)
    workers = []
    for _ in range(num_processes):
        worker_process = multiprocessing.Process(target=worker,
                                                 args=(params_queue, method,))
        worker_process.start()
        workers.append(worker_process)
    for worker_process in workers:
        worker_process.join()

    pool.close()
    pool.join()
