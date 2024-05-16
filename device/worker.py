import device.config
import tensorflow as tf
from utils.files import retry


@retry(max_attempts=1, retry_delay_seconds=10, active=not device.config.DEBUG)
def cpu_func(method, params):
    method(*params)


def worker(params_queue, method):
    while not params_queue.empty():
        params = params_queue.get()
        cpu_func(method, params)


@retry(max_attempts=1, retry_delay_seconds=40, active=not device.config.DEBUG)
def gpu_func(method, params):
    method(*params)


def gpu_worker(params_queue, method, idx_device):

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        tf.config.experimental.set_visible_devices(gpu_devices[idx_device], 'GPU')

    while not params_queue.empty():
        params = params_queue.get(timeout=1)
        gpu_func(method, params)
