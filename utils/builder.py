import pytz
import random
import datetime
import numpy as np
from dateutil import parser
from datetime import datetime, timedelta


def datetime_now_timezone(hours_timezone, delta=False):
    datetime_now = datetime.now(pytz.utc) + timedelta(hours=hours_timezone)
    if not delta:
        datetime_now = datetime_now.replace(tzinfo=None)
    return datetime_now


def generate_random_datetime(start_date, end_date):
    if isinstance(start_date, str):
        start_date = parser.parse(start_date)
    if isinstance(end_date, str):
        end_date = parser.parse(end_date)
    time_difference = random.randint(0, int((end_date - start_date).total_seconds()))
    rounded_time_difference = timedelta(minutes=(time_difference // 60 // 10) * 10)
    random_datetime = start_date + rounded_time_difference
    return random_datetime


def create_datetime_ranges(list_datetime, minutes):
    result = []
    for start in list_datetime:
        end = start + datetime.timedelta(minutes=minutes)
        datetime_range = np.arange(start, end, datetime.timedelta(minutes=1))
        result.append(datetime_range)
    return np.array(result)


def shuffle_arrays_match(array1, array2):
    len1, len2 = len(array1), len(array2)
    if len1 != len2:
        raise ValueError(f"LEN(array1) <> LEN(array2): received {len1} and {len2} ")
    permutation_indices = np.random.permutation(len(array1))
    shuffled_array1 = array1[permutation_indices]
    shuffled_array2 = array2[permutation_indices]
    return shuffled_array1, shuffled_array2


def zip_sorted(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Input arrays must have the same length")

    arr1, arr2 = np.array(arr1), np.array(arr2)
    sorted_indices = np.argsort(arr1)
    arr1 = arr1[sorted_indices]
    arr2 = arr2[sorted_indices]
    return arr1, arr2


def zip_filter(arr1, arr2, filter_value=-1):
    if len(arr1) != len(arr2):
        raise ValueError("Input arrays must have the same length")

    filtered_arr1, filtered_arr2 = [], []
    for i in range(len(arr1)):
        if arr1[i] != filter_value:
            filtered_arr1.append(arr1[i])
            filtered_arr2.append(arr2[i])
    return np.array(filtered_arr1), np.array(filtered_arr2)
