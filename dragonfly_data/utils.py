import re
import pytz
import dateutil.parser
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta


def pandas_timeframe_to_minutes(timeframe):
    if isinstance(timeframe, str):
        timeframe = pd.to_timedelta(timeframe)
        return int(timeframe.total_seconds() // 60)
    else:
        if timeframe.is_anchored():
            if "T" in str(timeframe.freqstr):
                return timeframe.freqstr.count("T")
            elif "H" in str(timeframe.freqstr):
                return timeframe.freqstr.count("H") * 60
            elif "D" in str(timeframe.freqstr):
                return 1440
            elif "W" in str(timeframe.freqstr):
                return 10080
            elif "M" in str(timeframe.freqstr):
                return 43200
            else:
                return None


def minutes_to_pandas_timeframe(minutes):
    if minutes is None:
        return None

    if minutes % 43200 == 0:
        months = minutes / 43200
        return pd.DateOffset(months=months)

    if minutes % 10080 == 0:
        weeks = minutes / 10080
        return pd.DateOffset(weeks=weeks)

    if minutes % 1440 == 0:
        days = minutes / 1440
        return pd.DateOffset(days=days)

    if minutes % 60 == 0:
        hours = minutes / 60
        return pd.DateOffset(hours=hours)

    return None


def dataframe_columns_to_numpy(dataframe, columns):
    for col in columns:
        if col not in dataframe.columns:
            raise ValueError(f"{col} not found")
    data_subset = dataframe[columns].to_numpy()
    return data_subset


def mt5_timeframe_to_string(timeframe):
    timeframe_dict = {
        mt5.TIMEFRAME_M1: 'M1',
        mt5.TIMEFRAME_M2: 'M2',
        mt5.TIMEFRAME_M3: 'M3',
        mt5.TIMEFRAME_M4: 'M4',
        mt5.TIMEFRAME_M5: 'M5',
        mt5.TIMEFRAME_M6: 'M6',
        mt5.TIMEFRAME_M10: 'M10',
        mt5.TIMEFRAME_M12: 'M12',
        mt5.TIMEFRAME_M15: 'M15',
        mt5.TIMEFRAME_M20: 'M20',
        mt5.TIMEFRAME_M30: 'M30',
        mt5.TIMEFRAME_H1: 'H1',
        mt5.TIMEFRAME_H2: 'H2',
        mt5.TIMEFRAME_H3: 'H3',
        mt5.TIMEFRAME_H4: 'H4',
        mt5.TIMEFRAME_H6: 'H6',
        mt5.TIMEFRAME_H8: 'H8',
        mt5.TIMEFRAME_H12: 'H12',
        mt5.TIMEFRAME_D1: 'D1',
        mt5.TIMEFRAME_W1: 'W1',
        mt5.TIMEFRAME_MN1: 'MN1',
    }

    timeframe_str = timeframe_dict.get(timeframe)
    if timeframe_str is None:
        raise ValueError(f'CUSTOM_ERROR: Invalid timeframe {timeframe}')

    return timeframe_str


def string_to_mt5_timeframe(timeframe_string):
    timeframe_dict = {
        'M1': mt5.TIMEFRAME_M1,
        'M2': mt5.TIMEFRAME_M2,
        'M3': mt5.TIMEFRAME_M3,
        'M4': mt5.TIMEFRAME_M4,
        'M5': mt5.TIMEFRAME_M5,
        'M6': mt5.TIMEFRAME_M6,
        'M10': mt5.TIMEFRAME_M10,
        'M12': mt5.TIMEFRAME_M12,
        'M15': mt5.TIMEFRAME_M15,
        'M20': mt5.TIMEFRAME_M20,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H2': mt5.TIMEFRAME_H2,
        'H3': mt5.TIMEFRAME_H3,
        'H4': mt5.TIMEFRAME_H4,
        'H6': mt5.TIMEFRAME_H6,
        'H8': mt5.TIMEFRAME_H8,
        'H12': mt5.TIMEFRAME_H12,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1,
    }

    mt5_timeframe = timeframe_dict.get(timeframe_string)
    if mt5_timeframe is None:
        raise ValueError(f'CUSTOM_ERROR: Invalid timeframe string {timeframe_string}')

    return mt5_timeframe


def get_start_date_update(end_date, timeframe, max_len_input=1000):
    const_size_down = int(max_len_input * 2)
    if isinstance(timeframe, str):
        timeframe = string_to_mt5_timeframe(timeframe)
    if timeframe == mt5.TIMEFRAME_M1:
        local_start_date = end_date - timedelta(minutes=const_size_down)
    elif timeframe == mt5.TIMEFRAME_M5:
        local_start_date = end_date - timedelta(minutes=const_size_down * 5)
    elif timeframe == mt5.TIMEFRAME_M15:
        local_start_date = end_date - timedelta(minutes=const_size_down * 15)
    elif timeframe == mt5.TIMEFRAME_M30:
        local_start_date = end_date - timedelta(minutes=const_size_down * 30)
    elif timeframe == mt5.TIMEFRAME_H1:
        local_start_date = end_date - timedelta(hours=const_size_down)
    elif timeframe == mt5.TIMEFRAME_H2:
        local_start_date = end_date - timedelta(hours=const_size_down * 2)
    elif timeframe == mt5.TIMEFRAME_H3:
        local_start_date = end_date - timedelta(hours=const_size_down * 3)
    elif timeframe == mt5.TIMEFRAME_H4:
        local_start_date = end_date - timedelta(hours=const_size_down * 4)
    elif timeframe == mt5.TIMEFRAME_H6:
        local_start_date = end_date - timedelta(hours=const_size_down * 6)
    elif timeframe == mt5.TIMEFRAME_H8:
        local_start_date = end_date - timedelta(hours=const_size_down * 8)
    elif timeframe == mt5.TIMEFRAME_H12:
        local_start_date = end_date - timedelta(hours=const_size_down * 12)
    elif timeframe == mt5.TIMEFRAME_D1:
        local_start_date = end_date - timedelta(weeks=const_size_down // 2)
    else:
        raise ValueError(f'CUSTOM_ERROR: Not timeframe {timeframe}')

    return local_start_date


def timeframe_string_to_minutes(timeframe_str):
    match = re.match(r'([A-Za-z]+)(\d+)', timeframe_str)
    if match:
        unit = match.group(1).upper()
        value = int(match.group(2))
        if unit == "M":
            return value
        elif unit == "H":
            return value * 60
        elif unit == "D":
            return value * 1440
        elif unit == "W":
            return value * 10080
        elif unit == "MN":
            return value * 43200
    return None


def minutes_to_timeframe_string(minutes):
    if minutes < 1:
        return None
    if minutes % 43200 == 0:
        return f"{minutes // 43200}MN"
    elif minutes % 10080 == 0:
        return f"{minutes // 10080}W"
    elif minutes % 1440 == 0:
        return f"{minutes // 1440}D"
    elif minutes % 60 == 0:
        return f"{minutes // 60}H"
    else:
        return f"{minutes}M"


def adjust_weekend_to_next_tuesday(input_date):
    if input_date.weekday() in [5, 6]:
        while input_date.weekday() != 1:
            input_date += timedelta(days=1)
    return input_date


def check_actual_dataframe(df, minutes, name_date_column='DATETIME', timezone_hours=0):
    if name_date_column not in df.columns:
        raise ValueError(f"{name_date_column} not found in DataFrame")
    last_datetime = df['DATETIME'].max()
    if isinstance(last_datetime, str):
        last_datetime = dateutil.parser.parse(last_datetime)
    current_datetime = datetime.now(pytz.utc) + timedelta(hours=timezone_hours)
    current_datetime = current_datetime.replace(tzinfo=None)
    time_difference = (current_datetime - last_datetime).total_seconds() / 60
    if time_difference <= minutes:
        return True
    else:
        return False


def get_date_latest_data(df, name_date_column='DATETIME', string=False):
    df = df.sort_values(by=name_date_column)
    last_row = df.iloc[-1]
    date_string = f"{last_row[name_date_column]}"
    if string:
        return date_string
    else:
        date_obj = dateutil.parser.parse(date_string)
        return date_obj


def drop_dupl_class(df, min_periods=10):
    full_class_list = [_ for _ in df.columns if 'CLASS_' in _ or 'REGRESS_' in _]
    for name_class in full_class_list:

        mask = df.duplicated(subset=['DATETIME', 'SYMBOL', name_class]) & (df[name_class] == 1.0)
        df = df.loc[~mask]

        df['DUPL'] = df[name_class].shift(-1)
        df['DATETIME_NEXT'] = df['DATETIME'].shift(-1)
        df = df[(df['DATETIME'] + pd.Timedelta(minutes=min_periods) <= df['DATETIME_NEXT']) | (df[name_class] == 0.0) | (
                    df['DUPL'] == 0.0)]

    df = df.drop(columns=['DUPL', 'DATETIME_NEXT'])
    return df
