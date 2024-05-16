import pandas as pd
import MetaTrader5 as mt5
from utils import files
import dragonfly_data


def mt5_load_symbol(symbol, timeframe, path_out, end_date, len_input=None, start_date=None, prefix_broker=''):
    if not mt5.initialize():
        files.update_log(f'{"Error init MetaTrader 5 API"}\n')
        mt5.shutdown()
        raise Exception("Error init MetaTrader 5 API")

    if start_date is None:
        if len_input is None:
            raise ValueError('CUSTOM_ERROR: If start_date is empty, enter the len_input')
        else:
            local_start_date = dragonfly_data.utils.get_start_date_update(end_date=end_date,
                                                                          timeframe=timeframe,
                                                                          max_len_input=len_input)
    else:
        local_start_date = start_date

    mt5_timeframe = dragonfly_data.utils.string_to_mt5_timeframe(timeframe)
    df = pd.DataFrame(mt5.copy_rates_range(f'{symbol}{prefix_broker}', mt5_timeframe, local_start_date, end_date))
    if not df.empty:
        df = dragonfly_data.convertor.from_MT5(df)

        file_name = f"{path_out}/{symbol}_{timeframe}.csv"
        files.track_dir(path=file_name)
        df.to_csv(file_name, index=False)

    mt5.shutdown()
