import random
import warnings
import numpy as np
import pandas as pd
import config
import utils as utl
import dragonfly_data
import dragonfly_trade
from ta.trend import adx, adx_neg, adx_pos

path_source = config.PATH_SOURCE_DATA
symbol = 'EURUSD'
sel_columns = config.SELECTED_COLUMNS

df = pd.read_csv(f'{path_source}/{symbol}_M1.csv')
df = dragonfly_data.convertor.to_dragonfly_format(dataframe=df)
df = dragonfly_data.builder.preparer_datetime(dataframe=df, column_time='DATETIME', indexing=False)

