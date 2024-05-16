import datetime
import os
import config
import joblib
import pandas as pd
import MetaTrader5 as mt5
from dataclasses import dataclass
from utils.builder import datetime_now_timezone


@dataclass
class Order:

    symbol: str
    direction: str
    volume: float
    stop_loss: float
    take_profit: float
    datetime_open: any = None
    datetime_close: any = None
    order_ticket: any = None
    datetime_kill: any = None


class Journal:

    def __init__(self,
                 max_open_orders=None):
        self.max_open_orders = max_open_orders
        self.open_orders = []

    def _get_filter_orders(self, symbol=None, direction=None):
        open_orders_symbol = self.open_orders
        if symbol is not None:
            open_orders_symbol = [o for o in open_orders_symbol if o.symbol == symbol]
        if direction is not None:
            open_orders_symbol = [o for o in open_orders_symbol if o.direction.lower() == direction.lower()]
        return open_orders_symbol

    def create_order(self, symbol, direction, volume, stop_loss, take_profit, datetime_kill=None):
        if self.max_open_orders is not None:
            if self.max_open_orders <= len(self.open_orders):
                return None
        if 0 < len([i for i in self.open_orders if i.symbol == symbol]):
            return None
        new_order = Order(symbol=symbol,
                          direction=direction,
                          volume=volume,
                          stop_loss=stop_loss,
                          take_profit=take_profit,
                          datetime_kill=datetime_kill)
        new_order_ticket = mt5_create_position(order=new_order)
        if new_order_ticket is None:
            return None
        new_order.order_ticket = new_order_ticket
        new_order.datetime_open = datetime_now_timezone(config.WORK_TIMEZONE, delta=False)
        self.open_orders.append(new_order)
        return new_order

    def update_orders(self, symbol, direction=None, datetime_kill=None):
        open_orders_symbol = self._get_filter_orders(symbol=symbol, direction=direction)
        for iter_order in open_orders_symbol:
            if datetime_kill is not None:
                iter_order.datetime_kill = datetime_kill

    def close_orders(self, symbol, direction=None, pips_profit=None, only_profit=False):
        open_orders_symbol = self._get_filter_orders(symbol=symbol, direction=direction)
        for iter_order in open_orders_symbol:
            if pips_profit is not None:
                if mt5_check_position(iter_order).profit < pips_profit:
                    continue
            if only_profit:
                if mt5_check_position(iter_order).profit < 0:
                    continue
            mt5_close_position(order=iter_order)

    def check_orders(self):
        self.open_orders = [o for o in self.open_orders if mt5_check_position(o) is not None]

        for iter_order in self.open_orders:
            if iter_order.datetime_kill is not None:
                if iter_order.datetime_kill < datetime_now_timezone(config.WORK_TIMEZONE, delta=False):
                    mt5_close_position(order=iter_order)


def load_journal(path_file):
    if not os.path.exists(path_file):
        return Journal(max_open_orders=config.TRADE_MAX_OPEN_ORDERS)
    return joblib.load(path_file)


def save_journal(journal, pathfile):
    joblib.dump(journal, pathfile)


def inverse_direction(direction, lower=False):
    if direction.lower() == 'up':
        res = 'DOWN'
    elif direction.lower() == 'down':
        res = 'UP'
    else:
        res = ''
    if lower:
        res = res.lower()
    return res


def mt5_create_position(order):
    symbol_broker = order.symbol + config.PREFIX_BROKER_SYMBOL

    if mt5.initialize():

        if order.direction.lower() == 'up':
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol_broker).ask
        elif order.direction.lower() == 'down':
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol_broker).bid
        else:
            raise TypeError(f'unknown order.direction {order.direction}')

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol_broker,
            "volume": order.volume,
            "type": order_type,
            "price": price,
            "comment": "GrigoriyAutoManager"
        }
        if order.stop_loss is not None:
            request['sl'] = order.stop_loss
        if order.take_profit is not None:
            request['tp'] = order.take_profit

        order_result = mt5.order_send(request)
        print(datetime_now_timezone(config.WORK_TIMEZONE, delta=False), request)
        if order_result.retcode != mt5.TRADE_RETCODE_DONE:
            print("Error placing order: ", order_result.comment)
            return None
        else:
            return order_result.order


def mt5_close_position(order):
    symbol_broker = order.symbol + config.PREFIX_BROKER_SYMBOL
    if mt5.initialize():
        mt5.Close(symbol_broker, ticket=order.order_ticket)


def mt5_check_position(order):
    if mt5.initialize():
        info_order = mt5.positions_get(ticket=order.order_ticket)
        if len(info_order) < 1:
            return None
        else:
            return info_order[0]


def mt5_get_spread(order=None, symbol=None, points=False):
    spread = None
    if order is not None:
        symbol_broker = order.symbol + config.PREFIX_BROKER_SYMBOL
    elif symbol is not None:
        symbol_broker = symbol + config.PREFIX_BROKER_SYMBOL
    else:
        symbol_broker = None
    if mt5.initialize() and symbol_broker is not None:
        syb_info = mt5.symbol_info(symbol_broker)
        spread = syb_info.spread
        mt5.shutdown()
        if not points:
            spread = spread * syb_info.point
    if spread is None:
        raise ValueError('MetaTrader5: spread is None')
    return spread


def mt5_mean_spread(symbol=None):
    symbol_broker = symbol + config.PREFIX_BROKER_SYMBOL
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=100)
    if mt5.initialize():
        point_size = mt5.symbol_info(symbol_broker).point
        df = pd.DataFrame(mt5.copy_rates_range(symbol_broker, mt5.TIMEFRAME_H1, start_date, end_date))
        mt5.shutdown()
        return df["spread"].mean() * point_size
    else:
        return None

