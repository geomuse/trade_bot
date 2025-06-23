import backtrader as bt
from binance.client import Client
from config import API_KEY, API_SECRET, SYMBOL

class BinanceBroker(bt.brokers.BackBroker):
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET)
        super().__init__()

    def buy(self, size, price=None):
        order = self.client.futures_create_order(
            symbol=SYMBOL,
            side='BUY',
            type='MARKET',
            quantity=size
        )
        return order

    def sell(self, size, price=None):
        order = self.client.futures_create_order(
            symbol=SYMBOL,
            side='SELL',
            type='MARKET',
            quantity=size
        )
        return order
