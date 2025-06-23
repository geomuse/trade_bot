import backtrader as bt
from binance_data import get_binance_ohlcv, BinanceData
from binance.client import Client
from config import API_KEY, API_SECRET
from strategy import TestStrategy

if __name__ == '__main__':

    client = Client(API_KEY, API_SECRET)
    client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    client.FUTURES_API_URL = 'https://testnet.binancefuture.com/fapi'
    
    cerebro = bt.Cerebro()
    df = get_binance_ohlcv()
    data = BinanceData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy, client=client)
    cerebro.run()
