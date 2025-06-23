import backtrader as bt
from binance.client import Client
import pandas as pd
from config import API_KEY, API_SECRET, SYMBOL, INTERVAL

def get_binance_ohlcv():
    client = Client(API_KEY, API_SECRET)
    client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    client.FUTURES_API_URL = 'https://testnet.binancefuture.com/fapi'
    klines = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=500)
    df = pd.DataFrame(klines, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    df = df.astype(float)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

class BinanceData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
    )