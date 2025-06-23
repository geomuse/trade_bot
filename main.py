import backtrader as bt
from binance_data import get_binance_ohlcv, BinanceData
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import API_KEY, API_SECRET
from strategy_ import ReinvestRFGridStrategy
import time
import logging

def run_strategy():
    client = Client(API_KEY, API_SECRET)
    client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    client.FUTURES_API_URL = 'https://testnet.binancefuture.com/fapi'
    
    cerebro = bt.Cerebro()
    df = get_binance_ohlcv()
    data = BinanceData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(ReinvestRFGridStrategy, client=client)
    cerebro.run()
# 11000
if __name__ == '__main__':

    while True:
        try:
            print("🤖 Bot 正在运行...")
            run_strategy()
            time.sleep(1)  
        except BinanceAPIException as e:
            logging.error(f"Binance API异常: {e}")
            time.sleep(10)
        except Exception as e:
            logging.error(f"其他异常: {e}")
            time.sleep(10)