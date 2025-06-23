import backtrader as bt
import pandas as pd
from strategy import ProGridStrategy , XGBGridProStrategy , RFGridStrategy , ReinvestRFGridStrategy # 你的策略文件
from binance_data import get_binance_ohlcv, BinanceData
from binance.client import Client
from config import API_KEY, API_SECRET

df = get_binance_ohlcv()
df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]
df = df.dropna()
# data = BinanceData(dataname=df)

client = Client(API_KEY, API_SECRET)
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
client.FUTURES_API_URL = 'https://testnet.binancefuture.com/fapi'

class PandasFuturesData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
    )

# 3. 初始化Cerebro
cerebro = bt.Cerebro()
data = PandasFuturesData(dataname=df)
cerebro.adddata(data)
cerebro.addstrategy(ReinvestRFGridStrategy, client=None)  # 不要传client参数
cerebro.broker.setcash(1000)      # 初始资金
cerebro.broker.setcommission(commission=0.0004)  # 设定手续费
# cerebro.addsizer(bt.sizers.FixedSize, stake=5)   # 每次买1份

# 4. 运行回测
print('Starting Portfolio Value: %.3f' % cerebro.broker.getvalue())
cerebro.run()
print('Final Portfolio Value: %.3f' % cerebro.broker.getvalue())
cerebro.plot()

