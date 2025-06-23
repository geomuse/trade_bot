import backtrader as bt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import numpy as np

class ReinvestRFGridStrategy(bt.Strategy):
    params = (
        ('lookback', 20),         # 特征窗口
        ('grid_num', 8),          # 网格数量
        ('grid_width_pct', 0.003),# 网格宽度百分比
        ('risk_pct', 0.1),        # 每次下单最大风险占总资金比例
        ('take_profit_pct', 0.1),# 止盈百分比
        ('stop_loss_pct', 0.01),  # 止损百分比
        ('max_position_pct', 0.5),# 最大持仓占总资金比例
    )

    def __init__(self, client=None):
        self.client = client
        self.dataclose = self.datas[0].close

        # 技术指标
        self.tema = bt.ind.TEMA(self.datas[0], period=10)
        self.rsi = bt.ind.RSI(self.datas[0], period=14)
        # self.macd = bt.ind.MACD(self.datas[0], period_me1=12, period_me2=26, period_signal=9)

        self.trend = None
        self.grid_center = self.dataclose[0]
        self.update_grid()
        self.last_grid = None
        self.entry_price = None

    def update_grid(self):
        self.grid_size = self.grid_center * self.p.grid_width_pct
        self.grid_upper = self.grid_center + (self.p.grid_num // 2) * self.grid_size
        self.grid_lower = self.grid_center - (self.p.grid_num // 2) * self.grid_size

    @staticmethod
    def format_quantity(symbol, qty, client):
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                precision = s['quantityPrecision']
                return float(f"{qty:.{precision}f}")
        return qty

    def format_price(symbol, price, client):
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                precision = s['pricePrecision']
                return float(f"{price:.{precision}f}")
        return price

    def next(self):
        price = self.dataclose[0]

        # 1. 随机森林预测趋势
        if len(self.dataclose) >= self.p.lookback + 2:
            X = []
            y = []
            for i in range(self.p.lookback, 0, -1):
                X.append([
                    float(self.tema[-i]),
                    float(self.rsi[-i]),
                    # float(self.macd.macd[-i]),
                    # float(self.macd.signal[-i]),
                    # float(self.macd.lines[2][-i])  # MACD直方图
                ])
                y.append(float(self.dataclose[-i+1]))  # 下一根K线的close

            X = np.array(X)
            y = np.array(y)
            X_train = X[-100:] if len(X) > 100 else X
            y_train = y[-100:] if len(y) > 100 else y

            model = RandomForestRegressor(n_estimators=20, max_depth=3)
            model.fit(X_train, y_train)

            X_pred = np.array([[
                float(self.tema[0]),
                float(self.rsi[0]),
                # float(self.macd.macd[0]),
                # float(self.macd.signal[0]),
                # float(self.macd.lines[2][0])
            ]]).reshape(1, -1)
            pred = model.predict(X_pred)[0]
            new_trend = 1 if pred > price else -1

            # 趋势反转，平仓重置网格
            if self.trend is not None and new_trend != self.trend:
                self.close()
                if self.client and self.position:
                    self.client.futures_create_order(
                        symbol='BTCUSDT',
                        side='SELL' if self.position.size > 0 else 'BUY',
                        type='MARKET',
                        quantity=abs(self.position.size)
                    )
                self.grid_center = price
                self.update_grid()
                self.last_grid = None
                self.entry_price = None

            self.trend = new_trend

        # 2. 动态调整网格区间
        if price > self.grid_upper or price < self.grid_lower:
            self.grid_center = price
            self.update_grid()
            self.last_grid = None
            return

        grid_level = int((price - self.grid_lower) / self.grid_size)
        grid_level = max(0, min(self.p.grid_num - 1, grid_level))

        # 3. 止盈止损风控
        if self.position and self.entry_price is not None:
            pnl = (price - self.entry_price) if self.position.size > 0 else (self.entry_price - price)
            pnl_pct = pnl / self.entry_price
            if pnl_pct >= self.p.take_profit_pct or pnl_pct <= -self.p.stop_loss_pct:
                self.close()
                if self.client:
                    self.client.futures_create_order(
                        symbol='SELL' if self.position.size > 0 else 'BUY',
                        type='MARKET',
                        quantity=abs(self.position.size)
                    )
                self.last_grid = grid_level
                self.entry_price = None
                return

        # 4. 最大持仓风控
        cash = self.broker.getvalue()
        max_position = cash * self.p.max_position_pct / price
        if abs(self.position.size) >= max_position:
            return

        # 5. 再投资（复利）动态下单量
        order_size = self.format_quantity('BTCUSDT', cash * self.p.risk_pct / price, self.client)

        # 6. 网格交易（只做趋势方向）
        if self.last_grid is None:
            self.last_grid = grid_level
            return

        if self.trend == 1:  # 多头趋势，只做多网格
            if grid_level < self.last_grid:
                self.buy(size=order_size)
                self.entry_price = price
                if self.client:
                    self.client.futures_create_order(
                        symbol='BTCUSDT',
                        side='BUY',
                        type='MARKET',
                        quantity=order_size
                    )
                self.last_grid = grid_level
        elif self.trend == -1:  # 空头趋势，只做空网格
            if grid_level > self.last_grid:
                self.sell(size=order_size)
                self.entry_price = price
                if self.client:
                    self.client.futures_create_order(
                        symbol='BTCUSDT',
                        side='SELL',
                        type='MARKET',
                        quantity=order_size
                    )
                self.last_grid = grid_level

        min_period = max(10, 14, 26)  # TEMA, RSI, MACD
        if len(self.dataclose) < min_period + self.p.lookback + 2:
            return

