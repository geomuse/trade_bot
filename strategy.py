import backtrader as bt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import xgboost as xgb

class ProGridStrategy(bt.Strategy):
    params = (
        ('grid_num', 10),           # 网格数量
        ('grid_width_pct', 0.01),   # 每格宽度（百分比）
        ('order_size', 0.001),      # 每格下单数量
        ('long_take_profit_pct', 0.02),
        ('long_stop_loss_pct', 0.01),
        ('short_take_profit_pct', 0.015),
        ('short_stop_loss_pct', 0.01),
        ('max_position_size', 0.01),  # 最大持仓
        ('reset_grid_pct', 0.02),   # 动态调整网格的触发百分比
    )

    def __init__(self, client=None):
        self.client = client
        self.last_grid = None
        self.entry_price = None
        self.grid_center = self.datas[0].close[0]
        self.update_grid()

    def update_grid(self):
        # 动态调整网格上下限
        self.grid_size = self.grid_center * self.p.grid_width_pct
        self.grid_upper = self.grid_center + (self.p.grid_num // 2) * self.grid_size
        self.grid_lower = self.grid_center - (self.p.grid_num // 2) * self.grid_size

    def next(self):
        price = self.datas[0].close[0]

        # 动态调整网格区间
        if price > self.grid_upper or price < self.grid_lower:
            if self.position:
                self.close()
                if self.client:
                    self.client.futures_create_order(
                        symbol='BTCUSDT',
                        side='SELL' if self.position.size > 0 else 'BUY',
                        type='MARKET',
                        quantity=abs(self.position.size)
                    )
                print(f"区间外强制平仓，价格: {price}")
            self.grid_center = price
            self.update_grid()
            self.last_grid = None
            self.entry_price = None
            return

        grid_level = int((price - self.grid_lower) / self.grid_size)
        grid_level = max(0, min(self.p.grid_num - 1, grid_level))

        # 首次进入网格
        if self.last_grid is None:
            self.last_grid = grid_level
            return

        # 止盈止损逻辑
        if self.position:
            pnl = (price - self.entry_price) if self.position.size > 0 else (self.entry_price - price)
            pnl_pct = pnl / self.entry_price
            if self.position.size > 0:
                # 多头止盈止损
                if pnl_pct >= self.p.long_take_profit_pct or pnl_pct <= -self.p.long_stop_loss_pct:
                    self.close()
                    if self.client:
                        self.client.futures_create_order(
                            symbol='BTCUSDT',
                            side='SELL',
                            type='MARKET',
                            quantity=abs(self.position.size)
                        )
                    print(f"多头平仓，价格: {price}，盈亏: {pnl_pct:.2%}")
                    self.last_grid = grid_level
                    self.entry_price = None
                    return
            else:
                # 空头止盈止损
                if pnl_pct >= self.p.short_take_profit_pct or pnl_pct <= -self.p.short_stop_loss_pct:
                    self.close()
                    if self.client:
                        self.client.futures_create_order(
                            symbol='BTCUSDT',
                            side='BUY',
                            type='MARKET',
                            quantity=abs(self.position.size)
                        )
                    print(f"空头平仓，价格: {price}，盈亏: {pnl_pct:.2%}")
                    self.last_grid = grid_level
                    self.entry_price = None
                    return

        # 最大持仓限制
        if abs(self.position.size) >= self.p.max_position_size:
            return

        # 网格交易逻辑
        if grid_level < self.last_grid:
            # 下穿网格，买入（开多）
            self.buy(size=self.p.order_size)
            self.entry_price = price
            if self.client:
                self.client.futures_create_order(
                    symbol='BTCUSDT',
                    side='BUY',
                    type='MARKET',
                    quantity=self.p.order_size
                )
            print(f"开多，价格: {price}")
            self.last_grid = grid_level

        elif grid_level > self.last_grid:
            # 上穿网格，卖出（开空）
            self.sell(size=self.p.order_size)
            self.entry_price = price
            if self.client:
                self.client.futures_create_order(
                    symbol='BTCUSDT',
                    side='SELL',
                    type='MARKET',
                    quantity=self.p.order_size
                )
            print(f"开空，价格: {price}")
            self.last_grid = grid_level
        
        # print(self.client.FUTURES_URL)

class LinRegGridProStrategy(bt.Strategy):
    params = (
        # 多头参数
        ('long_grid_num', 8),
        ('long_grid_width_pct', 0.003),
        ('long_order_size', 0.001),
        ('long_take_profit_pct', 0.01),
        ('long_stop_loss_pct', 0.01),
        # 空头参数
        ('short_grid_num', 8),
        ('short_grid_width_pct', 0.003),
        ('short_order_size', 0.001),
        ('short_take_profit_pct', 0.01),
        ('short_stop_loss_pct', 0.01),
        # 通用参数
        ('lookback', 20),
        ('max_position_size', 0.01),  # 最大持仓
        ('risk_pct', 0.1),            # 每次下单最大风险占总资金比例
    )

    def __init__(self, client=None):
        self.client = client
        self.dataclose = self.datas[0].close
        self.trend = None
        self.grid_center = self.dataclose[0]
        self.last_grid = None
        self.entry_price = None
        self.update_grid()

    def update_grid(self):
        # 根据当前趋势设置网格参数
        if self.trend == 1:  # 多头
            self.grid_num = self.p.long_grid_num
            self.grid_width_pct = self.p.long_grid_width_pct
        else:  # 空头
            self.grid_num = self.p.short_grid_num
            self.grid_width_pct = self.p.short_grid_width_pct
        self.grid_size = self.grid_center * self.grid_width_pct
        self.grid_upper = self.grid_center + (self.grid_num // 2) * self.grid_size
        self.grid_lower = self.grid_center - (self.grid_num // 2) * self.grid_size

    def next(self):
        price = self.dataclose[0]

        # 1. 线性回归判断趋势
        if len(self.dataclose) >= self.p.lookback + 1:
            y = np.array(self.dataclose.get(size=self.p.lookback))
            X = np.arange(self.p.lookback).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict(np.array([[self.p.lookback]]))[0]
            new_trend = 1 if pred > price else -1

            # 趋势反转，平掉所有持仓，重置网格
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
                self.trend = new_trend
                self.update_grid()
                self.last_grid = None
                self.entry_price = None
                return
            self.trend = new_trend
            self.update_grid()

        # 2. 动态调整网格区间
        if price > self.grid_upper or price < self.grid_lower:
            self.grid_center = price
            self.update_grid()
            self.last_grid = None
            return

        grid_level = int((price - self.grid_lower) / self.grid_size)
        grid_level = max(0, min(self.grid_num - 1, grid_level))

        # 3. 止盈止损风控
        if self.position:
            pnl = (price - self.entry_price) if self.position.size > 0 else (self.entry_price - price)
            pnl_pct = pnl / self.entry_price
            if self.position.size > 0:
                # 多头止盈止损
                if pnl_pct >= self.p.long_take_profit_pct or pnl_pct <= -self.p.long_stop_loss_pct:
                    self.close()
                    if self.client:
                        self.client.futures_create_order(
                            symbol='BTCUSDT',
                            side='SELL',
                            type='MARKET',
                            quantity=abs(self.position.size)
                        )
                    self.last_grid = grid_level
                    self.entry_price = None
                    return
            else:
                # 空头止盈止损
                if pnl_pct >= self.p.short_take_profit_pct or pnl_pct <= -self.p.short_stop_loss_pct:
                    self.close()
                    if self.client:
                        self.client.futures_create_order(
                            symbol='BTCUSDT',
                            side='BUY',
                            type='MARKET',
                            quantity=abs(self.position.size)
                        )
                    self.last_grid = grid_level
                    self.entry_price = None
                    return

        # 4. 最大持仓风控
        if abs(self.position.size) >= self.p.max_position_size:
            return

        # 5. 资金管理（动态调整下单量）
        # 以账户总资金的risk_pct为最大单笔风险
        cash = self.broker.getvalue()
        order_size = self.p.long_order_size if self.trend == 1 else self.p.short_order_size
        max_order_size = cash * self.p.risk_pct / price
        order_size = min(order_size, max_order_size)

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

class XGBGridProStrategy(bt.Strategy):
    params = (
        # 多头参数
        ('long_grid_num', 8),
        ('long_grid_width_pct', 0.003),
        ('long_order_size', 0.001),
        ('long_take_profit_pct', 0.01),
        ('long_stop_loss_pct', 0.01),
        # 空头参数
        ('short_grid_num', 8),
        ('short_grid_width_pct', 0.003),
        ('short_order_size', 0.001),
        ('short_take_profit_pct', 0.01),
        ('short_stop_loss_pct', 0.01),
        # 通用参数
        ('lookback', 20),
        ('max_position_size', 0.01),  # 最大持仓
        ('risk_pct', 0.1),            # 每次下单最大风险占总资金比例
    )

    def __init__(self, client=None):
        self.client = client
        self.dataclose = self.datas[0].close
        self.trend = None
        self.grid_center = self.dataclose[0]
        self.last_grid = None
        self.entry_price = None
        self.update_grid()

    def update_grid(self):
        # 根据当前趋势设置网格参数
        if self.trend == 1:  # 多头
            self.grid_num = self.p.long_grid_num
            self.grid_width_pct = self.p.long_grid_width_pct
        else:  # 空头
            self.grid_num = self.p.short_grid_num
            self.grid_width_pct = self.p.short_grid_width_pct
        self.grid_size = self.grid_center * self.grid_width_pct
        self.grid_upper = self.grid_center + (self.grid_num // 2) * self.grid_size
        self.grid_lower = self.grid_center - (self.grid_num // 2) * self.grid_size

    def next(self):
        if len(self.dataclose) < self.p.lookback + 2:
            return  # 数据不足

        # 构造特征和标签
        X = np.array([self.dataclose.get(size=self.p.lookback + 1)[1:]]).reshape(1, -1)  # 前lookback根
        y = np.array([self.dataclose.get(size=self.p.lookback + 1)[0]])  # 当前K线的close

        # 训练XGBoost
        model = xgb.XGBRegressor(n_estimators=20, max_depth=3, verbosity=0)
        model.fit(X, y)

        # 用最近lookback根K线预测下一根K线
        X_pred = np.array([self.dataclose.get(size=self.p.lookback)]).reshape(1, -1)
        pred = model.predict(X_pred)[0]
        price = self.dataclose[0]
        new_trend = 1 if pred > price else -1

        # 趋势反转，平掉所有持仓，重置网格
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
            self.trend = new_trend
            self.update_grid()
            self.last_grid = None
            self.entry_price = None
            return
        self.trend = new_trend
        self.update_grid()

        # 2. 动态调整网格区间
        if price > self.grid_upper or price < self.grid_lower:
            self.grid_center = price
            self.update_grid()
            self.last_grid = None
            return

        grid_level = int((price - self.grid_lower) / self.grid_size)
        grid_level = max(0, min(self.grid_num - 1, grid_level))

        # 3. 止盈止损风控
        if self.position:
            pnl = (price - self.entry_price) if self.position.size > 0 else (self.entry_price - price)
            pnl_pct = pnl / self.entry_price
            if self.position.size > 0:
                # 多头止盈止损
                if pnl_pct >= self.p.long_take_profit_pct or pnl_pct <= -self.p.long_stop_loss_pct:
                    self.close()
                    if self.client:
                        self.client.futures_create_order(
                            symbol='BTCUSDT',
                            side='SELL',
                            type='MARKET',
                            quantity=abs(self.position.size)
                        )
                    self.last_grid = grid_level
                    self.entry_price = None
                    return
            else:
                # 空头止盈止损
                if pnl_pct >= self.p.short_take_profit_pct or pnl_pct <= -self.p.short_stop_loss_pct:
                    self.close()
                    if self.client:
                        self.client.futures_create_order(
                            symbol='BTCUSDT',
                            side='BUY',
                            type='MARKET',
                            quantity=abs(self.position.size)
                        )
                    self.last_grid = grid_level
                    self.entry_price = None
                    return

        # 4. 最大持仓风控
        if abs(self.position.size) >= self.p.max_position_size:
            return

        # 5. 资金管理（动态调整下单量）
        # 以账户总资金的risk_pct为最大单笔风险
        cash = self.broker.getvalue()
        order_size = self.p.long_order_size if self.trend == 1 else self.p.short_order_size
        max_order_size = cash * self.p.risk_pct / price
        order_size = min(order_size, max_order_size)

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

class RFGridStrategy(bt.Strategy):
    params = (
        ('lookback', 20),         # 回归窗口
        ('grid_num', 8),          # 网格数量
        ('grid_width_pct', 0.003),# 网格宽度百分比
        ('order_size', 0.001),    # 每格下单量
    )

    def __init__(self, client=None):
        self.client = client
        self.dataclose = self.datas[0].close
        self.trend = None
        self.grid_center = self.dataclose[0]
        self.update_grid()
        self.last_grid = None

    def update_grid(self):
        self.grid_size = self.grid_center * self.p.grid_width_pct
        self.grid_upper = self.grid_center + (self.p.grid_num // 2) * self.grid_size
        self.grid_lower = self.grid_center - (self.p.grid_num // 2) * self.grid_size

    def next(self):
        price = self.dataclose[0]

        # 1. 随机森林回归判断趋势
        if len(self.dataclose) >= self.p.lookback + 2:
            # 构造特征和标签（只用最近一组样本，防止索引越界）
            X = np.array([self.dataclose.get(size=self.p.lookback + 1)[1:]]).reshape(1, -1)
            y = np.array([self.dataclose.get(size=self.p.lookback + 1)[0]])

            model = RandomForestRegressor(n_estimators=20, max_depth=3)
            model.fit(X, y)

            # 用最近lookback根K线预测下一根K线
            X_pred = np.array([self.dataclose.get(size=self.p.lookback)]).reshape(1, -1)
            pred = model.predict(X_pred)[0]
            new_trend = 1 if pred > price else -1

            # 趋势反转，平掉所有持仓，重置网格
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

            self.trend = new_trend

        # 2. 动态调整网格区间
        if price > self.grid_upper or price < self.grid_lower:
            self.grid_center = price
            self.update_grid()
            self.last_grid = None
            return

        grid_level = int((price - self.grid_lower) / self.grid_size)
        grid_level = max(0, min(self.p.grid_num - 1, grid_level))

        # 3. 网格交易（只做趋势方向）
        if self.last_grid is None:
            self.last_grid = grid_level
            return

        if self.trend == 1:  # 多头趋势，只做多网格
            if grid_level < self.last_grid:
                self.buy(size=self.p.order_size)
                if self.client:
                    self.client.futures_create_order(
                        symbol='BTCUSDT',
                        side='BUY',
                        type='MARKET',
                        quantity=self.p.order_size
                    )
                self.last_grid = grid_level
        elif self.trend == -1:  # 空头趋势，只做空网格
            if grid_level > self.last_grid:
                self.sell(size=self.p.order_size)
                if self.client:
                    self.client.futures_create_order(
                        symbol='BTCUSDT',
                        side='SELL',
                        type='MARKET',
                        quantity=self.p.order_size
                    )
                self.last_grid = grid_level