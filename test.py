from binance.client import Client
import time

# API_KEY = '你的futures_testnet_api_key'
# API_SECRET = '你的futures_testnet_api_secret'

from config import API_KEY, API_SECRET

client = Client(API_KEY, API_SECRET)
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
client.FUTURES_API_URL = 'https://testnet.binancefuture.com/fapi'

server_time = client.get_server_time()['serverTime'] // 1000
local_time = int(time.time())
print("本地时间:", local_time)
print("服务器时间:", server_time)
print("时间差（秒）:", abs(local_time - server_time))

print(client.futures_account())