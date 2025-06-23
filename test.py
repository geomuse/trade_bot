from binance.client import Client

API_KEY = '你的futures_testnet_api_key'
API_SECRET = '你的futures_testnet_api_secret'

from config import API_KEY, API_SECRET

client = Client(API_KEY, API_SECRET)
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
client.FUTURES_API_URL = 'https://testnet.binancefuture.com/fapi'

print(client.futures_account())