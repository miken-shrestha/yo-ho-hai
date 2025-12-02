import requests
import time

base = 'http://127.0.0.1:8000'

payload = {
    'usd_rate': 141.9,
    'inflation_rate': 0.0417,
    'gold_price_usd': 4000,
    'interest_rate': 0.06,
    'festivals': 0,
    'nepse_index': 2600
}

# wait for server to be ready with retries
for attempt in range(1, 8):
    try:
        r = requests.get(base + '/fetch-live-data/', timeout=3)
        print('\n[fetch-live-data] status:', r.status_code)
        print(r.text)
        break
    except requests.exceptions.RequestException as e:
        print(f'Attempt {attempt}: server not ready ({e}), sleeping 1s')
        time.sleep(1)
else:
    print('Server did not become ready; aborting tests')
    raise SystemExit(1)

# POST /predict
try:
    r = requests.post(base + '/predict/', json=payload, timeout=5)
    print('\n[predict] status:', r.status_code)
    print(r.text)
except Exception as e:
    print('[predict] request failed:', e)

# POST /predict-7-days
try:
    r = requests.post(base + '/predict-7-days/', json=payload, timeout=8)
    print('\n[predict-7-days] status:', r.status_code)
    print(r.text)
except Exception as e:
    print('[predict-7-days] request failed:', e)
