#!/usr/bin/env python3
"""🌊 إليوت 5 موجات — LIVE daemon (3m, 212 coins)
   نموذج Elliot 5-Wave هابطة + w5=0.382(w1+w3)
   خروج: نصف عند فيبو 0.5 للموجة 5 + نصف عند فيبو 1.0 (H3) + BE"""
import ccxt, json, time, os, sys
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag

# ═══ CONFIG ═══
EXCHANGE_ID = 'binance'
TIMEFRAME = '3m'
SCAN_INTERVAL = 180
MAX_POS = 1  # 100% risk — صفقة وحدة فقط
COMM = 0.20

DEPTH = 10; DEV = 1.0; D = DEPTH // 2; CONFIRM = D
DIST_FILTER = 0.5; TIME_BARS = 120; INIT_SL_PCT = -0.5

BASE_DIR = '/data/trading28'
CACHE_DIR = f'{BASE_DIR}/cache/elliot_live'
STATE_FILE = f'{BASE_DIR}/elliot_state.json'
REPORT_FILE = f'{BASE_DIR}/elliot_report.txt'
COINS_FILE = f'{BASE_DIR}/config/shariah_coins.json'

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
os.makedirs(CACHE_DIR, exist_ok=True)

# ═══ Exchange ═══
_exchange = None
def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binance({'timeout': 15000, 'options': {'defaultType': 'spot'}})
    return _exchange

def get_live_price(symbol):
    try:
        return get_exchange().fetch_ticker(symbol)['last']
    except:
        return None

def sleep_until_next_3m():
    now = datetime.now(timezone.utc)
    secs = now.minute * 60 + now.second + now.microsecond / 1_000_000
    wait = ((int(secs) // 180) + 1) * 180 - secs
    if wait > 0: time.sleep(wait)

def fetch_and_cache(symbol):
    fpath = f'{CACHE_DIR}/{symbol.replace("/","")}.json'
    cache = []
    if os.path.exists(fpath):
        with open(fpath) as f: cache = json.load(f)
    
    exchange = get_exchange()
    since = cache[-1]['ts'] + 1 if cache else None
    
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=500, since=since)
    except:
        if cache:
            import pandas as pd
            df = pd.DataFrame(cache)
            return df['close'].values, df['high'].values, df['low'].values
        return np.array([]), np.array([]), np.array([])
    
    if raw:
        new_candles = [{'ts': int(r[0]), 'o': r[1], 'h': r[2], 'l': r[3], 'c': r[4], 'v': r[5]} for r in raw]
        if cache and new_candles[0]['ts'] <= cache[-1]['ts']:
            cache = [c for c in cache if c['ts'] < new_candles[0]['ts']]
        cache.extend(new_candles)
        cache = cache[-500:]
        with open(fpath, 'w') as f: json.dump(cache, f)
    elif not cache:
        return np.array([]), np.array([]), np.array([])
    
    import pandas as pd
    df = pd.DataFrame(cache).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    return df['close'].values, df['high'].values, df['low'].values

# ═══ Elliott 5-Wave Detection ═══
def near(v, target, tol=0.03):
    return abs(v - target) <= tol

def find_5waves(pv):
    """Find 5-wave Elliott patterns (downward: H→L→H→L→H→L) with w5=0.382(w1+w3)"""
    pats = []
    for i in range(len(pv) - 5):
        pts = pv[i:i+6]
        if [pt[2] for pt in pts] != ['H','L','H','L','H','L']: continue
        H1,L1,H2,L2,H3,L3 = pts
        w1 = H1[1] - L1[1]; w2 = H2[1] - L1[1]; w3 = H2[1] - L2[1]
        w4 = H3[1] - L2[1]; w5 = H3[1] - L3[1]
        if w1 <= 0 or w2 <= 0 or w3 <= 0 or w4 <= 0 or w5 <= 0: continue
        if w2 >= w1 or w3 <= min(w1, w5): continue
        if H3[1] >= L1[1] or L3[1] >= L2[1]: continue
        if not near(w5/(w1+w3), 0.382): continue
        pats.append((H1, L1, H2, L2, H3, L3))
    return pats

def check_entry(close, high, low, coin):
    n = len(close)
    if n < D + 10: return None
    
    pv = zigzag(high, low, depth=DEPTH, dev=DEV)
    if len(pv) < 6: return None
    
    pats = find_5waves(pv)
    if not pats: return None
    
    H1, L1, H2, L2, H3, L3 = pats[-1]
    entry_bar = L3[0] + CONFIRM
    
    if entry_bar != n - 1: return None
    
    entry_price = close[entry_bar]
    dist_pct = (entry_price - L3[1]) / L3[1] * 100
    if dist_pct > DIST_FILTER: return None
    
    w5_size = H3[1] - L3[1]
    if w5_size <= 0: return None
    
    # Fib targets from wave 5
    fib_half = L3[1] + 0.5 * w5_size   # Fib 0.5 of wave 5
    fib_full = H3[1]                    # Fib 1.0 = H3
    
    if fib_full <= entry_price: return None
    if fib_half <= entry_price: return None
    
    # SL = L3 - 0.5%
    sl_price = L3[1] * (1 + INIT_SL_PCT / 100)
    if sl_price >= entry_price: return None
    
    return {
        'entry_price': entry_price,
        'sl': sl_price,
        'tp_half': fib_half,    # Fib 0.5 of wave 5
        'tp_full': fib_full,     # Fib 1.0 = H3
        'be': entry_price,
        'entry_bar_ts': int(datetime.now(timezone.utc).timestamp() * 1000),
        'L3_price': L3[1],
        'H3_price': H3[1],
        'w5_size': round(w5_size, 8),
    }

# ═══ State ═══
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'active': [], 'closed': [], 'total_trades': 0, 'wins': 0}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2)

def used_entries(state):
    used = set()
    for p in state['active'] + state['closed']:
        if 'entry_bar_ts' in p:
            used.add((p['symbol'], p['entry_bar_ts']))
    return used

# ═══ Scan ═══
def scan():
    state = load_state()
    used = used_entries(state)
    active_symbols = {p['symbol'] for p in state['active']}
    
    with open(COINS_FILE) as f:
        sh = json.load(f)
    coins = [c for c in sh['halal'] + sh['halal2'] if c not in STABLES]
    
    events = []
    
    # Check active positions
    new_active = []
    for pos in state['active']:
        symbol = pos['symbol']
        price = get_live_price(f'{symbol}/USDT')
        if price is None:
            new_active.append(pos)
            continue
        
        entry = pos['entry_price']
        exit_reason = None
        exit_pnl = 0.0
        
        if not pos.get('half_exited'):
            # Check half TP (Fib 0.5)
            if price >= pos['tp_half']:
                pos['half_exited'] = True
                pos['half_pnl'] = round((pos['tp_half']/entry - 1)*100 - COMM/2, 4)
                if price >= pos['tp_full']:
                    half2 = round((pos['tp_full']/entry - 1)*100 - COMM/2, 4)
                    exit_reason = 'FULL'
                    exit_pnl = round((pos['half_pnl'] + half2) / 2, 4)
                else:
                    new_active.append(pos)
                    continue
            else:
                close_arr, _, _ = fetch_and_cache(symbol)
                if len(close_arr) >= 2 and close_arr[-2] <= pos['sl']:
                    exit_reason = 'SL'
                    exit_pnl = round((close_arr[-2]/entry - 1)*100 - COMM, 4)
                else:
                    new_active.append(pos)
                    continue
        else:
            # Half already taken — check full TP (Fib 1.0) or BE
            if price >= pos['tp_full']:
                half2 = round((pos['tp_full']/entry - 1)*100 - COMM/2, 4)
                exit_reason = 'FULL'
                exit_pnl = round((pos['half_pnl'] + half2) / 2, 4)
            else:
                close_arr, _, _ = fetch_and_cache(symbol)
                if len(close_arr) >= 2 and close_arr[-2] <= pos['be']:
                    half2 = -COMM/2
                    exit_reason = 'BE'
                    exit_pnl = round((pos['half_pnl'] + half2) / 2, 4)
                else:
                    now_ts = datetime.now(timezone.utc).timestamp() * 1000
                    age_bars = (now_ts - pos.get('entry_bar_ts', now_ts)) / (180 * 1000)
                    if age_bars >= TIME_BARS:
                        half2 = round((price/entry - 1)*100 - COMM/2, 4)
                        exit_reason = 'TIME'
                        exit_pnl = round((pos['half_pnl'] + half2) / 2, 4)
                    else:
                        new_active.append(pos)
                        continue
        
        pos['exit_price'] = price
        pos['exit_reason'] = exit_reason
        pos['pnl_net'] = exit_pnl
        pos['exit_time'] = datetime.now(timezone.utc).isoformat()
        state['closed'].append(pos)
        state['total_trades'] += 1
        if exit_pnl > 0: state['wins'] += 1
        
        emoji = '🟢' if exit_pnl > 0 else '🔴'
        events.append(f"{emoji} خروج {pos['symbol']} {exit_reason} | {exit_pnl:+.2f}% | سعر {price:.6f}")
    
    state['active'] = new_active
    
    # Scan for new entries
    slots = MAX_POS - len(state['active'])
    if slots > 0:
        for coin in coins:
            if slots <= 0: break
            symbol = f'{coin}/USDT'
            if coin in active_symbols: continue
            
            close_arr, high_arr, low_arr = fetch_and_cache(symbol)
            if len(close_arr) < 200: continue
            
            sig = check_entry(close_arr, high_arr, low_arr, coin)
            if sig is None: continue
            if (coin, sig['entry_bar_ts']) in used: continue
            
            pos = {
                'symbol': coin,
                'entry_price': sig['entry_price'],
                'sl': sig['sl'],
                'tp_half': sig['tp_half'],
                'tp_full': sig['tp_full'],
                'be': sig['be'],
                'entry_bar_ts': sig['entry_bar_ts'],
                'entry_time': datetime.now(timezone.utc).isoformat(),
                'half_exited': False,
                'half_pnl': 0.0,
                'L3_price': sig['L3_price'],
                'H3_price': sig['H3_price'],
                'w5_size': sig['w5_size'],
            }
            state['active'].append(pos)
            used.add((coin, sig['entry_bar_ts']))
            active_symbols.add(coin)
            slots -= 1
            
            events.append(f"🌊 دخول {coin} | سعر {sig['entry_price']:.6f} | SL {sig['sl']:.6f} | هدف {sig['tp_full']:.6f}")
    
    save_state(state)
    
    # Report
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [f'🌊 إليوت 5 موجات — {now}', '━' * 30]
    
    if state['active']:
        lines.append(f'\n📊 مفتوح ({len(state["active"])}):')
        for pos in state['active']:
            sym = pos['symbol']; ep = pos['entry_price']
            price = get_live_price(f'{sym}/USDT') or ep
            pnl = (price/ep - 1) * 100
            half = '✓' if pos.get('half_exited') else ' '
            w5 = pos.get('w5_size', 0)
            lines.append(f'  {sym}: {ep:.5f} | حالياً {price:.5f} | PnL {pnl:+.2f}% | نصف:{half} | w5:{w5:.4f}')
    
    if events:
        lines.append(f'\n📋 أحداث:')
        lines.extend(events)
    
    total = state['total_trades']
    wins = state['wins']
    wr = f'{wins/total*100:.1f}%' if total > 0 else '-'
    lines.append(f'\n📈 الإجمالي: {total} صفقة | ربح {wins} | WR {wr}')
    
    if state['closed']:
        lines.append(f'\n📜 آخر الصفقات:')
        for pos in state['closed'][-5:]:
            lines.append(f'  {pos["symbol"]} {pos["exit_reason"]} {pos["pnl_net"]:+.2f}%')
    
    report = '\n'.join(lines)
    with open(REPORT_FILE, 'w') as f: f.write(report)
    return report

# ═══ Main ═══
if __name__ == '__main__':
    print(f'🌊 إليوت 5 موجات — بدأ {datetime.now(timezone.utc).isoformat()}', flush=True)
    sleep_until_next_3m()
    
    while True:
        try:
            report = scan()
            print(report, flush=True)
        except Exception as e:
            print(f'❌ خطأ: {e}', flush=True)
            import traceback
            traceback.print_exc()
        
        sleep_until_next_3m()
