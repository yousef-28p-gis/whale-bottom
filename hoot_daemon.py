#!/usr/bin/env python3
"""🐋 حوت الموجات — ZigZag V-Shape LIVE daemon (3m, 210 coins)"""
import ccxt, json, time, os, sys
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, '/data/trading28')
from strategies.zigzag import zigzag

# ═══ CONFIG ═══
EXCHANGE_ID = 'binance'
TIMEFRAME = '3m'
SCAN_INTERVAL = 180  # 3 min
MAX_POS = 2
COMM = 0.20

# Strategy params
DEPTH = 10; DEV = 1.0; D = DEPTH // 2; CONFIRM = D
TP_PCT = 1.0; HALF_TP_PCT = 0.5; SL_PCT = -0.5; DIST_FILTER = 0.5
TIME_BARS = 120  # 6h max

# Paths
BASE_DIR = '/data/trading28'
CACHE_DIR = f'{BASE_DIR}/cache/hoot_live'
STATE_FILE = f'{BASE_DIR}/hoot_state.json'
REPORT_FILE = f'{BASE_DIR}/hoot_report.txt'
COINS_FILE = f'{BASE_DIR}/config/shariah_coins.json'

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
EXCLUDED = {'TRX'}  # <5% net in backtest

os.makedirs(CACHE_DIR, exist_ok=True)

# ═══ Exchange ═══
_exchange = None
def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binance({'timeout': 15000, 'options': {'defaultType': 'spot'}})
    return _exchange

# ═══ Utils ═══
def get_live_price(symbol):
    try:
        ticker = get_exchange().fetch_ticker(symbol)
        return ticker['last']
    except:
        return None

def sleep_until_next_3m():
    now = datetime.now(timezone.utc)
    secs = now.minute * 60 + now.second + now.microsecond / 1_000_000
    next_boundary = ((int(secs) // 180) + 1) * 180
    wait = next_boundary - secs
    if wait > 0:
        time.sleep(wait)

def fetch_and_cache(symbol):
    """Fetch live OHLCV and merge with cache. Returns DataFrame with close/high/low arrays."""
    fpath = f'{CACHE_DIR}/{symbol.replace("/","")}.json'
    cache = []
    if os.path.exists(fpath):
        with open(fpath) as f: cache = json.load(f)
    
    exchange = get_exchange()
    if cache:
        since = cache[-1]['ts'] + 1  # next ms after last cached
    else:
        since = None
    
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=500, since=since)
    except Exception as e:
        # Fallback: use cached only
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
        if len(cache) > 500:
            cache = cache[-500:]
        with open(fpath, 'w') as f: json.dump(cache, f)
    elif not cache:
        return np.array([]), np.array([]), np.array([])
    
    import pandas as pd
    df = pd.DataFrame(cache)
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    return df['close'].values, df['high'].values, df['low'].values

# ═══ Pattern Detection ═══
def find_zpatterns(pv):
    pats = []
    for i in range(len(pv)-3):
        p0,p1,p2,p3 = pv[i],pv[i+1],pv[i+2],pv[i+3]
        if p0[2]=='H' and p1[2]=='L' and p2[2]=='H' and p3[2]=='L':
            A=p0[1]-p1[1]; B=p2[1]-p1[1]; C=p2[1]-p3[1]
            if A>0 and B>0 and C>0 and 0.38<=B/A<=0.79 and p3[1]<p1[1]:
                pats.append((p0,p1,p2,p3))
    return pats

def check_entry(close, high, low, coin):
    """Check if there's a new V-Shape entry signal."""
    n = len(close)
    if n < D + 10: return None
    
    pv = zigzag(high, low, depth=DEPTH, dev=DEV)
    if len(pv) < 4: return None
    
    pats = find_zpatterns(pv)
    if not pats: return None
    
    # Get the most recent pattern
    H1, L1, H2, L2 = pats[-1]
    entry_bar = L2[0] + CONFIRM
    
    # Must be the very last candle (just closed)
    if entry_bar != n - 1: return None
    
    entry_price = close[entry_bar]
    dist_pct = (entry_price - L2[1]) / L2[1] * 100
    if dist_pct > DIST_FILTER: return None
    
    sl_price = L2[1] * (1 + SL_PCT / 100)
    if sl_price >= entry_price: return None
    
    return {
        'entry_price': entry_price,
        'sl': sl_price,
        'tp_full': entry_price * (1 + TP_PCT / 100),
        'tp_half': entry_price * (1 + HALF_TP_PCT / 100),
        'be': entry_price,
        'entry_bar_ts': int(datetime.now(timezone.utc).timestamp() * 1000),
        'L2_price': L2[1],
    }

# ═══ State Management ═══
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'active': [], 'closed': [], 'total_trades': 0, 'wins': 0}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2)

def used_entries(state):
    """Build set of (symbol, entry_bar_ts) already used."""
    used = set()
    for p in state['active'] + state['closed']:
        if 'entry_bar_ts' in p:
            used.add((p['symbol'], p['entry_bar_ts']))
    return used

# ═══ Main Scan ═══
def scan():
    state = load_state()
    used = used_entries(state)
    active_symbols = {p['symbol'] for p in state['active']}
    
    # Load coins
    with open(COINS_FILE) as f:
        sh = json.load(f)
    coins = [c for c in sh['halal'] + sh['halal2'] if c not in STABLES and c not in EXCLUDED]
    
    events = []
    
    # Check active positions first
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
        
        # Half TP check (if not yet half-exited)
        if not pos.get('half_exited'):
            if price >= pos['tp_half']:
                pos['half_exited'] = True
                pos['half_pnl'] = round((pos['tp_half']/entry - 1)*100 - COMM/2, 4)
                # Check if also hits full TP
                if price >= pos['tp_full']:
                    half2 = round((pos['tp_full']/entry - 1)*100 - COMM/2, 4)
                    exit_reason = 'TP'
                    exit_pnl = round((pos['half_pnl'] + half2) / 2, 4)
                else:
                    # Continue: half secured, move to BE
                    new_active.append(pos)
                    continue
            else:
                # Check SL (close-only — use last closed candle)
                close_arr, _, _ = fetch_and_cache(symbol)
                if len(close_arr) >= 2 and close_arr[-2] <= pos['sl']:
                    exit_reason = 'SL'
                    exit_pnl = round((close_arr[-2]/entry - 1)*100 - COMM, 4)
                else:
                    new_active.append(pos)
                    continue
        else:
            # Already half-exited — check full TP or BE
            if price >= pos['tp_full']:
                half2 = round((pos['tp_full']/entry - 1)*100 - COMM/2, 4)
                exit_reason = 'TP'
                exit_pnl = round((pos['half_pnl'] + half2) / 2, 4)
            else:
                close_arr, _, _ = fetch_and_cache(symbol)
                if len(close_arr) >= 2 and close_arr[-2] <= pos['be']:
                    half2 = round((pos['be']/entry - 1)*100 - COMM/2, 4)
                    exit_reason = 'BE'
                    exit_pnl = round((pos['half_pnl'] + half2) / 2, 4)
                else:
                    # Check timeout
                    now_ts = datetime.now(timezone.utc).timestamp() * 1000
                    age_bars = (now_ts - pos.get('entry_bar_ts', now_ts)) / (180 * 1000)
                    if age_bars >= TIME_BARS:
                        half2 = round((price/entry - 1)*100 - COMM/2, 4)
                        exit_reason = 'TIME'
                        exit_pnl = round((pos['half_pnl'] + half2) / 2, 4)
                    else:
                        new_active.append(pos)
                        continue
        
        # Exit triggered
        pos['exit_price'] = price
        pos['exit_reason'] = exit_reason
        pos['pnl_net'] = exit_pnl
        pos['exit_time'] = datetime.now(timezone.utc).isoformat()
        state['closed'].append(pos)
        state['total_trades'] += 1
        if exit_pnl > 0:
            state['wins'] += 1
        
        emoji = '🟢' if exit_pnl > 0 else '🔴'
        events.append(f"{emoji} خروج {pos['symbol']} {exit_reason} | {exit_pnl:+.2f}% | سعر {price:.6f}")
    
    state['active'] = new_active
    
    # Scan for new entries
    slots_free = MAX_POS - len(state['active'])
    if slots_free > 0:
        for coin in coins:
            if slots_free <= 0: break
            symbol = f'{coin}/USDT'
            if coin in active_symbols: continue
            
            close_arr, high_arr, low_arr = fetch_and_cache(symbol)
            if len(close_arr) < 200: continue
            
            signal = check_entry(close_arr, high_arr, low_arr, coin)
            if signal is None: continue
            if (coin, signal['entry_bar_ts']) in used: continue
            
            # New entry
            pos = {
                'symbol': coin,
                'entry_price': signal['entry_price'],
                'sl': signal['sl'],
                'tp_full': signal['tp_full'],
                'tp_half': signal['tp_half'],
                'be': signal['be'],
                'entry_bar_ts': signal['entry_bar_ts'],
                'entry_time': datetime.now(timezone.utc).isoformat(),
                'half_exited': False,
                'half_pnl': 0.0,
            }
            state['active'].append(pos)
            used.add((coin, signal['entry_bar_ts']))
            active_symbols.add(coin)
            slots_free -= 1
            
            events.append(f"🐋 دخول {coin} | سعر {signal['entry_price']:.6f} | SL {signal['sl']:.6f}")
    
    save_state(state)
    
    # ── Build report ──
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [f'🐋 حوت الموجات — {now}', '━' * 30]
    
    if state['active']:
        lines.append(f'\n📊 مفتوح ({len(state["active"])}):')
        for pos in state['active']:
            sym = pos['symbol']; ep = pos['entry_price']
            price = get_live_price(f'{sym}/USDT') or ep
            pnl = (price/ep - 1) * 100
            half = '✓' if pos.get('half_exited') else ' '
            lines.append(f'  {sym}: {ep:.5f} | حالياً {price:.5f} | PnL {pnl:+.2f}% | نصف:{half}')
    
    if events:
        lines.append(f'\n📋 أحداث:')
        lines.extend(events)
    
    total = state['total_trades']
    wins = state['wins']
    wr = f'{wins/total*100:.1f}%' if total > 0 else '-'
    lines.append(f'\n📈 الإجمالي: {total} صفقة | ربح {wins} | WR {wr}')
    
    # Last 5 closed
    if state['closed']:
        lines.append(f'\n📜 آخر الصفقات:')
        for pos in state['closed'][-5:]:
            lines.append(f'  {pos["symbol"]} {pos["exit_reason"]} {pos["pnl_net"]:+.2f}%')
    
    report = '\n'.join(lines)
    with open(REPORT_FILE, 'w') as f:
        f.write(report)
    
    return report

# ═══ Main Loop ═══
if __name__ == '__main__':
    print(f'🐋 حوت الموجات — بدأ {datetime.now(timezone.utc).isoformat()}', flush=True)
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
