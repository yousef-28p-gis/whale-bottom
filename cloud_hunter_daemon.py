#!/usr/bin/env python3
"""Cloud Hunter Live Daemon — Ichimoku 8h + RSI>50 + 4h>EMA50"""
import ccxt, json, os, time, sys, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# === CONFIG ===
SCAN_INTERVAL = 8 * 3600  # every 8 hours
COMM = 0.2
MAX_SLIPPAGE = 1.5
TP = 5.0; SL = 2.5
MAX_POS = 2; POS_PCT = 0.50
CAPITAL = 1000
CACHE_DIR = '/data/trading28/cache/cloud_hunter/'
STATE_FILE = '/data/trading28/cloud_hunter_state.json'
REPORT_FILE = '/data/trading28/cloud_hunter_report.txt'
COINS_FILE = '/data/trading28/config/shariah_coins.json'
MIN_8H_CANDLES = 300

# === Exchange ===
exchange = ccxt.binance({'timeout': 15000, 'options': {'defaultType': 'spot'}})

def load_coins():
    with open(COINS_FILE) as f: d = json.load(f)
    return sorted(d['halal'] + d['halal2'])

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'active': [], 'closed': [], 'last_scan': 0}

def save_state(s):
    with open(STATE_FILE, 'w') as f: json.dump(s, f, default=str, indent=2)

def fetch_ohlcv(sym, limit=500):
    cache_path = Path(CACHE_DIR) / f'{sym}.json'
    cached = []
    if cache_path.exists():
        with open(cache_path) as f: cached = json.load(f)
    since = cached[-1]['ts'] + 1 if cached else None
    try:
        if since:
            raw = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', since=since, limit=limit)
        else:
            raw = exchange.fetch_ohlcv(f'{sym}/USDT', '15m', limit=limit)
    except:
        return cached
    for r in raw:
        cached.append({'ts': r[0], 'o': r[1], 'h': r[2], 'l': r[3], 'c': r[4]})
    # Deduplicate
    seen = set(); dedup = []
    for c in cached:
        if c['ts'] not in seen: seen.add(c['ts']); dedup.append(c)
    dedup.sort(key=lambda x: x['ts'])
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f: json.dump(dedup[-limit:], f)
    return dedup[-limit:]

def resample_8h(data):
    if len(data) < 100: return None
    df = pd.DataFrame(data)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    r = df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
    return r['c'].values, r['h'].values, r['l'].values, r['o'].values, r.index

def compute_rsi(c, p=14):
    n = len(c); r = np.full(n, np.nan)
    if n < p+1: return r
    d = np.diff(c); g = np.maximum(d, 0); l = np.abs(np.minimum(d, 0))
    for i in range(p+1, n+1):
        ag = np.mean(g[i-p:i]); al = np.mean(l[i-p:i])
        r[i-1] = 100 - 100/(1+ag/al) if al != 0 else 100
    return r

def check_signal(c, h, l, o):
    tk, kj, sk = 3, 9, 18; n = len(c)
    if n < sk + 30: return False, c[-1], None
    ht = pd.Series(h).rolling(tk).max().values; lt = pd.Series(l).rolling(tk).min().values
    ta = (ht + lt) / 2
    hk = pd.Series(h).rolling(kj).max().values; lk = pd.Series(l).rolling(kj).min().values
    ka = (hk + lk) / 2
    hs = pd.Series(h).rolling(sk).max().values; ls = pd.Series(l).rolling(sk).min().values
    sb = (hs + ls) / 2; sa = (ta + ka) / 2; sh = kj
    saf = np.full(n, np.nan); sbf = np.full(n, np.nan)
    for i in range(max(sh, sk), n - sh):
        if i + sh < n: saf[i+sh] = sa[i]; sbf[i+sh] = sb[i]
    ri = compute_rsi(c)
    ema4h = pd.Series(c).ewm(span=25, adjust=False).mean().values
    i = n - 1  # latest candle
    if np.isnan(saf[i]) or np.isnan(sbf[i]): return False, c[-1], None
    cloud_top = max(saf[i], sbf[i])
    above = c[i] > cloud_top
    golden = ta[i] > ka[i] and ta[i-1] <= ka[i-1]
    rsi_ok = not np.isnan(ri[i]) and ri[i] > 50
    ema_ok = c[i] > ema4h[i]
    signal = above and golden and rsi_ok and ema_ok
    return signal, c[i], ri[i]

def get_live_price(sym):
    try:
        t = exchange.fetch_ticker(f'{sym}/USDT')
        return t['last']
    except:
        return None

def scan():
    state = load_state()
    coins = load_coins()
    active_syms = {p['sym'] for p in state['active']}
    used_entries = set()
    for p in state['active'] + state['closed']:
        used_entries.add((p['sym'], p.get('entry_bar_ts', 0)))
    
    report_lines = [f"☁️ صياد السحابة | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"]
    report_lines.append(f"⚙️ 8h | RSI>50 | 4h>EMA50 | TP={TP}% SL={SL}% | {len(coins)} عملة")
    report_lines.append("")
    
    # Check active positions for TP/SL
    for pos in list(state['active']):
        sym = pos['sym']
        price = get_live_price(sym)
        if price is None: continue
        entry = pos['entry']
        
        if price >= entry * (1 + TP/100):
            pnl = TP - COMM
            report_lines.append(f"🟢 {sym}: TP +{TP}% | خروج=${price:.6f} | ربح=${pnl:.2f}%")
            state['active'].remove(pos)
            pos['pnl'] = pnl; pos['exit_price'] = price; pos['exit_type'] = 'TP'
            state['closed'].append(pos)
        elif price <= entry * (1 - SL/100):
            pnl = max((price/entry - 1)*100 - COMM, -SL*MAX_SLIPPAGE - COMM)
            report_lines.append(f"🔴 {sym}: SL -{SL}% | خروج=${price:.6f} | خسارة=${pnl:.2f}%")
            state['active'].remove(pos)
            pos['pnl'] = pnl; pos['exit_price'] = price; pos['exit_type'] = 'SL'
            state['closed'].append(pos)
    
    # Scan for new entries
    signals_found = 0
    for sym in coins:
        if sym in active_syms: continue
        if len(state['active']) >= MAX_POS: break
        
        data = fetch_ohlcv(sym)
        if len(data) < 400: continue
        
        rp = resample_8h(data)
        if rp is None: continue
        c8, h8, l8, o8, idx = rp
        if len(c8) < MIN_8H_CANDLES: continue
        
        signal, entry_price, rsi_val = check_signal(c8, h8, l8, o8)
        if not signal: continue
        
        # Entry bar timestamp for dedup
        entry_bar_ts = int(idx[-1].timestamp() * 1000)
        if (sym, entry_bar_ts) in used_entries: continue
        
        signals_found += 1
        price = get_live_price(sym) or entry_price
        sl_price = price * (1 - SL/100)
        tp_price = price * (1 + TP/100)
        
        pos = {'sym': sym, 'entry': price, 'sl': sl_price, 'tp': tp_price,
               'entry_bar_ts': entry_bar_ts, 'entry_time': str(idx[-1]),
               'rsi': round(rsi_val, 1) if rsi_val else None}
        state['active'].append(pos)
        active_syms.add(sym)
        used_entries.add((sym, entry_bar_ts))
        
        report_lines.append(f"🆕 {sym}: دخول=${price:.6f} | SL=${sl_price:.6f} | TP=${tp_price:.6f} | RSI={rsi_val:.1f}" if rsi_val else f"🆕 {sym}: دخول=${price:.6f}")
    
    # Summary
    state['last_scan'] = int(time.time() * 1000)
    
    total_pnl = sum(p.get('pnl', 0) for p in state['closed'])
    wins = sum(1 for p in state['closed'] if p.get('pnl', 0) > 0)
    losses = len(state['closed']) - wins
    
    report_lines.append("")
    report_lines.append(f"📊 نشط: {len(state['active'])}/{MAX_POS} | مغلق: {len(state['closed'])}")
    report_lines.append(f"🟢 ربح: {wins} | 🔴 خسارة: {losses} | 💵 صافي: ${total_pnl:+.2f}")
    if signals_found == 0 and len(state['active']) == 0:
        report_lines.append("😴 لا إشارات جديدة")
    
    report = '\n'.join(report_lines)
    with open(REPORT_FILE, 'w') as f: f.write(report)
    save_state(state)
    return report

def main():
    print("☁️ Cloud Hunter Daemon started", flush=True)
    while True:
        try:
            report = scan()
            print(report, flush=True)
            time.sleep(SCAN_INTERVAL)
        except KeyboardInterrupt:
            print("Stopped", flush=True)
            break
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            time.sleep(60)

if __name__ == '__main__':
    main()
