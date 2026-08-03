#!/usr/bin/env python3
"""Cloud Hunter — 3Y: RSI>50 + BTC crash protection filters"""
import json, os, numpy as np, pandas as pd

COMM = 0.002; MAX_SLIPPAGE = 1.5; COOLDOWN = 2

def load(sym, period):
    p = os.path.join(f'/data/trading28/data/whale_15m_{period}', f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f: j = json.load(f)
    return (np.array(j['c'],float), np.array(j['h'],float), np.array(j['l'],float),
            np.array(j['o'],float), j.get('ts',[]))

def resample_8h(c, h, l, o, ts):
    try:
        idx = pd.to_datetime(np.array(ts), unit='ms')
        df = pd.DataFrame({'o':o,'h':h,'l':l,'c':c}, index=idx)
        r = df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values, r['h'].values, r['l'].values, r['o'].values, r.index
    except: return None

def compute_rsi(c, period=14):
    n = len(c); rsi = np.full(n, np.nan)
    if n < period+1: return rsi
    delta = np.diff(c)
    gain = np.maximum(delta, 0); loss = np.abs(np.minimum(delta, 0))
    for i in range(period+1, n+1):
        avg_gain = np.mean(gain[i-period:i])
        avg_loss = np.mean(loss[i-period:i])
        rsi[i-1] = 100 - 100/(1+avg_gain/avg_loss) if avg_loss != 0 else 100
    return rsi

def ichimoku_trades(c, h, l, o, idx, btc_ok=None):
    tenkan, kijun, senkou = 3, 9, 18; tp, sl = 5, 2.5
    n = len(c)
    if n < senkou + 30: return [], 0
    h_t = pd.Series(h).rolling(tenkan).max().values
    l_t = pd.Series(l).rolling(tenkan).min().values
    t_arr = (h_t + l_t) / 2
    h_k = pd.Series(h).rolling(kijun).max().values
    l_k = pd.Series(l).rolling(kijun).min().values
    k_arr = (h_k + l_k) / 2
    h_s = pd.Series(h).rolling(senkou).max().values
    l_s = pd.Series(l).rolling(senkou).min().values
    sb_raw = (h_s + l_s) / 2; sa_raw = (t_arr + k_arr) / 2
    shift = kijun
    sa = np.full(n, np.nan); sb = np.full(n, np.nan)
    for i in range(max(shift, senkou), n - shift):
        if i + shift < n: sa[i+shift] = sa_raw[i]; sb[i+shift] = sb_raw[i]
    
    rsi = compute_rsi(c)
    
    # Match BTC index to our index
    btc_map = {}
    if btc_ok is not None:
        for j, dt in enumerate(idx):
            btc_map[dt] = btc_ok[j] if j < len(btc_ok) else True
    
    trades = []; signals = 0
    pos = 0; ep = 0; cool = 0; entry_idx = None
    
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top = max(sa[i], sb[i])
        above = c[i] > cloud_top
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        
        signal = above and golden and not np.isnan(rsi[i]) and rsi[i] > 50
        
        # BTC global filter
        if signal and btc_ok is not None:
            dt = idx[i]
            if dt in btc_map:
                signal = signal and btc_map[dt]
        
        if signal: signals += 1
        
        if pos:
            if h[i] >= ep * (1 + tp/100):
                trades.append((entry_idx, idx[i], tp - COMM*100)); pos = 0; cool = COOLDOWN
            elif l[i] <= ep * (1 - sl/100):
                pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = COOLDOWN
        
        if not pos and cool == 0 and signal:
            pos = 1; ep = c[i]; entry_idx = idx[i]
        
        if not pos and cool > 0: cool -= 1
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100
        trades.append((entry_idx, idx[-1], pnl))
    return trades, signals

def run_2positions(coin_trades):
    eq = 1000; eq_curve = [1000]; open_positions = {}
    timeline = []
    for sym, (trades, _) in coin_trades.items():
        for entry_t, exit_t, pnl in trades:
            timeline.append((entry_t, 'entry', sym, pnl))
            timeline.append((exit_t, 'exit', sym, pnl))
    timeline.sort()
    executed = 0; wins = 0
    for t, etype, sym, pnl in timeline:
        if etype == 'entry':
            if len(open_positions) < 2:
                open_positions[sym] = eq / 2
        elif etype == 'exit':
            if sym in open_positions:
                alloc = open_positions.pop(sym)
                new_val = alloc * (1 + pnl/100)
                eq += (new_val - alloc)
                eq_curve.append(eq)
                executed += 1
                if pnl > 0: wins += 1
    s = pd.Series(eq_curve); peak = s.expanding().max()
    dd = ((s - peak) / peak * 100).min()
    wr = wins/executed*100 if executed else 0
    return {'pnl': eq-1000, 'dd': dd, 'trades': executed, 'wr': wr, 'eq': eq}

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
all_coins = sorted(d['halal'] + d['halal2'])

# Pre-compute BTC state for each period
def get_btc_filter(period, filter_type):
    """Returns btc_ok array aligned to 8h index, or None"""
    btc_data = load('BTCUSDT', period)
    if btc_data is None: return None, None
    c, h, l, o, ts = btc_data
    rp = resample_8h(c, h, l, o, ts)
    if rp is None: return None, None
    c8, h8, l8, o8, idx = rp
    
    ok = np.ones(len(c8), bool)
    rsi = compute_rsi(c8)
    ema50 = pd.Series(c8).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(c8).ewm(span=200, adjust=False).mean().values
    
    for i in range(len(c8)):
        if filter_type == 'btc_ema200':
            if not np.isnan(ema200[i]):
                ok[i] = c8[i] > ema200[i]
        elif filter_type == 'btc_ema50':
            if not np.isnan(ema50[i]):
                ok[i] = c8[i] > ema50[i]
        elif filter_type == 'btc_rsi50':
            if not np.isnan(rsi[i]):
                ok[i] = rsi[i] > 50
        elif filter_type == 'btc_ema200_rsi50':
            ok[i] = (not np.isnan(ema200[i]) and c8[i] > ema200[i]) and (not np.isnan(rsi[i]) and rsi[i] > 50)
    
    return ok, idx

btc_filters = [
    (None, 'RSI>50 فقط'),
    ('btc_ema200', 'RSI>50 + BTC>EMA200'),
    ('btc_ema50', 'RSI>50 + BTC>EMA50'),
    ('btc_rsi50', 'RSI>50 + BTC RSI>50'),
    ('btc_ema200_rsi50', 'RSI>50 + BTC>EMA200+RSI>50'),
]

print(f"☁️ صياد السحابة | 3 سنوات | كل العملات | صفقتين × $500\n")

for btc_flt, label in btc_filters:
    print(f"📋 {label}")
    print(f"{'الفترة':>6s} | {'عملات':>4s} | {'إشارات':>5s} | {'منفذ':>4s} | {'WR':>5s} | {'ربح$':>8s} | {'سحب':>6s} | {'نهائي$':>8s}")
    print("─" * 68)
    grand = 0; grand_sig = 0; grand_tr = 0
    for pname, pdir in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
        # Get BTC state for this period+filter
        btc_ok, btc_idx = get_btc_filter(pdir, btc_flt) if btc_flt else (None, None)
        
        coin_trades = {}
        for sym in all_coins:
            data = load(sym, pdir)
            if data is None: continue
            resampled = resample_8h(*data)
            if resampled is None: continue
            c8, h8, l8, o8, idx = resampled
            trades, signals = ichimoku_trades(c8, h8, l8, o8, idx, btc_ok=btc_ok)
            if len(trades) >= 3:
                coin_trades[sym] = (trades, signals)
        
        N = len(coin_trades)
        total_sig = sum(v[1] for v in coin_trades.values())
        m = run_2positions(coin_trades)
        grand_sig += total_sig; grand_tr += m['trades']
        
        print(f"{pname:>6s} | {N:4d} | {total_sig:5d} | {m['trades']:4d} | {m['wr']:4.1f}% | ${m['pnl']:+7,.0f} | {m['dd']:5.1f}% | ${m['eq']:7,.0f}")
        grand += m['pnl']
    
    print(f"{'─'*68}")
    print(f"💰 الإجمالي | {'':>4s} | {grand_sig:5d} | {grand_tr:4d} | {'':>5s} | ${grand:+7,.0f}\n\n")
