#!/usr/bin/env python3
"""Top 60 coins: 2 positions $500 with trend filters"""
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

def ichimoku_trades(c, h, l, o, idx, trend_filter=None):
    """trend_filter: None, 'ema200', 'ema50', '4h_trend'"""
    tenkan, kijun, senkou = 3, 9, 18; tp, sl = 5, 2.5
    n = len(c)
    if n < senkou + 30: return []
    
    # Compute trend indicators
    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().values
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    
    # 4h trend check: resample internal 8h bars to approximate 4h
    ema50_4h_approx = pd.Series(c).ewm(span=100, adjust=False).mean().values  # 8h bars, 100 ≈ 50 on 4h
    
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
    
    trades = []
    pos = 0; ep = 0; cool = 0; side = 0; entry_idx = None
    
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top = max(sa[i], sb[i]); cloud_bot = min(sa[i], sb[i])
        above = c[i] > cloud_top; below_cloud = c[i] < cloud_bot
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        death = t_arr[i] < k_arr[i] and t_arr[i-1] >= k_arr[i-1]
        
        # Trend filter check
        long_ok = True; short_ok = True
        if trend_filter == 'ema200':
            long_ok = c[i] > ema200[i]
            short_ok = c[i] < ema200[i]
        elif trend_filter == 'ema50':
            long_ok = c[i] > ema50[i]
            short_ok = c[i] < ema50[i]
        elif trend_filter == '4h_trend':
            long_ok = ema50[i] > ema200[i] if i >= 200 else True
            short_ok = ema50[i] < ema200[i] if i >= 200 else True
        
        if pos:
            if side == 1:
                if h[i] >= ep * (1 + tp/100):
                    trades.append((entry_idx, idx[i], tp - COMM*100)); pos = 0; cool = COOLDOWN
                elif l[i] <= ep * (1 - sl/100):
                    pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = COOLDOWN
            else:
                if l[i] <= ep * (1 - tp/100):
                    trades.append((entry_idx, idx[i], tp - COMM*100)); pos = 0; cool = COOLDOWN
                elif h[i] >= ep * (1 + sl/100):
                    pnl = max((1 - c[i]/ep)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = COOLDOWN
        
        if not pos and cool == 0:
            if above and golden and long_ok:
                pos = 1; ep = c[i]; side = 1; entry_idx = idx[i]
            elif below_cloud and death and short_ok:
                pos = 1; ep = c[i]; side = -1; entry_idx = idx[i]
        
        if not pos and cool > 0: cool -= 1
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        trades.append((entry_idx, idx[-1], pnl))
    return trades

def run_2positions(coin_trades):
    """Max 2 concurrent, $500 each"""
    eq = 1000; eq_curve = [1000]
    open_positions = {}  # sym -> alloc
    
    timeline = []
    for sym, trades in coin_trades.items():
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

# Load
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Rank top 60
coin_pnls = {}
for pname, pdir in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) >= 3:
            coin_pnls[sym] = coin_pnls.get(sym, 0) + sum(p for _, _, p in trades)

ranked = sorted(coin_pnls.items(), key=lambda x: x[1], reverse=True)
top60 = set(c for c, _ in ranked[:60])

print(f"🏆 أقوى 60 عملة | صفقتين × $500\n")

filters = [None, 'ema200', 'ema50', '4h_trend']
filter_names = {None: 'بدون فلتر', 'ema200': 'EMA200 8h', 'ema50': 'EMA50 8h', '4h_trend': 'EMA50>EMA200'}

for pname, pdir in [('PREV','prev')]:  # Test PREV only for speed
    print(f"{'='*55}")
    print(f"📅 {pname}")
    
    # Build coin trades for each filter
    for ft in filters:
        coin_trades = {}
        for sym in top60:
            data = load(sym, pdir)
            if data is None: continue
            resampled = resample_8h(*data)
            if resampled is None: continue
            c8, h8, l8, o8, idx = resampled
            trades = ichimoku_trades(c8, h8, l8, o8, idx, trend_filter=ft)
            if len(trades) >= 3:
                coin_trades[sym] = trades
        
        if not coin_trades: continue
        total_avail = sum(len(v) for v in coin_trades.values())
        m = run_2positions(coin_trades)
        
        print(f"\n📊 {filter_names[ft]}: {len(coin_trades)} عملة | {total_avail} إشارة")
        print(f"   منفذ: {m['trades']} صفقة | WR={m['wr']:.1f}% | ${m['pnl']:+,.0f} | سحب={m['dd']:.1f}% | نهائي=${m['eq']:,.0f}")
