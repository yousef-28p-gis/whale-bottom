#!/usr/bin/env python3
"""Quick test: different SL values with TP=5"""
import json, os, numpy as np, pandas as pd

COMM = 0.002; CAP = 1000; MAX_SLIPPAGE = 1.5; COOLDOWN = 2

def load(sym, period):
    p = os.path.join(f'/data/trading28/data/whale_15m_{period}', f'{sym}.json')
    if not os.path.exists(p): return None
    with open(p) as f:
        j = json.load(f)
    return (np.array(j['c'],float), np.array(j['h'],float), np.array(j['l'],float),
            np.array(j['o'],float), j.get('ts',[]))

def resample_8h(c, h, l, o, ts):
    try:
        idx = pd.to_datetime(np.array(ts), unit='ms')
        df = pd.DataFrame({'o':o, 'h':h, 'l':l, 'c':c}, index=idx)
        r = df.resample('8h').agg({'o':'first','h':'max','l':'min','c':'last'}).dropna()
        return r['c'].values, r['h'].values, r['l'].values, r['o'].values, r.index
    except: return None

def ichimoku_trades(c, h, l, o, idx, sl):
    tenkan, kijun, senkou = 3, 9, 18
    tp, cooldown = 5, 2
    n = len(c)
    if n < senkou + 30: return []
    
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
        if i + shift < n:
            sa[i+shift] = sa_raw[i]; sb[i+shift] = sb_raw[i]
    
    trades = []
    pos = 0; ep = 0; cool = 0; side = 0
    
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top = max(sa[i], sb[i]); cloud_bot = min(sa[i], sb[i])
        above = c[i] > cloud_top; below = c[i] < cloud_bot
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        death = t_arr[i] < k_arr[i] and t_arr[i-1] >= k_arr[i-1]
        
        if pos:
            if side == 1:
                if h[i] >= ep * (1 + tp/100):
                    trades.append((idx[i], tp - COMM*100)); pos = 0; cool = cooldown
                elif l[i] <= ep * (1 - sl/100):
                    pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((idx[i], pnl)); pos = 0; cool = cooldown
            else:
                if l[i] <= ep * (1 - tp/100):
                    trades.append((idx[i], tp - COMM*100)); pos = 0; cool = cooldown
                elif h[i] >= ep * (1 + sl/100):
                    pnl = max((1 - c[i]/ep)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((idx[i], pnl)); pos = 0; cool = cooldown
        
        if not pos and cool == 0:
            if above and golden: pos = 1; ep = c[i]; side = 1
            elif below and death: pos = 1; ep = c[i]; side = -1
        if not pos and cool > 0: cool -= 1
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        trades.append((idx[-1], pnl))
    return trades

def calc_metrics(coin_trades, N):
    per_coin = CAP / N
    coin_eq = {}
    events = []
    for sym, trades in coin_trades.items():
        coin_eq[sym] = per_coin
        for ts, pnl in trades:
            events.append((ts, sym, pnl))
    events.sort()
    
    pf_eq = CAP
    eq_curve = [CAP]
    for _, sym, pnl in events:
        old = coin_eq[sym]
        coin_eq[sym] *= (1 + pnl/100)
        pf_eq += (coin_eq[sym] - old)
        eq_curve.append(pf_eq)
    
    s = pd.Series(eq_curve)
    peak = s.expanding().max()
    dd = ((s - peak) / peak * 100).min()
    total = len(events)
    wins = sum(1 for _, _, p in events if p > 0)
    
    return {
        'pnl': pf_eq - CAP, 'dd': dd, 'trades': total,
        'wr': wins/total*100 if total else 0
    }

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Test all periods with different SL
for period_name in ['PREV']:  # fastest, most coins
    print(f"\n📅 {period_name}")
    print(f"{'SL':>6s}  {'WR':>6s}  {'صفقات':>6s}  {'ربح$':>8s}  {'سحب':>6s}  {'R:R فعلي'}")
    print(f"{'─'*50}")
    
    # Pre-load all coin data
    coin_data = {}
    for sym in tradeable:
        data = load(sym, period_name.lower() if period_name != 'PREV' else 'prev')
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        coin_data[sym] = resampled
    
    for sl in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.5]:
        coin_trades = {}
        for sym, (c8, h8, l8, o8, idx) in coin_data.items():
            trades = ichimoku_trades(c8, h8, l8, o8, idx, sl)
            if len(trades) >= 3:
                coin_trades[sym] = trades
        
        if not coin_trades: continue
        N = len(coin_trades)
        m = calc_metrics(coin_trades, N)
        
        # Actual R:R (avg win / avg loss)
        all_pnls = [p for trades in coin_trades.values() for _, p in trades]
        wins_pnl = [p for p in all_pnls if p > 0]
        losses_pnl = [abs(p) for p in all_pnls if p < 0]
        avg_w = np.mean(wins_pnl) if wins_pnl else 0
        avg_l = np.mean(losses_pnl) if losses_pnl else 0
        rr = avg_w / avg_l if avg_l else 0
        
        expected_random_wr = sl / (5 + sl) * 100  # what WR would be random
        
        print(f"SL={sl:3.1f}%  WR={m['wr']:5.1f}%  {m['trades']:6d}  {m['pnl']:+8.0f}$  DD={m['dd']:5.1f}%  R:R={rr:.2f}  (عشوائي={expected_random_wr:.0f}%)")
