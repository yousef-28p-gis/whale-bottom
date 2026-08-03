#!/usr/bin/env python3
"""Ichimoku 8h Ultra — V2: portfolio-level DD, filter losing coins"""
import json, os, numpy as np, pandas as pd

COMM = 0.002; CAP = 1000; MAX_SLIPPAGE = 1.5; COOLDOWN = 2

DATA_DIRS = {
    '2023': '/data/trading28/data/whale_15m_2023',
    'PREV': '/data/trading28/data/whale_15m_prev',
    'CUR':  '/data/trading28/data/whale_15m_1y',
}

def load(sym, period):
    p = os.path.join(DATA_DIRS[period], f'{sym}.json')
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
        return r['c'].values, r['h'].values, r['l'].values, r['o'].values
    except:
        return None

def ichimoku_trades(c, h, l, o, tenkan=3, kijun=9, senkou=18, tp=5, sl=2.5, cooldown=2):
    """Return list of (timestamp_idx, pnl) tuples"""
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
    sb_raw = (h_s + l_s) / 2
    sa_raw = (t_arr + k_arr) / 2
    
    shift = kijun
    sa = np.full(n, np.nan); sb = np.full(n, np.nan)
    for i in range(max(shift, senkou), n - shift):
        if i + shift < n:
            sa[i+shift] = sa_raw[i]
            sb[i+shift] = sb_raw[i]
    
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
                    pnl = tp - COMM * 100
                    trades.append((i, pnl)); pos = 0; cool = cooldown
                elif l[i] <= ep * (1 - sl/100):
                    pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((i, pnl)); pos = 0; cool = cooldown
            else:
                if l[i] <= ep * (1 - tp/100):
                    pnl = tp - COMM * 100
                    trades.append((i, pnl)); pos = 0; cool = cooldown
                elif h[i] >= ep * (1 + sl/100):
                    pnl = max((1 - c[i]/ep)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((i, pnl)); pos = 0; cool = cooldown
        
        if not pos and cool == 0:
            if above and golden:
                pos = 1; ep = c[i]; side = 1
            elif below and death:
                pos = 1; ep = c[i]; side = -1
        
        if not pos and cool > 0:
            cool -= 1
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        trades.append((n-1, pnl))
    
    return trades

def portfolio_dd(equity_curve):
    """Calculate max drawdown from equity curve"""
    s = pd.Series(equity_curve)
    peak = s.expanding().max()
    dd = (s - peak) / peak * 100
    return dd.min()

# Load coins
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

print("🎯 Ichimoku 8h Ultra | TP5/SL2.5 | Portfolio-level DD\n")

for period_name in ['2023', 'PREV', 'CUR']:
    print(f"{'='*60}")
    print(f"📅 {period_name}")
    
    coin_trades = {}
    coin_pnl = {}
    
    for sym in tradeable:
        data = load(sym, period_name)
        if data is None: continue
        c, h, l, o, ts = data
        resampled = resample_8h(c, h, l, o, ts)
        if resampled is None: continue
        c8, h8, l8, o8 = resampled
        trades = ichimoku_trades(c8, h8, l8, o8)
        if len(trades) < 3: continue
        
        pnl_sum = sum(p for _, p in trades)
        coin_trades[sym] = trades
        coin_pnl[sym] = pnl_sum
    
    # === ALL COINS ===
    all_trades = []
    for sym, trades in coin_trades.items():
        all_trades.extend([(sym, idx, pnl) for idx, pnl in trades])
    # Sort by time index
    all_trades.sort(key=lambda x: x[1])
    
    eq = CAP
    eq_curve = [CAP]
    total_trades = len(all_trades)
    wins = sum(1 for _, _, p in all_trades if p > 0)
    
    for _, _, pnl in all_trades:
        eq *= (1 + pnl/100)
        eq_curve.append(eq)
    
    all_dd = portfolio_dd(eq_curve)
    wr = wins/total_trades*100 if total_trades else 0
    
    green_coins = sum(1 for p in coin_pnl.values() if p > 0)
    red_coins = sum(1 for p in coin_pnl.values() if p <= 0)
    
    print(f"\n📊 كل العملات ({len(coin_pnl)}):")
    print(f"   صفقات={total_trades} | WR={wr:.1f}% | 🟢{green_coins} 🔴{red_coins}")
    print(f"   💵 ${eq-CAP:+,.0f} | 📉 سحب المحفظة={all_dd:.1f}%")
    
    # === PROFITABLE ONLY ===
    profitable = {s: t for s, t in coin_trades.items() if coin_pnl[s] > 0}
    
    all_trades_prof = []
    for sym, trades in profitable.items():
        all_trades_prof.extend([(sym, idx, pnl) for idx, pnl in trades])
    all_trades_prof.sort(key=lambda x: x[1])
    
    eq = CAP
    eq_curve = [CAP]
    for _, _, pnl in all_trades_prof:
        eq *= (1 + pnl/100)
        eq_curve.append(eq)
    
    prof_dd = portfolio_dd(eq_curve)
    prof_trades = len(all_trades_prof)
    prof_wins = sum(1 for _, _, p in all_trades_prof if p > 0)
    prof_wr = prof_wins/prof_trades*100 if prof_trades else 0
    prof_pnl = eq - CAP
    
    # === TOP 75% by PnL ===
    sorted_coins = sorted(coin_pnl.items(), key=lambda x: x[1], reverse=True)
    top75_count = max(int(len(sorted_coins) * 0.75), 10)
    top75 = dict(sorted_coins[:top75_count])
    
    all_trades_75 = []
    for sym, trades in coin_trades.items():
        if sym in top75:
            all_trades_75.extend([(sym, idx, pnl) for idx, pnl in trades])
    all_trades_75.sort(key=lambda x: x[1])
    
    eq = CAP
    eq_curve = [CAP]
    for _, _, pnl in all_trades_75:
        eq *= (1 + pnl/100)
        eq_curve.append(eq)
    
    dd75 = portfolio_dd(eq_curve)
    t75 = len(all_trades_75)
    w75 = sum(1 for _, _, p in all_trades_75 if p > 0)
    
    print(f"\n📊 🟢 مربحة فقط ({len(profitable)}):")
    print(f"   صفقات={prof_trades} | WR={prof_wr:.1f}%")
    print(f"   💵 ${prof_pnl:+,.0f} | 📉 سحب={prof_dd:.1f}%")
    
    print(f"\n📊 🥇 أفضل 75% ({top75_count}):")
    print(f"   صفقات={t75} | WR={w75/t75*100:.1f}%" if t75 else "   لا صفقات")
    print(f"   💵 ${eq-CAP:+,.0f} | 📉 سحب={dd75:.1f}%")
    
    # Excluded coins
    excluded = [s for s in coin_pnl if s not in top75]
    if excluded:
        excl_pnl = sum(coin_pnl[s] for s in excluded)
        print(f"   🗑️ مستبعدين ({len(excluded)}): ${excl_pnl:+,.0f}  {', '.join(excluded[:10])}{'...' if len(excluded)>10 else ''}")

print(f"\n{'='*60}")
print("✅ تم")
