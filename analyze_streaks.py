#!/usr/bin/env python3
"""Analyze worst coin: losing streaks, and test trend filter"""
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

def ichimoku_trades_detail(c, h, l, o, idx, use_ema_filter=False):
    tenkan, kijun, senkou = 3, 9, 18
    tp, sl, cooldown = 5, 2.5, 2
    n = len(c)
    if n < senkou + 30: return [], []
    
    # EMA filter
    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().values
    
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
    signals = []  # (idx, type, reason)
    pos = 0; ep = 0; cool = 0; side = 0
    
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        if np.isnan(ema200[i]): continue
        
        cloud_top = max(sa[i], sb[i]); cloud_bot = min(sa[i], sb[i])
        above = c[i] > cloud_top; below_cloud = c[i] < cloud_bot
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        death = t_arr[i] < k_arr[i] and t_arr[i-1] >= k_arr[i-1]
        
        # EMA filter
        trend_ok = True
        if use_ema_filter:
            trend_ok = c[i] > ema200[i]  # long only above EMA200
        
        if pos:
            if side == 1:
                if h[i] >= ep * (1 + tp/100):
                    trades.append((idx[i], tp - COMM*100, 'TP')); pos = 0; cool = cooldown
                elif l[i] <= ep * (1 - sl/100):
                    pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((idx[i], pnl, 'SL')); pos = 0; cool = cooldown
            else:
                if l[i] <= ep * (1 - tp/100):
                    trades.append((idx[i], tp - COMM*100, 'TP')); pos = 0; cool = cooldown
                elif h[i] >= ep * (1 + sl/100):
                    pnl = max((1 - c[i]/ep)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((idx[i], pnl, 'SL')); pos = 0; cool = cooldown
        
        if not pos and cool == 0 and trend_ok:
            if above and golden:
                pos = 1; ep = c[i]; side = 1
                signals.append((idx[i], 'LONG'))
            elif below_cloud and death:
                if not use_ema_filter:
                    pos = 1; ep = c[i]; side = -1
                    signals.append((idx[i], 'SHORT'))
        
        if not pos and cool > 0:
            cool -= 1
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        trades.append((idx[-1], pnl, 'CLOSE'))
    return trades, signals

# Find worst coin in PREV
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# PREV analysis
coin_pnls = {}
for sym in tradeable:
    data = load(sym, 'prev')
    if data is None: continue
    resampled = resample_8h(*data)
    if resampled is None: continue
    c8, h8, l8, o8, idx = resampled
    trades, _ = ichimoku_trades_detail(c8, h8, l8, o8, idx)
    if len(trades) < 3: continue
    pnl_sum = sum(p for _, p, _ in trades)
    coin_pnls[sym] = pnl_sum

# Worst 3 coins
worst = sorted(coin_pnls.items(), key=lambda x: x[1])[:3]
print("🔴 أسوأ 3 عملات في PREV:")
for sym, pnl in worst:
    data = load(sym, 'prev')
    c8, h8, l8, o8, idx = resample_8h(*data)
    trades, _ = ichimoku_trades_detail(c8, h8, l8, o8, idx)
    
    # Analyze losing streaks
    pnls_only = [p for _, p, reason in trades]
    wins = sum(1 for p in pnls_only if p > 0)
    wr = wins/len(pnls_only)*100
    
    # Find longest losing streak
    max_losing = 0
    current_losing = 0
    losing_streaks = []
    for i, (ts, p, reason) in enumerate(trades):
        if p < 0:
            current_losing += 1
        else:
            if current_losing > 0:
                losing_streaks.append(current_losing)
                current_losing = 0
    if current_losing > 0:
        losing_streaks.append(current_losing)
    
    max_losing = max(losing_streaks) if losing_streaks else 0
    
    # Time between consecutive losses
    loss_timestamps = [ts for ts, p, _ in trades if p < 0]
    gaps = []
    for i in range(1, len(loss_timestamps)):
        gap_hours = (loss_timestamps[i] - loss_timestamps[i-1]).total_seconds() / 3600
        gaps.append(gap_hours)
    
    avg_gap = np.mean(gaps) if gaps else 0
    
    print(f"\n  {sym}: {len(trades)} صفقة | WR={wr:.0f}% | ${pnl:+,.0f}")
    print(f"  أطول سلسلة خسائر: {max_losing} متتالية")
    print(f"  متوسط الوقت بين الخسائر: {avg_gap:.0f} ساعة")
    print(f"  توزيع الخسائر المتتالية: {losing_streaks[:10]}{'...' if len(losing_streaks)>10 else ''}")

# Now test with trend filter
print(f"\n{'='*55}")
print(f"🔍 اختبار فلتر EMA200 — PREV فقط")
print(f"{'='*55}")

def calc_portfolio(coin_trades, N):
    per_coin = CAP / N
    coin_eq = {}
    events = []
    for sym, trades in coin_trades.items():
        coin_eq[sym] = per_coin
        for ts, pnl, reason in trades:
            events.append((ts, sym, pnl))
    events.sort()
    pf_eq = CAP; eq_curve = [CAP]
    for _, sym, pnl in events:
        old = coin_eq[sym]; coin_eq[sym] *= (1 + pnl/100)
        pf_eq += (coin_eq[sym] - old); eq_curve.append(pf_eq)
    s = pd.Series(eq_curve)
    peak = s.expanding().max()
    dd = ((s - peak) / peak * 100).min()
    total = len(events)
    wins = sum(1 for _, _, p in events if p > 0)
    return {'pnl': pf_eq-CAP, 'dd': dd, 'trades': total, 'wr': wins/total*100 if total else 0}

# Without filter
print("\n❌ بدون فلتر:")
coin_trades_nf = {}
for sym in tradeable[:50]:  # test on 50 coins for speed
    data = load(sym, 'prev')
    if data is None: continue
    resampled = resample_8h(*data)
    if resampled is None: continue
    c8, h8, l8, o8, idx = resampled
    trades, _ = ichimoku_trades_detail(c8, h8, l8, o8, idx, False)
    if len(trades) >= 3: coin_trades_nf[sym] = trades

if coin_trades_nf:
    N = len(coin_trades_nf)
    m = calc_portfolio(coin_trades_nf, N)
    print(f"  {N} عملة | {m['trades']} صفقة | WR={m['wr']:.1f}% | ${m['pnl']:+,.0f} | سحب={m['dd']:.1f}%")

# With EMA200 filter
print("\n✅ مع فلتر EMA200:")
coin_trades_f = {}
for sym in tradeable[:50]:
    data = load(sym, 'prev')
    if data is None: continue
    resampled = resample_8h(*data)
    if resampled is None: continue
    c8, h8, l8, o8, idx = resampled
    trades, _ = ichimoku_trades_detail(c8, h8, l8, o8, idx, True)
    if len(trades) >= 3: coin_trades_f[sym] = trades

if coin_trades_f:
    N = len(coin_trades_f)
    m = calc_portfolio(coin_trades_f, N)
    print(f"  {N} عملة | {m['trades']} صفقة | WR={m['wr']:.1f}% | ${m['pnl']:+,.0f} | سحب={m['dd']:.1f}%")
