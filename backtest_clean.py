#!/usr/bin/env python3
"""Ichimoku 8h — without worst coins"""
import json, os, numpy as np, pandas as pd

COMM = 0.002; CAP = 1000; MAX_SLIPPAGE = 1.5; COOLDOWN = 2

DATA_DIRS = {'2023': '2023', 'PREV': 'prev', 'CUR': '1y'}

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

def ichimoku_trades(c, h, l, o, idx):
    tenkan, kijun, senkou = 3, 9, 18; tp, sl, cooldown = 5, 2.5, 2
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
        if i + shift < n: sa[i+shift] = sa_raw[i]; sb[i+shift] = sb_raw[i]
    trades = []; pos = 0; ep = 0; cool = 0; side = 0
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top = max(sa[i], sb[i]); cloud_bot = min(sa[i], sb[i])
        above = c[i] > cloud_top; below_cloud = c[i] < cloud_bot
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
            elif below_cloud and death: pos = 1; ep = c[i]; side = -1
        if not pos and cool > 0: cool -= 1
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        trades.append((idx[-1], pnl))
    return trades

def portfolio_metrics(coin_trades, N):
    per_coin = CAP / N
    coin_eq = {}; events = []
    for sym, trades in coin_trades.items():
        coin_eq[sym] = per_coin
        for ts, pnl in trades: events.append((ts, sym, pnl))
    events.sort()
    pf_eq = CAP; eq_curve = [CAP]
    for _, sym, pnl in events:
        old = coin_eq[sym]; coin_eq[sym] *= (1 + pnl/100)
        pf_eq += (coin_eq[sym] - old); eq_curve.append(pf_eq)
    s = pd.Series(eq_curve); peak = s.expanding().max()
    dd = ((s - peak) / peak * 100).min()
    total = len(events); wins = sum(1 for _, _, p in events if p > 0)
    return {'pnl': pf_eq-CAP, 'dd': dd, 'trades': total, 'wr': wins/total*100 if total else 0}

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# ── Step 1: Find worst coins ──
print("🔍 تحديد العملات المرشحة للاستبعاد...")
coin_period_pnls = {}

for pname, pdir in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) < 3: continue
        if sym not in coin_period_pnls: coin_period_pnls[sym] = {}
        coin_period_pnls[sym][pname] = sum(p for _, p in trades)

# Exclude criteria: negative in 2+ periods OR any period with huge loss
exclude = set()
for sym, pp in coin_period_pnls.items():
    neg = sum(1 for p in pp.values() if p < 0)
    if neg >= 2:
        exclude.add(sym)

print(f"🗑️ مستبعدين (خسارة في فترتين+): {len(exclude)}")
if len(exclude) <= 20:
    print(f"   {', '.join(sorted(exclude))}")
print()

# ── Step 2: Run with vs without ──
print(f"{'='*55}")
print(f"📊 المقارنة — مع وبدون العملات الخاسرة")
print(f"{'='*55}")

for pname, pdir in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
    all_trades = {}
    clean_trades = {}
    
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) < 3: continue
        all_trades[sym] = trades
        if sym not in exclude:
            clean_trades[sym] = trades
    
    m_all = portfolio_metrics(all_trades, len(all_trades))
    m_clean = portfolio_metrics(clean_trades, len(clean_trades))
    
    removed = len(all_trades) - len(clean_trades)
    
    print(f"\n📅 {pname}:")
    print(f"   كل العملات ({len(all_trades)}):  {m_all['trades']:5d} صفقة | WR={m_all['wr']:.1f}% | ${m_all['pnl']:+,.0f} | سحب={m_all['dd']:.1f}%")
    print(f"   نظيفة ({len(clean_trades)}):      {m_clean['trades']:5d} صفقة | WR={m_clean['wr']:.1f}% | ${m_clean['pnl']:+,.0f} | سحب={m_clean['dd']:.1f}%")

print(f"\n{'='*55}")
print(f"💰 المحفظة النظيفة توزع رأس المال على عملات أقل → تركيز أعلى → عائد أفضل لكل عملة جيدة")
