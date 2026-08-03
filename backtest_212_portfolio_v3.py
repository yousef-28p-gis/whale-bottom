#!/usr/bin/env python3
"""Ichimoku 8h Ultra — V3: correct portfolio DD (equal allocation per coin)"""
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
        return r['c'].values, r['h'].values, r['l'].values, r['o'].values, r.index
    except:
        return None

def ichimoku_trades(c, h, l, o, idx):
    tenkan, kijun, senkou = 3, 9, 18
    tp, sl, cooldown = 5, 2.5, 2
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
                    pnl = tp - COMM * 100
                    trades.append((idx[i], pnl)); pos = 0; cool = cooldown
                elif l[i] <= ep * (1 - sl/100):
                    pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((idx[i], pnl)); pos = 0; cool = cooldown
            else:
                if l[i] <= ep * (1 - tp/100):
                    pnl = tp - COMM * 100
                    trades.append((idx[i], pnl)); pos = 0; cool = cooldown
                elif h[i] >= ep * (1 + sl/100):
                    pnl = max((1 - c[i]/ep)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    trades.append((idx[i], pnl)); pos = 0; cool = cooldown
        
        if not pos and cool == 0:
            if above and golden:
                pos = 1; ep = c[i]; side = 1
            elif below and death:
                pos = 1; ep = c[i]; side = -1
        if not pos and cool > 0:
            cool -= 1
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        trades.append((idx[-1], pnl))
    return trades

def portfolio_metrics(coin_trades, num_coins):
    """Calculate portfolio metrics with equal allocation per coin"""
    per_coin_cap = CAP / num_coins
    coin_eq = {}  # coin -> current equity
    
    # Build timeline: collect all trade timestamps
    all_events = []
    for sym, trades in coin_trades.items():
        coin_eq[sym] = per_coin_cap
        for ts, pnl in trades:
            all_events.append((ts, sym, pnl))
    all_events.sort()
    
    portfolio_eq = CAP
    eq_curve = [CAP]
    
    for ts, sym, pnl in all_events:
        old_val = coin_eq[sym]
        coin_eq[sym] *= (1 + pnl/100)
        portfolio_eq += (coin_eq[sym] - old_val)
        eq_curve.append(portfolio_eq)
    
    # DD
    s = pd.Series(eq_curve)
    peak = s.expanding().max()
    dd_pct = ((s - peak) / peak * 100).min()
    
    total_trades = len(all_events)
    wins = sum(1 for _, _, p in all_events if p > 0)
    wr = wins / total_trades * 100 if total_trades else 0
    
    # Per-coin PnL (absolute dollar)
    coin_pnl_dollar = {}
    for sym, trades in coin_trades.items():
        eq = per_coin_cap
        for _, pnl in trades:
            eq *= (1 + pnl/100)
        coin_pnl_dollar[sym] = eq - per_coin_cap
    
    return {
        'pnl': portfolio_eq - CAP,
        'eq': portfolio_eq,
        'dd': dd_pct,
        'trades': total_trades,
        'wins': wins,
        'wr': wr,
        'coin_pnl': coin_pnl_dollar,
    }

# Load tradeable coins
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

print("🎯 Ichimoku 8h Ultra | TP5/SL2.5 | توزيع متساوي\n")

for period_name in ['2023', 'PREV', 'CUR']:
    print(f"{'='*55}")
    print(f"📅 {period_name}")
    
    coin_trades = {}
    coin_n = {}  # number of trades per coin
    
    for sym in tradeable:
        data = load(sym, period_name)
        if data is None: continue
        c, h, l, o, ts = data
        resampled = resample_8h(c, h, l, o, ts)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) < 3: continue
        coin_trades[sym] = trades
        coin_n[sym] = len(trades)
    
    N = len(coin_trades)
    if N == 0:
        print("  ❌ لا عملات\n")
        continue
    
    # All coins
    m = portfolio_metrics(coin_trades, N)
    green = sum(1 for v in m['coin_pnl'].values() if v > 0)
    red = N - green
    
    print(f"\n📊 كل العملات ({N}):")
    print(f"   صفقات={m['trades']} | WR={m['wr']:.1f}% | 🟢{green} 🔴{red}")
    print(f"   💵 {m['pnl']:+,.0f}$ | 📉 سحب={m['dd']:.1f}%")
    
    # Profitable only
    prof = {s: t for s, t in coin_trades.items() if m['coin_pnl'][s] > 0}
    if prof:
        mp = portfolio_metrics(prof, N)  # keep same N for allocation fairness
        print(f"\n📊 🟢 مربحة فقط ({len(prof)}):")
        print(f"   صفقات={mp['trades']} | WR={mp['wr']:.1f}%")
        print(f"   💵 {mp['pnl']:+,.0f}$ | 📉 سحب={mp['dd']:.1f}%")
    
    # Top 75%
    sorted_coins = sorted(m['coin_pnl'].items(), key=lambda x: x[1], reverse=True)
    top75_count = max(int(N * 0.75), 10)
    top75_coins = set(c for c, _ in sorted_coins[:top75_count])
    excluded = [c for c, _ in sorted_coins[top75_count:]]
    
    top75_trades = {s: t for s, t in coin_trades.items() if s in top75_coins}
    mt = portfolio_metrics(top75_trades, N)
    
    print(f"\n📊 🥇 أفضل 75% ({top75_count}):")
    print(f"   صفقات={mt['trades']} | WR={mt['wr']:.1f}%")
    print(f"   💵 {mt['pnl']:+,.0f}$ | 📉 سحب={mt['dd']:.1f}%")
    
    excl_pnl = sum(m['coin_pnl'][s] for s in excluded)
    print(f"   🗑️ مستبعدين ({len(excluded)}): {excl_pnl:+,.0f}$")
    # Show excluded
    excl_str = '  '.join(excluded[:12])
    print(f"   {excl_str}{'...' if len(excluded)>12 else ''}")
    
    # Top 50%
    top50_count = max(int(N * 0.5), 5)
    top50_coins = set(c for c, _ in sorted_coins[:top50_count])
    top50_trades = {s: t for s, t in coin_trades.items() if s in top50_coins}
    m50 = portfolio_metrics(top50_trades, N)
    excl50_pnl = sum(m['coin_pnl'][s] for s in [c for c,_ in sorted_coins[top50_count:]])
    
    print(f"\n📊 🥇 أفضل 50% ({top50_count}):")
    print(f"   صفقات={m50['trades']} | WR={m50['wr']:.1f}%")
    print(f"   💵 {m50['pnl']:+,.0f}$ | 📉 سحب={m50['dd']:.1f}%")
    print(f"   🗑️ مستبعدين ({N-top50_count}): {excl50_pnl:+,.0f}$")

print(f"\n{'='*55}")
print("✅ تم")
