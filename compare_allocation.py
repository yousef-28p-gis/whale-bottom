#!/usr/bin/env python3
"""Compare: equal distribution vs max 2 concurrent positions"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict

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
    
    # Return list of (entry_time, exit_time, pnl) — need both for position tracking
    trades = []
    pos = 0; ep = 0; cool = 0; side = 0; entry_idx = None
    
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top = max(sa[i], sb[i]); cloud_bot = min(sa[i], sb[i])
        above = c[i] > cloud_top; below_cloud = c[i] < cloud_bot
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        death = t_arr[i] < k_arr[i] and t_arr[i-1] >= k_arr[i-1]
        
        if pos:
            exited = False
            if side == 1:
                if h[i] >= ep * (1 + 5/100):
                    trades.append((entry_idx, idx[i], 5 - COMM*100)); pos = 0; cool = COOLDOWN; exited = True
                elif l[i] <= ep * (1 - 2.5/100):
                    pnl = max((c[i]/ep - 1)*100 - COMM*100, -2.5*MAX_SLIPPAGE - COMM*100)
                    trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = COOLDOWN; exited = True
            else:
                if l[i] <= ep * (1 - 5/100):
                    trades.append((entry_idx, idx[i], 5 - COMM*100)); pos = 0; cool = COOLDOWN; exited = True
                elif h[i] >= ep * (1 + 2.5/100):
                    pnl = max((1 - c[i]/ep)*100 - COMM*100, -2.5*MAX_SLIPPAGE - COMM*100)
                    trades.append((entry_idx, idx[i], pnl)); pos = 0; cool = COOLDOWN; exited = True
        
        if not pos and cool == 0:
            if above and golden:
                pos = 1; ep = c[i]; side = 1; entry_idx = idx[i]
            elif below_cloud and death:
                pos = 1; ep = c[i]; side = -1; entry_idx = idx[i]
        
        if not pos and cool > 0:
            cool -= 1
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        trades.append((entry_idx, idx[-1], pnl))
    return trades

def run_equal_distribution(all_coin_trades, N_coins):
    """Equal allocation: CAP/N per coin, all can trade simultaneously"""
    per_coin = CAP / N_coins
    coin_eq = {}
    events = []
    for sym, trades in all_coin_trades.items():
        coin_eq[sym] = per_coin
        for entry, exit_t, pnl in trades:
            events.append((exit_t, sym, pnl))
    events.sort()
    
    pf_eq = CAP; eq_curve = [CAP]
    for _, sym, pnl in events:
        old = coin_eq[sym]; coin_eq[sym] *= (1 + pnl/100)
        pf_eq += (coin_eq[sym] - old); eq_curve.append(pf_eq)
    
    s = pd.Series(eq_curve); peak = s.expanding().max()
    dd = ((s - peak) / peak * 100).min()
    total = len(events); wins = sum(1 for _, _, p in events if p > 0)
    return {'pnl': pf_eq-CAP, 'dd': dd, 'trades': total, 'wr': wins/total*100 if total else 0, 'eq': pf_eq}

def run_2positions(all_coin_trades):
    """Max 2 concurrent positions, each 50% of current equity"""
    # Build timeline: (time, type, sym, data)
    # type: 'entry', 'exit'
    events = []
    for sym, trades in all_coin_trades.items():
        for entry_t, exit_t, pnl in trades:
            events.append((entry_t, 'entry', sym, pnl))
            events.append((exit_t, 'exit', sym, pnl))
    events.sort()
    
    # Track: currently open positions (max 2)
    open_positions = []  # [(sym, entry_time, expected_pnl_if_closed_now)]
    pending_queue = []   # signals waiting for a slot
    eq = CAP
    eq_curve = [CAP]
    equity_at_entry = {}  # sym -> equity when entered
    entry_order = []  # order of entry for FIFO
    
    i = 0
    while i < len(events):
        t, etype, sym, pnl = events[i]
        
        if etype == 'entry':
            if len(open_positions) < 2:
                # Open position with 50% of current equity
                position_size = eq / 2
                open_positions.append(sym)
                equity_at_entry[sym] = eq
                entry_order.append(sym)
            # else: skip - no free slot (or queue it)
        
        elif etype == 'exit':
            if sym in open_positions:
                # Close position - apply PnL to that half
                position_size = equity_at_entry.get(sym, eq/2) / 2
                old_val = position_size
                new_val = position_size * (1 + pnl/100)
                eq += (new_val - old_val)
                eq_curve.append(eq)
                open_positions.remove(sym)
                entry_order.remove(sym)
                del equity_at_entry[sym]
        
        i += 1
    
    s = pd.Series(eq_curve)
    if len(s) < 2: return None
    peak = s.expanding().max()
    dd = ((s - peak) / peak * 100).min()
    
    # Count actual trades that were executed
    executed = sum(1 for _, etype, _, _ in events if etype == 'exit')
    
    return {'pnl': eq-CAP, 'dd': dd, 'trades': executed, 'eq': eq}

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Find worst coins first (from previous run)
coin_pp = {}
for pname, pdir in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) < 3: continue
        if sym not in coin_pp: coin_pp[sym] = {}
        coin_pp[sym][pname] = sum(p for _, _, p in trades)

exclude = set()
for sym, pp in coin_pp.items():
    neg = sum(1 for p in pp.values() if p < 0)
    if neg >= 2: exclude.add(sym)

print(f"🗑️ مستبعدين: {len(exclude)} عملة\n")

for pname, pdir in [('2023','2023'),('PREV','prev'),('CUR','1y')]:
    print(f"{'='*55}")
    print(f"📅 {pname}")
    
    all_trades = {}
    for sym in tradeable:
        if sym in exclude: continue  # skip bad coins
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        trades = ichimoku_trades(c8, h8, l8, o8, idx)
        if len(trades) >= 3:
            all_trades[sym] = trades
    
    N = len(all_trades)
    
    # Method 1: Equal distribution
    m1 = run_equal_distribution(all_trades, N)
    print(f"\n📊 توزيع متساوي ({N} عملة × ${CAP/N:.1f}):")
    print(f"   {m1['trades']} صفقة | WR={m1['wr']:.1f}% | ${m1['pnl']:+,.0f} | سحب={m1['dd']:.1f}% | نهائي=${m1['eq']:,.0f}")
    
    # Method 2: Max 2 positions
    m2 = run_2positions(all_trades)
    if m2:
        # Count win/loss
        wins_2p = 0; total_2p = 0
        # Recalculate WR for 2p
        for sym, trades in all_trades.items():
            for _, _, pnl in trades:
                total_2p += 1
                if pnl > 0: wins_2p += 1
        wr2 = wins_2p/total_2p*100 if total_2p else 0
        
        print(f"\n📊 صفقتين كحد أقصى (50% لكل مركز):")
        print(f"   {m2['trades']} صفقة منفذة | ${m2['pnl']:+,.0f} | سحب={m2['dd']:.1f}% | نهائي=${m2['eq']:,.0f}")
    
    print()

print("═" * 55)
print("📌 ملاحظة: أسلوب الصفقتين أقل كفاءة لأنه يفوّت صفقات رابحة أثناء انتظار الصفقات المفتوحة")
