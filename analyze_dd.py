#!/usr/bin/env python3
"""Analyze drawdown events for Cloud Hunter RSI>50"""
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

def ichimoku_trades(c, h, l, o, idx):
    tenkan, kijun, senkou = 3, 9, 18; tp, sl = 5, 2.5
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
    
    rsi = compute_rsi(c)
    
    trades = []
    pos = 0; ep = 0; cool = 0; entry_idx = None
    
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top = max(sa[i], sb[i])
        above = c[i] > cloud_top
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        
        signal = above and golden and not np.isnan(rsi[i]) and rsi[i] > 50
        
        if pos:
            if h[i] >= ep * (1 + 5/100):
                trades.append((entry_idx, idx[i], 5 - COMM*100))
                pos = 0; cool = COOLDOWN
            elif l[i] <= ep * (1 - 2.5/100):
                pnl = max((c[i]/ep - 1)*100 - COMM*100, -2.5*MAX_SLIPPAGE - COMM*100)
                trades.append((entry_idx, idx[i], pnl))
                pos = 0; cool = COOLDOWN
        
        if not pos and cool == 0 and signal:
            pos = 1; ep = c[i]; entry_idx = idx[i]
        
        if not pos and cool > 0: cool -= 1
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100
        trades.append((entry_idx, idx[-1], pnl))
    return trades

with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
all_coins = sorted(d['halal'] + d['halal2'])

# Collect all trades with timestamps for PREV period (worst DD)
period = 'prev'
pname = 'PREV'

print(f"🔍 تحليل السحب — {pname} — كل العملات — RSI>50\n")

all_timeline = []
coin_stats = {}

for sym in all_coins:
    data = load(sym, period)
    if data is None: continue
    resampled = resample_8h(*data)
    if resampled is None: continue
    c8, h8, l8, o8, idx = resampled
    trades = ichimoku_trades(c8, h8, l8, o8, idx)
    if len(trades) < 3: continue
    
    coin_pnl = sum(p for _, _, p in trades)
    coin_wins = sum(1 for _, _, p in trades if p > 0)
    coin_losses = len(trades) - coin_wins
    coin_max_loss = min((p for _, _, p in trades if p < 0), default=0)
    
    # Streak analysis
    curr_streak = 0; max_loss_streak = 0
    for _, _, p in trades:
        if p < 0:
            curr_streak += 1
            max_loss_streak = max(max_loss_streak, curr_streak)
        else:
            curr_streak = 0
    
    # Timeline entries
    for entry_t, exit_t, pnl in trades:
        all_timeline.append((entry_t, 'entry', sym, pnl))
        all_timeline.append((exit_t, 'exit', sym, pnl))
    
    coin_stats[sym] = {
        'trades': len(trades), 'pnl': coin_pnl,
        'wins': coin_wins, 'losses': coin_losses,
        'wr': coin_wins/len(trades)*100,
        'max_loss': coin_max_loss,
        'max_loss_streak': max_loss_streak,
    }

all_timeline.sort()

# Simulate equity curve with detailed tracking
eq = 1000; peak = 1000; max_dd = 0
open_positions = {}  # sym -> (alloc, entry_eq)
eq_history = []
dd_events = []
executed = 0; wins = 0

for t, etype, sym, pnl in all_timeline:
    if etype == 'entry':
        if len(open_positions) < 2:
            alloc = eq / 2
            open_positions[sym] = (alloc, eq)
    elif etype == 'exit':
        if sym in open_positions:
            alloc, entry_eq = open_positions.pop(sym)
            new_val = alloc * (1 + pnl/100)
            eq += (new_val - alloc)
            executed += 1
            if pnl > 0: wins += 1
            
            if eq > peak:
                if peak > 0 and (eq - peak)/peak < -0.02:  # recovered from DD
                    pass
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
                dd_events.append({
                    'eq': eq, 'dd': dd, 'peak': peak,
                    'sym': sym, 'pnl': pnl,
                    'open_count': len(open_positions),
                    'time': t,
                })

# Print summary
print(f"📊 إجمالي:")
print(f"  صفقات منفذة: {executed}")
print(f"  WR: {wins/executed*100:.1f}%")
print(f"  ربح: ${eq-1000:+.0f}")
print(f"  سحب: {max_dd:.1f}%")
print(f"  رأس المال النهائي: ${eq:.0f}")

# Worst drawdown events
print(f"\n📉 أسوأ 5 أحداث سحب:")
dd_events.sort(key=lambda x: x['dd'])
for i, ev in enumerate(dd_events[:5]):
    dt = pd.to_datetime(ev['time'], unit='ms')
    print(f"  {i+1}. {dt} | {ev['sym']} | PnL={ev['pnl']:+.1f}% | سحب={ev['dd']:.1f}% | رصيد=${ev['eq']:.0f} | مفتوح={ev['open_count']}")

# Clustered losses analysis
print(f"\n🔴 تحليل تجمع الخسائر:")
losses = [(t, sym, pnl) for t, etype, sym, pnl in all_timeline if etype == 'exit' and pnl < 0]
losses.sort()

# Find clusters (5+ losses within 7 days)
cluster_count = 0
cluster_losses = 0
total_cluster_loss = 0
i = 0
while i < len(losses):
    cluster = [losses[i]]
    j = i + 1
    while j < len(losses) and losses[j][0] - cluster[0][0] < 7 * 24 * 3600 * 1000:
        cluster.append(losses[j])
        j += 1
    if len(cluster) >= 5:
        cluster_count += 1
        cluster_losses += len(cluster)
        total_cluster_loss += sum(p for _, _, p in cluster)
        print(f"  كتلة {cluster_count}: {len(cluster)} خسائر في {(cluster[-1][0]-cluster[0][0])/(3600*1000):.0f}h | خسارة=${sum(p for _,_,p in cluster):.0f}")
    i = j

# Coin stats — worst offenders
print(f"\n👎 أسوأ 10 عملات (حسب سلسلة الخسائر):")
worst = sorted(coin_stats.items(), key=lambda x: x[1]['max_loss_streak'], reverse=True)[:10]
for sym, st in worst:
    print(f"  {sym}: {st['trades']} صفقة, WR={st['wr']:.0f}%, أقصى خسائر متتالية={st['max_loss_streak']}, PnL=${st['pnl']:+.0f}")

# Market correlation analysis
print(f"\n📈 تحليل فترات السحب العميق:")
dd_events.sort(key=lambda x: x['dd'])
for ev in dd_events[:3]:
    dt = pd.to_datetime(ev['time'], unit='ms')
    # Check how many coins were in drawdown at that time
    near_losses = [l for l in losses if abs(l[0] - ev['time']) < 3 * 24 * 3600 * 1000]
    unique_coins = len(set(s for _, s, _ in near_losses))
    print(f"  {dt}: سحب={ev['dd']:.1f}%, {unique_coins} عملة خاسرة خلال 3 أيام, رصيد=${ev['eq']:.0f}")
