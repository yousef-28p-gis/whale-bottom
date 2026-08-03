#!/usr/bin/env python3
"""Per-coin Ichimoku optimization: find best params per coin on PREV, validate on 2023+CUR"""
import json, os, numpy as np, pandas as pd

COMM = 0.002; MAX_SLIPPAGE = 1.5; COOLDOWN_8H = 2

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

def run_ichimoku(c, h, l, o, tenkan, kijun, senkou, tp, sl):
    n = len(c)
    if n < senkou + 30: return None
    
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
    
    pnls = []; eq = 1000; cv = [1000]
    pos = 0; ep = 0; cool = 0; side = 0
    
    for i in range(senkou + shift, n):
        if np.isnan(sa[i]) or np.isnan(sb[i]): continue
        cloud_top = max(sa[i], sb[i]); cloud_bot = min(sa[i], sb[i])
        above = c[i] > cloud_top; below_cloud = c[i] < cloud_bot
        golden = t_arr[i] > k_arr[i] and t_arr[i-1] <= k_arr[i-1]
        death = t_arr[i] < k_arr[i] and t_arr[i-1] >= k_arr[i-1]
        
        if pos:
            if side == 1:
                if h[i] >= ep * (1 + tp/100):
                    pnls.append(tp - COMM*100); eq *= (1 + (tp-COMM*100)/100); pos = 0; cool = COOLDOWN_8H
                elif l[i] <= ep * (1 - sl/100):
                    pnl = max((c[i]/ep - 1)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    pnls.append(pnl); eq *= (1 + pnl/100); pos = 0; cool = COOLDOWN_8H
            else:
                if l[i] <= ep * (1 - tp/100):
                    pnls.append(tp - COMM*100); eq *= (1 + (tp-COMM*100)/100); pos = 0; cool = COOLDOWN_8H
                elif h[i] >= ep * (1 + sl/100):
                    pnl = max((1 - c[i]/ep)*100 - COMM*100, -sl*MAX_SLIPPAGE - COMM*100)
                    pnls.append(pnl); eq *= (1 + pnl/100); pos = 0; cool = COOLDOWN_8H
        
        if not pos and cool == 0:
            if above and golden: pos = 1; ep = c[i]; side = 1
            elif below_cloud and death: pos = 1; ep = c[i]; side = -1
        if not pos and cool > 0: cool -= 1
        cv.append(eq)
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100 if side == 1 else (1 - c[-1]/ep)*100 - COMM*100
        pnls.append(pnl); eq *= (1 + pnl/100)
    
    if len(pnls) < 5: return None
    w = sum(1 for p in pnls if p > 0)
    s = pd.Series(cv)
    dd = ((s - s.expanding().max()) / s.expanding().max() * 100).min()
    return {'trades': len(pnls), 'wr': w/len(pnls)*100, 'pnl': eq-1000, 'dd': dd}

# Param grid
params = [
    (3, 9, 18, 'Ultra'),
    (5, 13, 26, 'Fast'),
    (7, 22, 44, 'Crypto'),
    (9, 26, 52, 'Standard'),
]
tp_sl = [(3,1.5), (4,2), (5,2.5), (6,3)]

# Load coins
with open('/data/trading28/config/shariah_coins.json') as f:
    d = json.load(f)
tradeable = sorted(d['halal'] + d['halal2'])

# Find bad coins to exclude
cpp = {}
for pdir in ['2023','prev','1y']:
    for sym in tradeable:
        data = load(sym, pdir)
        if data is None: continue
        resampled = resample_8h(*data)
        if resampled is None: continue
        c8, h8, l8, o8, idx = resampled
        r = run_ichimoku(c8, h8, l8, o8, 3, 9, 18, 5, 2.5)
        if r:
            if sym not in cpp: cpp[sym] = {}
            cpp[sym][pdir] = r['pnl']

exclude = set()
for sym, pp in cpp.items():
    neg = sum(1 for p in pp.values() if p < 0)
    if neg >= 2: exclude.add(sym)

clean = sorted(set(tradeable) - exclude)
print(f"عملات نظيفة: {len(clean)}")
print(f"تجربة {len(params)}×{len(tp_sl)}={len(params)*len(tp_sl)} مجموعة لكل عملة...\n")

# Optimize per coin: train on PREV, pick best
best_configs = {}
total_best_trades = 0
results_2023 = []; results_prev = []; results_cur = []
grand_total_trades = 0

for sym in clean:
    best_pnl = -99999
    best_cfg = None
    best_r_prev = None
    
    data_prev = load(sym, 'prev')
    if data_prev is None: continue
    rp = resample_8h(*data_prev)
    if rp is None: continue
    c8p, h8p, l8p, o8p, idxp = rp
    
    for tenkan, kijun, senkou, pname in params:
        for tp, sl in tp_sl:
            r = run_ichimoku(c8p, h8p, l8p, o8p, tenkan, kijun, senkou, tp, sl)
            if r is None: continue
            if r['pnl'] > best_pnl:
                best_pnl = r['pnl']
                best_cfg = (sym, tenkan, kijun, senkou, pname, tp, sl)
                best_r_prev = r
    
    if best_cfg is None: continue
    
    # Validate on 2023 and CUR with same config
    _, tenkan, kijun, senkou, pname, tp, sl = best_cfg
    
    # 2023
    r23 = None
    data_23 = load(sym, '2023')
    if data_23:
        rp23 = resample_8h(*data_23)
        if rp23:
            c8, h8, l8, o8, idx = rp23
            r23 = run_ichimoku(c8, h8, l8, o8, tenkan, kijun, senkou, tp, sl)
    
    # CUR
    rcur = None
    data_cur = load(sym, '1y')
    if data_cur:
        rpcur = resample_8h(*data_cur)
        if rpcur:
            c8, h8, l8, o8, idx = rpcur
            rcur = run_ichimoku(c8, h8, l8, o8, tenkan, kijun, senkou, tp, sl)
    
    best_configs[sym] = best_cfg
    if best_r_prev: results_prev.append((sym, best_r_prev))
    if r23: results_2023.append((sym, r23))
    if rcur: results_cur.append((sym, rcur))
    
    t23 = r23['trades'] if r23 else 0
    tp_ = best_r_prev['trades']
    tc = rcur['trades'] if rcur else 0
    total_best_trades += tp_
    grand_total_trades += tp_ + t23 + tc

print(f"✅ تم تحسين {len(best_configs)} عملة\n")
print(f"📊 إجمالي الصفقات في PREV (التدريب): {total_best_trades}")
print(f"📊 إجمالي الصفقات في 3 فترات: {grand_total_trades}")

# Summary stats
print(f"\n{'='*70}")
print(f"📋 ملخص الإعدادات المختارة:")
param_counts = {}
tp_sl_counts = {}
for sym, cfg in best_configs.items():
    _, tenkan, kijun, senkou, pname, tp, sl = cfg
    key = f"{pname} ({tenkan}/{kijun}/{senkou})"
    param_counts[key] = param_counts.get(key, 0) + 1
    tskey = f"TP{tp}/SL{sl}"
    tp_sl_counts[tskey] = tp_sl_counts.get(tskey, 0) + 1

print("\nإعدادات إيشيموكو:")
for k, v in sorted(param_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} عملة")

print("\nTP/SL:")
for k, v in sorted(tp_sl_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} عملة")

# Top 10 coins by PREV PnL
print(f"\n🏆 أفضل 10 عملات (PREV):")
for sym, r in sorted(results_prev, key=lambda x: -x[1]['pnl'])[:10]:
    cfg = best_configs[sym]
    print(f"  {sym:>8s} | {cfg[4]:8s} TP{cfg[5]}/SL{cfg[6]} | {r['trades']:3d} صفقة | WR={r['wr']:.0f}% | ${r['pnl']:+,.0f}")

# Aggregate PnL by period
prev_total_pnl = sum(r['pnl'] for _, r in results_prev)
p23_total_pnl = sum(r['pnl'] for _, r in results_2023)
cur_total_pnl = sum(r['pnl'] for _, r in results_cur)
print(f"\n💰 إجمالي الأرباح (تدريب PREV): ${prev_total_pnl:+,.0f}")
print(f"💰 إجمالي الأرباح (اختبار 2023): ${p23_total_pnl:+,.0f}")
print(f"💰 إجمالي الأرباح (اختبار CUR): ${cur_total_pnl:+,.0f}")
