#!/usr/bin/env python3
"""
استراتيجية المليون دولار — EMA Ribbon + Multi-TF
ستوب صغير + هدف صغير
"""
import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import time, warnings
warnings.filterwarnings('ignore')

COMM = 0.002; CAP = 1000; DAYS = 60
COINS = ['SOL','NEAR','SUI','BNB','XRP','DOGE','DOT','ADA','AVAX','LINK','MATIC','UNI','ATOM','ARB','OP','INJ']

def fetch_tf(symbol, tf, days):
    ex = ccxt.binance({'timeout': 15000})
    since = ex.parse8601((datetime.utcnow() - timedelta(days=days+3)).isoformat())
    all_c = []
    while True:
        batch = ex.fetch_ohlcv(f'{symbol}/USDT', tf, since=since, limit=1000)
        if not batch: break
        all_c.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000: break
    df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True); df.sort_index(inplace=True)
    return df

def align(hi_series, target_idx):
    return hi_series.shift(1).reindex(target_idx, method='ffill').values

def ema(s,p):
    return pd.Series(s).ewm(span=p,adjust=False).mean().values

def simulate(le, c15, h15, l15, n15, tp, sl, cooldown=6, trail=False):
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0; cd = 0
    for i in range(200, n15):
        if pos == 1:
            # SL hit first
            if l15[i] <= ep * (1-sl/100):
                pnl = -sl - COMM*100
                trades.append(pnl); eq *= (1+pnl/100); pos = 0; cd = cooldown
            # TP hit
            elif h15[i] >= ep * (1+tp/100):
                pnl = tp - COMM*100
                trades.append(pnl); eq *= (1+pnl/100); pos = 0; cd = cooldown
        if pos == 0 and cd == 0 and le[i]:
            pos = 1; ep = c15[i]
        if pos == 0 and cd > 0: cd -= 1
        curve.append(eq)
    if pos:
        pnl = (c15[-1]/ep - 1)*100 - COMM*100
        trades.append(pnl); eq *= (1+pnl/100); curve.append(eq)
    return trades, curve, eq

def metrics(le, c15, h15, l15, n15, tp, sl):
    tr, cv, eq = simulate(le, c15, h15, l15, n15, tp, sl)
    if len(tr) < 3: return None
    w = [p for p in tr if p > 0]; l = [p for p in tr if p <= 0]
    wr = len(w)/len(tr)*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    dd = ((pd.Series(cv) - pd.Series(cv).expanding().max()) / pd.Series(cv).expanding().max() * 100).min()
    return {'t': len(tr), 'wr': wr, 'aw': aw, 'al': al, 'dd': dd, 'eq': eq, 'sigs': le.sum()}

def test_coin(symbol):
    try:
        d15 = fetch_tf(symbol, '15m', DAYS)
        d1h = fetch_tf(symbol, '1h', DAYS)
        d4h = fetch_tf(symbol, '4h', DAYS)
        d1d = fetch_tf(symbol, '1d', DAYS+200)
    except Exception as e:
        return None

    c = d15['close'].values; h = d15['high'].values
    l_ = d15['low'].values; o = d15['open'].values
    v = d15['volume'].values; idx = d15.index
    n = len(c)

    # Multi-TF: EMA50 > EMA200 on 15m+1h+4h
    ema50_d = align(pd.Series(ema(d1d['close'],50), index=d1d.index), idx)
    ema200_d = align(pd.Series(ema(d1d['close'],200), index=d1d.index), idx)
    ema50_4 = align(pd.Series(ema(d4h['close'],50), index=d4h.index), idx)
    ema200_4 = align(pd.Series(ema(d4h['close'],200), index=d4h.index), idx)
    ema50_1 = align(pd.Series(ema(d1h['close'],50), index=d1h.index), idx)
    ema200_1 = align(pd.Series(ema(d1h['close'],200), index=d1h.index), idx)

    mtf_ok = np.zeros(n, bool)
    for i in range(200, n):
        ok = (not np.isnan(ema50_1[i]) and not np.isnan(ema200_1[i]) and
              ema50_1[i] > ema200_1[i] and
              not np.isnan(ema50_4[i]) and not np.isnan(ema200_4[i]) and
              ema50_4[i] > ema200_4[i] and
              not np.isnan(ema50_d[i]) and not np.isnan(ema200_d[i]) and
              ema50_d[i] > ema200_d[i])
        mtf_ok[i] = ok
    gp = mtf_ok.sum() / n * 100

    # EMA Ribbon: 5, 9, 13, 21, 34, 50
    e5 = ema(c,5); e9 = ema(c,9); e13 = ema(c,13)
    e21 = ema(c,21); e34 = ema(c,34); e50 = ema(c,50)

    # Entry 1: Ribbon alignment — all EMAs in order: price > e5 > e9 > e13 > e21 > e34 > e50
    ribbon_entry = np.zeros(n, bool)
    for i in range(200, n):
        if (c[i] > e5[i] > e9[i] > e13[i] > e21[i] > e34[i] > e50[i] and
            not (c[i-1] > e5[i-1] > e9[i-1] > e13[i-1] > e21[i-1] > e34[i-1] > e50[i-1])):
            ribbon_entry[i] = True

    # Entry 2: Price crosses above EMA Ribbon cluster (price above e21 + e13 > e21)
    ribbon_cross = np.zeros(n, bool)
    for i in range(200, n):
        if (c[i] > e21[i] and c[i-1] <= e21[i-1] and c[i] > o[i] and
            e13[i] > e21[i]):
            ribbon_cross[i] = True

    # Entry 3: Ribbon fan-out (EMAs spreading apart) + price above
    ribbon_fan = np.zeros(n, bool)
    for i in range(200, n):
        spread = (e5[i] - e50[i]) / e50[i] * 100
        prev_spread = (e5[i-1] - e50[i-1]) / e50[i-1] * 100
        if c[i] > e5[i] and spread > prev_spread and spread > 0.5:
            ribbon_fan[i] = True

    # Entry 4: Simple — price above e21, e21 > e50, pullback to e9
    ribbon_pullback = np.zeros(n, bool)
    for i in range(200, n):
        if (c[i] > e9[i] and c[i] < e5[i] and c[i] > c[i-1] and
            e21[i] > e50[i] and c[i] > e21[i]):
            ribbon_pullback[i] = True

    entries = {}
    for name, raw in [
        ('Ribbon ترتيب كامل', ribbon_entry),
        ('Ribbon تقاطع+EMA21', ribbon_cross),
        ('Ribbon تمدد المراوح', ribbon_fan),
        ('Ribbon ارتداد E9', ribbon_pullback),
    ]:
        entries[f'{name} ✅فلتر'] = raw & mtf_ok
        entries[f'{name} ⚠️بدون'] = raw

    # TP/SL combos — small targets
    tp_sl_pairs = [
        (0.8, 0.4), (1.0, 0.5), (1.2, 0.6), (1.5, 0.75),
        (2.0, 1.0), (1.0, 0.4), (1.5, 0.5),
    ]

    results = []
    for ename, le in entries.items():
        for tp, sl in tp_sl_pairs:
            m = metrics(le, c, h, l_, n, tp, sl)
            if m and m['t'] >= 3:
                m['name'] = ename; m['tp'] = tp; m['sl'] = sl
                m['gp'] = gp
                results.append(m)

    return {'symbol': symbol, 'gp': gp, 'results': results}

# ── Run ──
print('🔄 جاري اختبار استراتيجية المليون دولار...')
all_data = []
for sym in COINS:
    print(f'  {sym}...', end=' ', flush=True)
    r = test_coin(sym)
    if r:
        all_data.append(r)
        print(f'{len(r["results"])} variants')
    else:
        print('SKIP')
    time.sleep(0.3)

# ── Group by entry+filter ──
by_key = defaultdict(list)
for coin in all_data:
    for r in coin['results']:
        key = (r['name'], r['tp'], r['sl'])
        by_key[key].append({**r, 'symbol': coin['symbol']})

print(f'\n{"="*90}')
print(f'💵 استراتيجية المليون دولار | 16 عملة | 60 يوم | 15m')
print(f'{"="*90}')

# Find best TP/SL per entry type
entry_names = sorted(set(k[0] for k in by_key))
for ename in entry_names:
    is_filtered = '✅' in ename
    print(f'\n── {ename} ──')
    print(f'{"TP/SL":>10} {"عملات":>5} {"صفقات":>5} {"WR":>6} {"R:R":>5} {"سحب":>6} {"محفظة":>10}')
    print('-'*55)

    filtered_keys = [(k, v) for k, v in by_key.items() if k[0] == ename]
    for (_, tp, sl), items in sorted(filtered_keys, key=lambda x: -sum(i['eq'] for i in x[1])):
        total_t = sum(i['t'] for i in items)
        total_w = sum(len([p for p in []] ) for i in items)  # placeholder
        # Recalculate
        all_trades_flat = []
        for i in items:
            # recalc from simulate
            pass
        eq_sum = sum(i['eq'] for i in items)
        avg_wr = np.mean([i['wr'] for i in items])
        avg_dd = np.mean([i['dd'] for i in items])
        active = sum(1 for i in items if i['t'] >= 3)
        ico = '+' if eq_sum > CAP * len(items) else '-'
        profit = eq_sum - CAP * len(items)
        print(f'{tp:.1f}%/{sl:.1f}%  {active:>4}/{len(items):<2} {total_t:>5} {avg_wr:>5.1f}% {np.mean([i["aw"] for i in items]):>5.2f}x {avg_dd:>5.1f}% {ico}${profit:>+9.0f}')

# Best individual
print(f'\n🏆 أفضل 15 صفقة فردية:\n')
best = []
for coin in all_data:
    for r in coin['results']:
        if r['t'] >= 3:
            best.append({**r, 'symbol': coin['symbol']})
best.sort(key=lambda x: x['eq'], reverse=True)
for r in best[:15]:
    ico = '+' if r['eq'] > CAP else '-'
    has_filter = '✅' in r['name']
    print(f'{r["name"]:<35} {r["symbol"]:<6} TP{r["tp"]:.1f}/SL{r["sl"]:.1f} {r["t"]:>3d}t WR{r["wr"]:>5.1f}% DD{r["dd"]:>5.1f}% {ico}${r["eq"]-CAP:>+7.1f}')

# Best by coin
print(f'\n📈 أفضل استراتيجية لكل عملة:\n')
for coin in sorted(all_data, key=lambda x: x['symbol']):
    if not coin['results']: continue
    best_r = max(coin['results'], key=lambda x: x['eq'])
    ico = '+' if best_r['eq'] > CAP else '-'
    print(f'{coin["symbol"]:<6} GP{coin["gp"]:>4.0f}% | {best_r["name"]:<35} TP{best_r["tp"]:.1f}/SL{best_r["sl"]:.1f} {best_r["t"]:>3d}t WR{best_r["wr"]:>5.1f}% {ico}${best_r["eq"]-CAP:>+7.1f}')

print('\n✅ انتهى')
