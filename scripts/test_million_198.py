#!/usr/bin/env python3
"""
استراتيجية المليون دولار — 198 عملة × سنة كاملة — بيانات محملة
EMA Ribbon + Multi-TF + ستوب صغير + هدف صغير
"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

COMM = 0.002; CAP = 1000
DATA = '/data/trading28/data/whale_15m_1y'

def load_coin(symbol):
    p = os.path.join(DATA, f'{symbol}.json')
    if not os.path.exists(p): return None
    with open(p) as f:
        d = json.load(f)
    return {
        'ts': pd.to_datetime(d['ts'], unit='ms'),
        'open': np.array(d['o'], dtype=float),
        'high': np.array(d['h'], dtype=float),
        'low': np.array(d['l'], dtype=float),
        'close': np.array(d['c'], dtype=float),
        'volume': np.array(d['v'], dtype=float),
    }

def ema(s, p):
    return pd.Series(s).ewm(span=p, adjust=False).mean().values

def resample_4h(d):
    """Resample 15m data to 4h for higher TF filter"""
    df = pd.DataFrame({'c': d['close']}, index=d['ts'])
    return df['c'].resample('4h').last().dropna().values

def simulate(le, c, h, l_, n, tp, sl, cd_bars=12):
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0; cd = 0
    for i in range(200, n):
        if pos == 1:
            if h[i] >= ep * (1+tp/100):
                trades.append(tp - COMM*100); eq *= (1+(tp-COMM*100)/100); pos = 0; cd = cd_bars
            elif l_[i] <= ep * (1-sl/100):
                trades.append(-sl - COMM*100); eq *= (1+(-sl-COMM*100)/100); pos = 0; cd = cd_bars
        if pos == 0 and cd == 0 and le[i]:
            pos = 1; ep = c[i]
        if pos == 0 and cd > 0: cd -= 1
        curve.append(eq)
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100
        trades.append(pnl); eq *= (1+pnl/100); curve.append(eq)
    return trades, curve, eq

def metrics(le, c, h, l_, n, tp, sl):
    tr, cv, eq = simulate(le, c, h, l_, n, tp, sl)
    if len(tr) < 5: return None
    w = [p for p in tr if p > 0]; l = [p for p in tr if p <= 0]
    wr = len(w)/len(tr)*100
    aw = np.mean(w) if w else 0; al = abs(np.mean(l)) if l else 0
    dd = ((pd.Series(cv) - pd.Series(cv).expanding().max()) / pd.Series(cv).expanding().max() * 100).min()
    return {'t': len(tr), 'wr': wr, 'aw': aw, 'al': al, 'dd': dd, 'eq': eq}

def test_coin(sym):
    d = load_coin(sym)
    if d is None: return None
    c = d['close']; h = d['high']; l_ = d['low']; o = d['open']; v = d['volume']
    n = len(c)
    if n < 500: return None

    # ── EMA Ribbon ──
    e5 = ema(c,5); e9 = ema(c,9); e13 = ema(c,13)
    e21 = ema(c,21); e34 = ema(c,34); e50 = ema(c,50)

    # Entry 1: Ribbon alignment start — all ordered
    ribbon_align = np.zeros(n, bool)
    for i in range(200, n):
        if (c[i] > e5[i] > e9[i] > e13[i] > e21[i] > e34[i] > e50[i] and
            not (c[i-1] > e5[i-1] > e9[i-1] > e13[i-1] > e21[i-1] > e34[i-1] > e50[i-1])):
            ribbon_align[i] = True

    # Entry 2: Price cross above e21 + e13 > e21
    ribbon_cross = np.zeros(n, bool)
    for i in range(200, n):
        if c[i] > e21[i] and c[i-1] <= e21[i-1] and c[i] > o[i] and e13[i] > e21[i]:
            ribbon_cross[i] = True

    # Entry 3: Pullback to e9 within ribbon uptrend
    ribbon_pb = np.zeros(n, bool)
    for i in range(200, n):
        if (c[i] > e9[i] and c[i] < e5[i] and c[i] > c[i-1] and
            e21[i] > e50[i] and c[i] > e21[i]):
            ribbon_pb[i] = True

    # Multi-TF filter from 15m data: resample to 4h
    try:
        c4h = resample_4h(d)
        e50_4h = ema(c4h, 50); e200_4h = ema(c4h, 200)
        # Align to 15m: every 16 bars = 4h
        mtf_ok = np.zeros(n, bool)
        for i in range(200, n):
            idx_4h = i // 16
            if idx_4h >= len(e50_4h) or idx_4h >= len(e200_4h): continue
            if np.isnan(e50_4h[idx_4h]) or np.isnan(e200_4h[idx_4h]): continue
            if e50_4h[idx_4h] > e200_4h[idx_4h]:
                mtf_ok[i] = True
    except:
        mtf_ok = np.ones(n, bool)

    results = []
    entries = {
        'Ribbon ترتيب': ribbon_align,
        'Ribbon تقاطع': ribbon_cross,
        'Ribbon ارتداد': ribbon_pb,
    }

    tp_sl_pairs = [(0.8,0.4),(1.0,0.5),(1.2,0.6),(1.5,0.75),(1.5,0.5),(2.0,1.0)]

    for ename, raw in entries.items():
        # With filter
        le_f = raw & mtf_ok
        for tp, sl in tp_sl_pairs:
            m = metrics(le_f, c, h, l_, n, tp, sl)
            if m:
                m['name'] = f'{ename} ✅'
                m['tp'] = tp; m['sl'] = sl
                results.append(m)
        # Without filter
        for tp, sl in [(1.0,0.5),(1.5,0.75),(2.0,1.0)]:
            m = metrics(raw, c, h, l_, n, tp, sl)
            if m:
                m['name'] = f'{ename} ⚠️'
                m['tp'] = tp; m['sl'] = sl
                results.append(m)

    return {'symbol': sym, 'results': results, 'n': n}

# ── Run ──
print('🔄 اختبار 198 عملة...')
with open(os.path.join(DATA, '_manifest.json')) as f:
    manifest = json.load(f)

coins = sorted([f.replace('.json','') for f in os.listdir(DATA) if f.endswith('.json') and f != '_manifest.json'])
print(f'   {len(coins)} عملة متاحة\n')

all_data = []
for i, sym in enumerate(coins):
    r = test_coin(sym)
    if r:
        all_data.append(r)
    if (i+1) % 30 == 0:
        print(f'   {i+1}/{len(coins)}...')

print(f'\nتم: {len(all_data)} عملة\n')

# ── Aggregate ──
print('='*80)
print('💵 المليون دولار | 198 عملة | سنة كاملة | 15m | ستوب صغير + هدف صغير')
print('='*80)

by_key = defaultdict(list)
for coin in all_data:
    for r in coin['results']:
        by_key[(r['name'], r['tp'], r['sl'])].append({**r, 'symbol': coin['symbol']})

# Best TP/SL per entry type
for ename in ['Ribbon ترتيب ✅', 'Ribbon تقاطع ✅', 'Ribbon ارتداد ✅']:
    print(f'\n── {ename} ──')
    print(f'{"TP/SL":>10} {"عملات":>5} {"صفقات":>6} {"WR":>6} {"R:R":>5} {"سحب":>6} {"محفظة":>10} {"ربحانة":>6}')
    print('-'*60)
    
    keys = [(k,v) for k,v in by_key.items() if k[0] == ename]
    for (_, tp, sl), items in sorted(keys, key=lambda x: -sum(i['eq'] for i in x[1])):
        total_t = sum(i['t'] for i in items)
        if total_t < 20: continue
        eq_sum = sum(i['eq'] for i in items)
        avg_wr = np.mean([i['wr'] for i in items])
        avg_dd = np.mean([i['dd'] for i in items])
        avg_rr = np.mean([i['aw']/(i['al']+0.001) for i in items])
        winning = sum(1 for i in items if i['eq'] > CAP)
        ico = '+' if eq_sum > CAP*len(items) else '-'
        profit = eq_sum - CAP * len(items)
        print(f'{tp:.1f}%/{sl:.1f}%  {len(items):>4} {total_t:>6} {avg_wr:>5.1f}% {avg_rr:>4.2f}x {avg_dd:>5.1f}% {ico}${profit:>+9.0f} {winning:>5}')

# Compare: no filter
print(f'\n── بدون فلتر (مقارنة) ──')
print(f'{"TP/SL":>10} {"عملات":>5} {"صفقات":>6} {"WR":>6} {"محفظة":>10} {"ربحانة":>6}')
print('-'*50)
for ename in ['Ribbon ترتيب ⚠️', 'Ribbon تقاطع ⚠️', 'Ribbon ارتداد ⚠️']:
    keys = [(k,v) for k,v in by_key.items() if k[0] == ename]
    for (_, tp, sl), items in sorted(keys, key=lambda x: -sum(i['eq'] for i in x[1])):
        total_t = sum(i['t'] for i in items)
        if total_t < 20: continue
        eq_sum = sum(i['eq'] for i in items)
        avg_wr = np.mean([i['wr'] for i in items])
        winning = sum(1 for i in items if i['eq'] > CAP)
        profit = eq_sum - CAP * len(items)
        ico = '+' if profit > 0 else '-'
        cn = ename.replace(' ⚠️','')
        print(f'{cn:<12} {tp:.1f}%/{sl:.1f}%  {len(items):>4} {total_t:>6} {avg_wr:>5.1f}% {ico}${profit:>+9.0f} {winning:>5}')

# ── TOP COINS ──
print(f'\n🏆 أفضل 20 عملة (أعلى ربح):\n')
best_coins = []
for coin in all_data:
    best = max(coin['results'], key=lambda x: x['eq'])
    best_coins.append({**best, 'symbol': coin['symbol']})
best_coins.sort(key=lambda x: x['eq'], reverse=True)

print(f'{"عملة":<8} {"استراتيجية":<22} {"TP/SL":>10} {"صفقات":>5} {"WR":>6} {"سحب":>6} {"ربح":>8}')
print('-'*70)
for r in best_coins[:20]:
    ico = '+' if r['eq'] > CAP else '-'
    print(f'{r["symbol"]:<8} {r["name"]:<22} {r["tp"]:.1f}%/{r["sl"]:.1f}%  {r["t"]:>5} {r["wr"]:>5.1f}% {r["dd"]:>5.1f}% {ico}${r["eq"]-CAP:>+7.1f}')

# ── Overall stats ──
print(f'\n📊 إحصائيات عامة (كل الاستراتيجيات + الفلتر):')
all_with_filter = [r for coin in all_data for r in coin['results'] if '✅' in r['name']]
profitable = [r for r in all_with_filter if r['eq'] > CAP]
print(f'   عملات مختبرة: {len(all_data)}')
print(f'   نتائج كلية: {len(all_with_filter)}')
print(f'   ربحانة: {len(profitable)} ({len(profitable)/max(1,len(all_with_filter))*100:.0f}%)')
print(f'   متوسط WR للربحانة: {np.mean([r["wr"] for r in profitable]):.1f}%' if profitable else '')
print(f'   متوسط ربح للربحانة: ${np.mean([r["eq"]-CAP for r in profitable]):.1f}' if profitable else '')

print('\n✅ انتهى')
