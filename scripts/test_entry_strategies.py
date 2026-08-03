#!/usr/bin/env python3
"""Test 7 different entry strategies on 16 coins with Multi-TF filter"""
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
    s = hi_series.shift(1)
    return s.reindex(target_idx, method='ffill').values

def simulate(le, c15, h15, n15, tp=5.0, sl=2.5, cooldown=6):
    trades = []; eq = CAP; curve = [CAP]; pos = 0; ep = 0; cd = 0
    for i in range(200, n15):
        if pos == 1:
            if h15[i] >= ep * (1+tp/100):
                pnl = tp - COMM*100; trades.append(pnl); eq *= (1+pnl/100); pos = 0; cd = cooldown
            elif c15[i] <= ep * (1-sl/100):
                pnl = (c15[i]/ep - 1)*100 - COMM*100; trades.append(pnl); eq *= (1+pnl/100); pos = 0; cd = cooldown
        if pos == 0 and cd == 0 and le[i]:
            pos = 1; ep = c15[i]
        if pos == 0 and cd > 0: cd -= 1
        curve.append(eq)
    if pos:
        pnl = (c15[-1]/ep - 1)*100 - COMM*100; trades.append(pnl); eq *= (1+pnl/100); curve.append(eq)
    return trades, curve, eq

def metrics(name, le, c15, h15, n15, tp, sl):
    tr, cv, eq = simulate(le, c15, h15, n15, tp, sl)
    if len(tr) < 3: return None
    w = [p for p in tr if p > 0]; l = [p for p in tr if p <= 0]
    wr = len(w)/len(tr)*100
    aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
    dd = ((pd.Series(cv) - pd.Series(cv).expanding().max()) / pd.Series(cv).expanding().max() * 100).min()
    return {'name': name, 't': len(tr), 'wr': wr, 'aw': aw, 'al': al,
            'dd': dd, 'eq': eq, 'sigs': le.sum(), 'trades': tr}

# ── Entry strategies ──
def entry_ema21_bounce(c, o, ema21, n):
    le = np.zeros(n, bool)
    for i in range(200,n):
        if c[i] > o[i] and c[i] > ema21[i] and c[i-1] <= ema21[i-1]*1.01:
            le[i] = True
    return le

def entry_ssl_breakout(c, h, l, n):
    le = np.zeros(n, bool)
    ssl_up = pd.Series(h).rolling(10).mean().values
    for i in range(200, n):
        if c[i] > ssl_up[i] and c[i-1] <= ssl_up[i-1]:
            le[i] = True
    return le

def entry_ema_cross(c, ema5, ema21, n):
    le = np.zeros(n, bool)
    for i in range(200, n):
        if ema5[i] > ema21[i] and ema5[i-1] <= ema21[i-1]:
            le[i] = True
    return le

def entry_ema9_pullback(c, ema9, n):
    le = np.zeros(n, bool)
    for i in range(200, n):
        if c[i] > c[i-1] and abs(c[i] - ema9[i])/ema9[i] < 0.005 and c[i] > ema9[i]*0.995:
            le[i] = True
    return le

def entry_breakout_high(c, h, n, lookback=20):
    le = np.zeros(n, bool)
    for i in range(200, n):
        rh = h[max(0,i-lookback):i].max()
        if c[i] > rh and c[i-1] <= rh:
            le[i] = True
    return le

def entry_engulfing(c, o, v, n):
    le = np.zeros(n, bool)
    va = pd.Series(v).rolling(20).mean().values
    for i in range(200, n):
        if (c[i] > o[i] and o[i-1] > c[i-1] and o[i] <= c[i-1] and c[i] >= o[i-1] and v[i] > va[i]*1.3):
            le[i] = True
    return le

def entry_volume_surge(c, v, n):
    le = np.zeros(n, bool)
    va = pd.Series(v).rolling(20).mean().values
    for i in range(200, n):
        if v[i] > va[i]*2.0 and (c[i]-c[i-1])/c[i-1] > 0.005:
            le[i] = True
    return le

# ── Test coin ──
def test_coin(symbol):
    print(f'\n🪙 {symbol}...', end=' ', flush=True)
    try:
        d15 = fetch_tf(symbol, '15m', DAYS)
        d1h = fetch_tf(symbol, '1h', DAYS)
        d4h = fetch_tf(symbol, '4h', DAYS)
    except Exception as e:
        print(f'SKIP: {e}')
        return None

    c15 = d15['close'].values; h15 = d15['high'].values
    l15 = d15['low'].values; o15 = d15['open'].values
    v15 = d15['volume'].values; idx15 = d15.index
    n15 = len(c15)

    ema = lambda s,p: pd.Series(s).ewm(span=p,adjust=False).mean().values

    ema50_15_a = align(pd.Series(ema(d15['close'],50), index=d15.index), idx15)
    ema200_15_a = align(pd.Series(ema(d15['close'],200), index=d15.index), idx15)
    ema50_1h_a = align(pd.Series(ema(d1h['close'],50), index=d1h.index), idx15)
    ema200_1h_a = align(pd.Series(ema(d1h['close'],200), index=d1h.index), idx15)
    ema50_4h_a = align(pd.Series(ema(d4h['close'],50), index=d4h.index), idx15)
    ema200_4h_a = align(pd.Series(ema(d4h['close'],200), index=d4h.index), idx15)

    mtf_ok = np.zeros(n15, bool)
    for i in range(200, n15):
        ok = (not np.isnan(ema50_15_a[i]) and not np.isnan(ema200_15_a[i]) and
              ema50_15_a[i] > ema200_15_a[i] and
              not np.isnan(ema50_1h_a[i]) and not np.isnan(ema200_1h_a[i]) and
              ema50_1h_a[i] > ema200_1h_a[i] and
              not np.isnan(ema50_4h_a[i]) and not np.isnan(ema200_4h_a[i]) and
              ema50_4h_a[i] > ema200_4h_a[i])
        mtf_ok[i] = ok
    gp = mtf_ok.sum()/n15*100

    ema5_15 = pd.Series(c15).ewm(span=5,adjust=False).mean().values
    ema9_15 = pd.Series(c15).ewm(span=9,adjust=False).mean().values
    ema21_15 = pd.Series(c15).ewm(span=21,adjust=False).mean().values

    entries = {
        'EMA21 ارتداد': entry_ema21_bounce(c15,o15,ema21_15,n15),
        'SSL اختراق': entry_ssl_breakout(c15,h15,l15,n15),
        'EMA5x21 تقاطع': entry_ema_cross(c15,ema5_15,ema21_15,n15),
        'EMA9 ارتداد': entry_ema9_pullback(c15,ema9_15,n15),
        'اختراق 20 قمة': entry_breakout_high(c15,h15,n15,20),
        'ابتلاع+حجم': entry_engulfing(c15,o15,v15,n15),
        'حجم متضاعف': entry_volume_surge(c15,v15,n15),
    }

    results = []
    for name, raw in entries.items():
        le = raw & mtf_ok
        m = metrics(name, le, c15, h15, n15, 5.0, 2.5)
        if m:
            m['green_pct'] = gp; m['raw_sigs'] = raw.sum()
            results.append(m)
        # Without filter
        m2 = metrics(f'{name} ⚠️', raw, c15, h15, n15, 5.0, 2.5)
        if m2:
            m2['green_pct'] = gp
            results.append(m2)

    print(f'{len(results)} variants')
    return {'symbol': symbol, 'green_pct': gp, 'results': results, 'n': n15}

# ── Run ──
all_data = []
for sym in COINS:
    r = test_coin(sym)
    if r: all_data.append(r)
    time.sleep(0.3)

# ── Summary ──
print('\n' + '='*95)
print('🧪 16 عملة | 60 يوم | 15m | فلتر 15m+1h+4h | TP5%/SL2.5%')
print('='*95)

by_strat = defaultdict(list)
for coin in all_data:
    for r in coin['results']:
        by_strat[r['name']].append({**r, 'symbol': coin['symbol']})

# Filtered only
print(f'\n📊 مع الفلتر:\n')
print(f'{"الاستراتيجية":<22} {"عملات":>6} {"صفقات":>5} {"WR":>6} {"سحب":>6} {"محفظة":>10}')
print('-'*60)

for name, items in sorted(by_strat.items(), key=lambda x: -sum(i['eq'] for i in x[1])):
    if '⚠️' in name: continue
    total_t = sum(i['t'] for i in items)
    if total_t == 0: continue
    total_w = sum(len([p for p in i['trades'] if p>0]) for i in items)
    total_l = sum(len([p for p in i['trades'] if p<=0]) for i in items)
    wr_a = total_w/(total_w+total_l)*100 if (total_w+total_l)>0 else 0
    avg_dd = np.mean([i['dd'] for i in items if i['t']>0])
    eq_s = sum(i['eq'] for i in items)
    active = sum(1 for i in items if i['t']>=3)
    ico = '+' if eq_s > CAP*len(items) else '-'
    print(f'{name:<22} {active:>3}/{len(items):<2} {total_t:>5} {wr_a:>5.1f}% {avg_dd:>5.1f}% {ico}${eq_s-CAP*len(items):>+9.0f}')

# No filter
print(f'\n⚠️ بدون فلتر:\n')
print(f'{"الاستراتيجية":<22} {"صفقات":>5} {"WR":>6} {"محفظة":>10}')
print('-'*50)
for name, items in sorted(by_strat.items(), key=lambda x: -sum(i['eq'] for i in x[1])):
    if '⚠️' not in name: continue
    total_t = sum(i['t'] for i in items)
    if total_t == 0: continue
    total_w = sum(len([p for p in i['trades'] if p>0]) for i in items)
    total_l = sum(len([p for p in i['trades'] if p<=0]) for i in items)
    wr_a = total_w/(total_w+total_l)*100 if (total_w+total_l)>0 else 0
    eq_s = sum(i['eq'] for i in items)
    ico = '+' if eq_s > CAP*len(items) else '-'
    cn = name.replace(' ⚠️','')
    print(f'{cn:<22} {total_t:>5} {wr_a:>5.1f}% {ico}${eq_s-CAP*len(items):>+9.0f}')

# Best individual
print(f'\n🏆 أفضل الصفقات الفردية (مع فلتر):\n')
best = []
for coin in all_data:
    for r in coin['results']:
        if '⚠️' not in r['name'] and r['t']>=3:
            best.append(r)
best.sort(key=lambda x: x['eq'], reverse=True)
for r in best[:15]:
    ico = '+' if r['eq']>CAP else '-'
    sym = r.get('symbol', '?')
    print(f'{r["name"]:<22} {sym:<6} {r["t"]:>3d}t WR{r["wr"]:>5.1f}% DD{r["dd"]:>5.1f}% {ico}${r["eq"]-CAP:>+7.1f}')

# Distribution: coins per strategy minimum
print(f'\n📈 توزيع العملات الرابحة (≥$1000):\n')
for name, items in sorted(by_strat.items(), key=lambda x: -sum(1 for i in x[1] if i['eq']>CAP and i['t']>=3)):
    if '⚠️' in name: continue
    wcoins = [(i['symbol'], i['eq']) for i in items if i['eq']>CAP and i['t']>=3]
    print(f'{name:<22} {len(wcoins)} عملات ربحانة: {", ".join(f"{s}(${e:.0f})" for s,e in wcoins) if wcoins else "-"}')

print('\n✅ انتهى')
