#!/usr/bin/env python3
"""Whale+SSL — ADA/USDT 1m — LB50/E3/SSL10"""
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

COMM = 0.002; CAP = 1000

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def whale_signal(l, n, LB=50, smooth=3):
    ln = pd.Series(l).rolling(LB).min().values
    at_low = l <= ln
    low_change = np.zeros(n)
    for i in range(1,n): low_change[i] = abs(l[i]-l[i-1])/l[i]*100
    sc = pd.Series(low_change).ewm(span=smooth,adjust=False).mean().values
    hc = pd.Series(sc).rolling(LB).max().values
    strength = np.where(at_low, (sc + hc*2)/3, 0)
    wp = pd.Series(strength).ewm(span=smooth,adjust=False).mean().values
    up = wp > np.roll(wp, 1)
    return wp, up

def ssl_lines(h, l, n, period=10):
    sup = pd.Series(h).rolling(period).mean().values
    sdn = pd.Series(l).rolling(period).mean().values
    return sup, sdn

def sim(le, c, h, l_, n, tp, sl, cd=12):
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                t.append(tp-COMM*100); eq*=(1+(tp-COMM*100)/100); pos=0; cool=cd
            elif l_[i]<=ep*(1-sl/100):
                t.append(-sl-COMM*100); eq*=(1+(-sl-COMM*100)/100); pos=0; cool=cd
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100); cv.append(eq)
    return t,cv,eq

# Fetch
print('Fetching ADA 1m...')
ex = ccxt.binance({'timeout': 15000})
# Try 30 days, but 1m data limit may be shorter
since = ex.parse8601((datetime.utcnow() - timedelta(days=7)).isoformat())
all_c = []
while True:
    batch = ex.fetch_ohlcv('ADA/USDT', '1m', since=since, limit=1000)
    if not batch: break
    all_c.extend(batch)
    since = batch[-1][0] + 1
    if len(batch) < 1000: break

df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','volume'])
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df.set_index('ts', inplace=True); df.sort_index(inplace=True)
print(f'   {len(df)} candles | {df.index[0]} → {df.index[-1]}')

c=df['close'].values; h=df['high'].values
l_=df['low'].values; o=df['open'].values; n=len(c)

# Whale + SSL
wp, wp_up = whale_signal(l_, n, LB=50, smooth=3)
sup, sdn = ssl_lines(h, l_, n, period=10)

# Entry: whale rising + price > SSL up
le = np.zeros(n, bool)
for i in range(200, n):
    if wp_up[i] and wp[i] > wp[i-2]*2 and c[i] > sup[i]:
        le[i] = True

# Also: whale only (for comparison)
le_whale = np.zeros(n, bool)
for i in range(200, n):
    if wp_up[i] and wp[i] > wp[i-2]*1.5 and c[i] > o[i]:
        le_whale[i] = True

# SSL only
le_ssl = np.zeros(n, bool)
for i in range(200, n):
    if c[i] > sup[i] and c[i-1] <= sup[i-1] and c[i] > o[i]:
        le_ssl[i] = True

print(f'\n🐋 Whale+SSL signals: {le.sum()}')
print(f'🐋 Whale only signals: {le_whale.sum()}')
print(f'🔒 SSL only signals: {le_ssl.sum()}')

print(f'\n{"="*75}')
print(f'ADA/USDT 1m | LB50/E3/SSL10 | {len(df)} candles | {df.index[0].date()} → {df.index[-1].date()}')
print(f'{"="*75}')
print(f'{"Strategy":<18} {"TP/SL":>10} {"Trades":>7} {"WR":>6} {"R:R":>5} {"DD":>6} {"Equity":>9}')
print('-'*65)

for tp, sl in [(0.5,0.25),(0.8,0.4),(1.0,0.5),(1.5,0.75),(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
    for name, le_arr in [('Whale+SSL', le), ('Whale only', le_whale), ('SSL only', le_ssl)]:
        tr,cv,eq = sim(le_arr, c, h, l_, n, tp, sl)
        if len(tr) < 3: continue
        w=[p for p in tr if p>0]; lo=[p for p in tr if p<=0]
        wr=len(w)/len(tr)*100
        aw=np.mean(w) if w else 0; al=abs(np.mean(lo)) if lo else 0
        rr=aw/(al+0.001)
        dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
        ico='+' if eq>CAP else '-'
        print(f'{name:<18} {tp:.1f}%/{sl:.1f}%   {len(tr):>5} {wr:>5.1f}% {rr:>4.2f}x {dd:>5.1f}% {ico}${eq-CAP:>+8.1f}')

# Best for Whale+SSL
print(f'\n🏆 Best Whale+SSL configs:')
best = []
for tp, sl in [(0.5,0.25),(0.8,0.4),(1.0,0.5),(1.5,0.75),(2.0,1.0),(3.0,1.5),(5.0,2.5)]:
    tr,cv,eq = sim(le, c, h, l_, n, tp, sl)
    if len(tr)>=3: best.append((tp,sl,eq,len(tr)))
best.sort(key=lambda x: -x[2])
for tp,sl,eq,nt in best[:5]:
    print(f'   TP{tp:.1f}%/SL{sl:.1f}%: {nt}t → ${eq:.1f} ({eq-CAP:+.1f})')

print('\n✅ Done')
