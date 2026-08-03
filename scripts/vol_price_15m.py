#!/usr/bin/env python3
"""
Volume + Price Action strategies — FET/USDT 15m — 180 days
No RSI, no lagging indicators
"""
import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta

SYMBOL = 'FET/USDT'; COMM = 0.002; DAYS = 180; CAP = 1000

def fetch(tf, days):
    ex = ccxt.binance({'timeout': 15000})
    since = ex.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_c = []
    while True:
        batch = ex.fetch_ohlcv(SYMBOL, tf, since=since, limit=1000)
        if not batch: break
        all_c.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000: break
    df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True); df.sort_index(inplace=True)
    return df

def sim(c, h, l, le, se, exit_mode='reverse', tp=None, sl=None, trail=None):
    n=len(c); w=200; trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0; peak=0
    for i in range(w, n):
        if pos==1:
            ex=False; xp=c[i]
            if exit_mode=='reverse' and se[i]: ex=True
            elif exit_mode=='tp_sl':
                if h[i]>=ep*(1+tp/100): ex=True; xp=ep*(1+tp/100)
                elif c[i]<=ep*(1-sl/100): ex=True; xp=c[i]
                elif se[i]: ex=True
            elif exit_mode=='trail':
                peak=max(peak,h[i])
                if c[i]<=peak*(1-trail/100): ex=True; xp=c[i]
                elif se[i]: ex=True
            if ex:
                pnl=(xp/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0; peak=0
                if exit_mode=='reverse' and se[i]: pos=-1; ep=c[i]; peak=l[i]
        elif pos==-1:
            ex=False; xp=c[i]
            if exit_mode=='reverse' and le[i]: ex=True
            elif exit_mode=='tp_sl':
                if l[i]<=ep*(1-tp/100): ex=True; xp=ep*(1-tp/100)
                elif c[i]>=ep*(1+sl/100): ex=True; xp=c[i]
                elif le[i]: ex=True
            elif exit_mode=='trail':
                peak=min(peak,l[i]) if peak!=0 else l[i]
                if c[i]>=peak*(1+trail/100): ex=True; xp=c[i]
                elif le[i]: ex=True
            if ex:
                pnl=(1-xp/ep)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0; peak=0
                if exit_mode=='reverse' and le[i]: pos=1; ep=c[i]; peak=h[i]
        if pos==0:
            if le[i]: pos=1; ep=c[i]; peak=h[i]
            elif se[i]: pos=-1; ep=c[i]; peak=l[i]
        curve.append(eq)
    if pos:
        pnl=((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve

def mets(tr, cv):
    if not tr or len(tr)<5: return None
    nt=len(tr); w=[p for p in tr if p>0]; l=[p for p in tr if p<=0]
    wr=len(w)/nt*100
    aw=np.mean(w) if w else 0; al=np.mean(l) if l else 0
    rr=abs(aw/al) if al else 99
    dd=((pd.Series(cv)-pd.Series(cv).expanding().max())/pd.Series(cv).expanding().max()*100).min()
    return wr,rr,dd,cv[-1],len(w),len(l),aw,al

print('Fetching FET 15m...')
df=fetch('15m',180)
c=df['close'].values; o=df['open'].values; h=df['high'].values; l=df['low'].values
v=df['volume'].values; n=len(c); w=200
print(f'{len(df)} candles')

# ═══════════ PRE-COMPUTE ═══════════
# Volume MA
vsma = pd.Series(v).rolling(20).mean().values  # volume SMA 20

# Volume percentile (top 20% & top 10%)
v_roll = pd.Series(v).rolling(100).apply(lambda x: (x.iloc[-1] > x.quantile(0.8)), raw=False).fillna(0).values.astype(bool)
v_top10 = pd.Series(v).rolling(100).apply(lambda x: (x.iloc[-1] > x.quantile(0.9)), raw=False).fillna(0).values.astype(bool)

# Candle body and wick
body = np.abs(c - o)
body_pct = body / c * 100
lower_wick = np.minimum(c, o) - l
upper_wick = h - np.maximum(c, o)
wick_ratio = (lower_wick + upper_wick) / (body + 0.0001)
pin_bar_bull = (lower_wick > body*3) & (upper_wick < body*0.5) & (body>0)  # long lower wick
pin_bar_bear = (upper_wick > body*3) & (lower_wick < body*0.5) & (body>0)  # long upper wick

# Candle range
candle_range = h - l
atr = pd.Series(candle_range).rolling(14).mean().values

# Marubozu (big body candle)
marubozu_bull = (body_pct > 1.0) & (lower_wick < body*0.2) & (upper_wick < body*0.2) & (c>o)
marubozu_bear = (body_pct > 1.0) & (lower_wick < body*0.2) & (upper_wick < body*0.2) & (c<o)

# Swing highs/lows (20 bar lookback)
swing_high = pd.Series(h).rolling(20).max().values
swing_low = pd.Series(l).rolling(20).min().values

# Close near swing
near_swing_high = (h > swing_high*0.995)
near_swing_low = (l < swing_low*1.005)

# Volume climax (2x average)
v_climax = v > vsma*2

# ═══════════ STRATEGIES ═══════════
all_res = []

# === 1: Pin Bar + Volume spike ===
le=np.zeros(n,bool); se=np.zeros(n,bool)
for i in range(w, n):
    if pin_bar_bull[i] and v[i] > vsma[i] and l[i] <= l[i-3:i].min():
        le[i]=True
    elif pin_bar_bear[i] and v[i] > vsma[i] and h[i] >= h[i-3:i].max():
        se[i]=True
tr,cv=sim(c,h,l,le,se,'tp_sl',2.0,1.0)
mr=mets(tr,cv)
if mr:
    wr,rr,dd,eq,nw,nl,aw,al=mr
    all_res.append((eq,wr,dd,rr,nl+nw,aw,al,'1-PinBar+Vol TP2/SL1'))

for tp,sl,trail,ext in [(3.0,1.0,None,'TP3/SL1'),(4.0,1.5,None,'TP4/SL1.5'),(None,None,0.2,'TR0.2')]:
    ext_mode = 'trail' if trail else 'tp_sl'
    tr,cv=sim(c,h,l,le,se,ext_mode,tp,sl,trail)
    mr=mets(tr,cv)
    if mr:
        wr,rr,dd,eq,nw,nl,aw,al=mr
        all_res.append((eq,wr,dd,rr,nl+nw,aw,al,f'1-PinBar+Vol {ext}'))

# === 2: Volume Climax + Reversal ===
le=np.zeros(n,bool); se=np.zeros(n,bool)
for i in range(w+1, n):
    # Huge volume + price stopped falling + next candle green
    if v[i-1] > vsma[i-1]*3 and l[i-1] < l[i-5:i-1].min() and c[i] > c[i-1] and c[i] > o[i]:
        le[i]=True
    elif v[i-1] > vsma[i-1]*3 and h[i-1] > h[i-5:i-1].max() and c[i] < c[i-1] and c[i] < o[i]:
        se[i]=True
tr,cv=sim(c,h,l,le,se,'tp_sl',3.0,1.0)
mr=mets(tr,cv)
if mr:
    wr,rr,dd,eq,nw,nl,aw,al=mr
    all_res.append((eq,wr,dd,rr,nl+nw,aw,al,'2-VolClimax+Rev TP3/SL1'))

# === 3: Big Candle + Volume surge (momentum) ===
le=np.zeros(n,bool); se=np.zeros(n,bool)
for i in range(w, n):
    if marubozu_bull[i] and v[i] > vsma[i]*1.5 and c[i] > h[i-1] and candle_range[i] > atr[i]*1.2:
        le[i]=True
    elif marubozu_bear[i] and v[i] > vsma[i]*1.5 and c[i] < l[i-1] and candle_range[i] > atr[i]*1.2:
        se[i]=True
tr,cv=sim(c,h,l,le,se,'trail',None,None,0.3)
mr=mets(tr,cv)
if mr:
    wr,rr,dd,eq,nw,nl,aw,al=mr
    all_res.append((eq,wr,dd,rr,nl+nw,aw,al,'3-Maru+Vol+Range TR0.3'))

# === 4: Swing break + Volume ===
le=np.zeros(n,bool); se=np.zeros(n,bool)
for i in range(w, n):
    if c[i] > swing_high[i-1] and v[i] > vsma[i]*1.2 and c[i] > o[i]:
        le[i]=True
    elif c[i] < swing_low[i-1] and v[i] > vsma[i]*1.2 and c[i] < o[i]:
        se[i]=True
tr,cv=sim(c,h,l,le,se,'trail',None,None,0.3)
mr=mets(tr,cv)
if mr:
    wr,rr,dd,eq,nw,nl,aw,al=mr
    all_res.append((eq,wr,dd,rr,nl+nw,aw,al,'4-SwingBreak+Vol TR0.3'))

# === 5: Sell climax + Bullish engulf ===
le=np.zeros(n,bool); se=np.zeros(n,bool)
for i in range(w+1, n):
    # Previous red candle with high vol, current green candle engulfs it
    if c[i-1]<o[i-1] and v[i-1]>vsma[i-1]*1.5 and c[i]>c[i-1] and o[i]<c[i-1] and c[i]>o[i-1]:
        le[i]=True
    elif c[i-1]>o[i-1] and v[i-1]>vsma[i-1]*1.5 and c[i]<c[i-1] and o[i]>c[i-1] and c[i]<o[i-1]:
        se[i]=True
tr,cv=sim(c,h,l,le,se,'tp_sl',2.5,1.0)
mr=mets(tr,cv)
if mr:
    wr,rr,dd,eq,nw,nl,aw,al=mr
    all_res.append((eq,wr,dd,rr,nl+nw,aw,al,'5-Engulf+Vol TP2.5/SL1'))

# === 6: Inside bar breakout + Volume ===
inside_bar = np.zeros(n, bool)
for i in range(w, n):
    if h[i] < h[i-1] and l[i] > l[i-1]: inside_bar[i]=True
le=np.zeros(n,bool); se=np.zeros(n,bool)
for i in range(w+1, n):
    if inside_bar[i-1] and c[i] > h[i-1] and v[i] > vsma[i]:
        le[i]=True
    elif inside_bar[i-1] and c[i] < l[i-1] and v[i] > vsma[i]:
        se[i]=True
tr,cv=sim(c,h,l,le,se,'tp_sl',2.0,1.0)
mr=mets(tr,cv)
if mr:
    wr,rr,dd,eq,nw,nl,aw,al=mr
    all_res.append((eq,wr,dd,rr,nl+nw,aw,al,'6-InsideBar+Vol TP2/SL1'))

# === RANK ===
print(f'\n{"="*75}')
print(f'VOLUME + PRICE ACTION — FET 15m — {DAYS}d')
print(f'{"="*75}')
print(f'{"#":>3} {"Strategy":<35} {"T":>4} {"WR":>6} {"R:R":>5} {"DD":>6} {"$":>8} {"aW":>6} {"aL":>6}')
print('-'*75)
for i,x in enumerate(sorted(all_res, key=lambda x: x[1], reverse=True)):
    eq,wr,dd,rr,n,aw,al,nm=x
    ico='+' if eq>1000 else '-'
    print(f'{i+1:>3} {nm:<35} {n:>4} {wr:>5.1f}% {rr:>4.2f}x {dd:>5.1f}% {ico}\${eq-1000:>7.0f} {aw:>+5.2f}% {al:>+5.2f}%')

print(f'\n{"="*75}')
print('BEST CONFIGS:')
print(f'{"="*75}')
for i,x in enumerate(sorted(all_res, key=lambda x: x[0], reverse=True)[:10]):
    eq,wr,dd,rr,n,aw,al,nm=x
    print(f'{i+1:>2}. {nm:<35} {n:>4}t WR {wr:>5.1f}% R:R {rr:.2f}x DD {dd:>5.1f}% ${eq-1000:>+7.0f}')
