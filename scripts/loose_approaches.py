#!/usr/bin/env python3
"""QQE/SSL variants — no EMA, loose approaches — FET 1h 180d"""
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

def rsi_s(s, p):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    return 100 - 100/(1 + g.ewm(alpha=1/p, adjust=False).mean()/l.ewm(alpha=1/p, adjust=False).mean())

def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def hma(s, l):
    half = int(max(l/2, 2)); sq = int(max(np.sqrt(l), 1))
    w1 = s.rolling(half).apply(lambda x: np.average(x, weights=np.arange(1,half+1)), raw=True)
    w2 = s.rolling(l).apply(lambda x: np.average(x, weights=np.arange(1,l+1)), raw=True)
    return (2*w1 - w2).rolling(sq).apply(lambda x: np.average(x, weights=np.arange(1,sq+1)), raw=True)

def compute_qqe(close, rsi_len, smooth, factor):
    wl = rsi_len*2-1
    rv = rsi_s(close, rsi_len); sr = ema(rv, smooth)
    ar = (sr-sr.shift(1)).abs(); sa = ema(ar, wl); da = sa*factor
    n = len(close); lb = np.full(n, np.nan); sb = np.full(n, np.nan)
    warm = max(wl+10, 50)
    for i in range(warm, n):
        ns = sr.iloc[i]+da.iloc[i]; nl = sr.iloc[i]-da.iloc[i]
        if not np.isnan(lb[i-1]) and sr.iloc[i-1]>lb[i-1] and sr.iloc[i]>lb[i-1]: lb[i]=max(lb[i-1],nl)
        else: lb[i]=nl
        if not np.isnan(sb[i-1]) and sr.iloc[i-1]<sb[i-1] and sr.iloc[i]<sb[i-1]: sb[i]=min(sb[i-1],ns)
        else: sb[i]=ns
    return sr.values, lb, sb

def sim(c, h, l, le, se):
    n=len(c); w=200; trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
    for i in range(w, n):
        if pos==0:
            if le[i]: pos=1; ep=c[i]
            elif se[i]: pos=-1; ep=c[i]
        elif pos==1:
            if se[i]:
                pnl=(c[i]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100)
                pos=-1; ep=c[i]
        elif pos==-1:
            if le[i]:
                pnl=(1-c[i]/ep)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100)
                pos=1; ep=c[i]
        curve.append(eq)
    if pos:
        pnl=((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
        trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
    return trades, curve

def mets(trades, curve):
    if not trades or len(trades)<5: return None
    pnl=trades; nt=len(pnl)
    w=[p for p in pnl if p>0]; l=[p for p in pnl if p<=0]
    wr=len(w)/nt*100
    aw=np.mean(w) if w else 0; al=np.mean(l) if l else 0
    rr=abs(aw/al) if al else 99
    dd=((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
    return wr,rr,dd,curve[-1],len(w),len(l),aw,al

print('Fetching...')
df=fetch('1h',180)
c=df['close'].values; h=df['high'].values; l=df['low'].values
n=len(c); w=200; print(f'{len(df)} candles')

pr,_,_=compute_qqe(df['close'],6,5,2.0)
sr,_,_=compute_qqe(df['close'],6,5,1.61)
pz=pr-50; sz=sr-50

# SSL variants
SSL={}
for sl in [5,10,15]:
    eh=hma(df['high'],sl).values; el=hma(df['low'],sl).values
    hv=np.zeros(n); sv=np.full(n,np.nan)
    for i in range(1,n):
        if np.isnan(eh[i]): hv[i]=hv[i-1]
        elif c[i]>eh[i]: hv[i]=1
        elif c[i]<el[i]: hv[i]=-1
        else: hv[i]=hv[i-1]
        sv[i]=eh[i] if hv[i]<0 else el[i]
    sb=np.zeros(n,bool); se=np.zeros(n,bool); up=np.zeros(n,bool); dn=np.zeros(n,bool)
    for i in range(2,n):
        if not np.isnan(sv[i]):
            sb[i]=c[i]>sv[i] and c[i-1]<=sv[i-1]
            se[i]=c[i]<sv[i] and c[i-1]>=sv[i-1]
            up[i]=c[i]>sv[i]
            dn[i]=c[i]<sv[i]
    SSL[sl]=(sb,se,up,dn)

all_res = []

# === APPROACH 1: QQE only (no SSL) ===
print('\n--- QQE ONLY ---')
for thr in [1.0, 1.5, 2.0, 2.5, 3.0]:
    qb=(sz>thr)&(pz>0); qr=(sz<-thr)&(pz<0)
    le=np.zeros(n,bool); se=np.zeros(n,bool)
    for i in range(w,n):
        if qb[i]: le[i]=True
        elif qr[i]: se[i]=True
    tr,cv=sim(c,h,l,le,se)
    mr=mets(tr,cv)
    if mr:
        wr,rr,dd,eq,nw,nl,aw,al=mr
        sg=le.sum()+se.sum(); nm=f'QQE-only T{thr}'
        print(f'  {nm:<20} {nl+nw:>4d}t ({sg}s) | WR {wr:>5.1f}% | R:R {rr:.2f}x | DD {dd:>6.1f}% | ${eq-1000:>+8.0f}')
        all_res.append((eq,wr,dd,rr,nl+nw,sg,aw,al,nm))

# === APPROACH 2: SSL only ===
print('\n--- SSL ONLY ---')
for sl in [5,10,15]:
    sb,se,_,_=SSL[sl]
    le=np.zeros(n,bool); se2=np.zeros(n,bool)
    for i in range(w,n):
        if sb[i]: le[i]=True
        elif se[i]: se2[i]=True
    tr,cv=sim(c,h,l,le,se2)
    mr=mets(tr,cv)
    if mr:
        wr,rr,dd,eq,nw,nl,aw,al=mr
        sg=sb.sum()+se.sum(); nm=f'SSL-only L{sl}'
        print(f'  {nm:<20} {nl+nw:>4d}t ({sg}s) | WR {wr:>5.1f}% | R:R {rr:.2f}x | DD {dd:>6.1f}% | ${eq-1000:>+8.0f}')
        all_res.append((eq,wr,dd,rr,nl+nw,sg,aw,al,nm))

# === APPROACH 3: QQE blue/red + SSL direction (not crossover) ===
print('\n--- QQE + SSL DIRECTION ---')
for sl in [5,10]:
    _,_,up,dn=SSL[sl]
    for bl,bm,thr in [(30,0.5,2.0),(30,0.35,1.5),(30,0.5,1.5),(50,0.5,1.5),(50,0.35,1.5)]:
        bb_b=pd.Series(pz).rolling(bl).mean().values
        bb_s=pd.Series(pz).rolling(bl).std().values
        bb_u=bb_b+bm*bb_s; bb_l=bb_b-bm*bb_s
        qb=(sz>thr)&(pz>bb_u); qr=(sz<-thr)&(pz<bb_l)
        le=np.zeros(n,bool); se=np.zeros(n,bool)
        for i in range(w,n):
            if qb[i] and up[i]: le[i]=True
            elif qr[i] and dn[i]: se[i]=True
        tr,cv=sim(c,h,l,le,se)
        mr=mets(tr,cv)
        if mr:
            wr,rr,dd,eq,nw,nl,aw,al=mr
            sg=le.sum()+se.sum(); nm=f'QQE+SSLdir L{sl} BB{bl}x{bm} T{thr}'
            if eq>CAP:
                print(f'  {nm:<35} {nl+nw:>4d}t ({sg}s) | WR {wr:>5.1f}% | R:R {rr:.2f}x | DD {dd:>6.1f}% | ${eq-1000:>+8.0f}')
            all_res.append((eq,wr,dd,rr,nl+nw,sg,aw,al,nm))

# === APPROACH 4: QQE signal OR SSL cross (whichever comes first) ===
print('\n--- QQE + SSL CROSS (OR logic) ---')
for sl in [10]:
    sb,se,_,_=SSL[sl]
    for bl,bm,thr in [(30,0.5,2.0),(30,0.35,1.5),(50,0.35,1.5)]:
        bb_b=pd.Series(pz).rolling(bl).mean().values
        bb_s=pd.Series(pz).rolling(bl).std().values
        bb_u=bb_b+bm*bb_s; bb_l=bb_b-bm*bb_s
        qb=(sz>thr)&(pz>bb_u); qr=(sz<-thr)&(pz<bb_l)
        le=np.zeros(n,bool); se2=np.zeros(n,bool)
        for i in range(w,n):
            if qb[i] or sb[i]: le[i]=True
            elif qr[i] or se[i]: se2[i]=True
        tr,cv=sim(c,h,l,le,se2)
        mr=mets(tr,cv)
        if mr:
            wr,rr,dd,eq,nw,nl,aw,al=mr
            sg=le.sum()+se2.sum(); nm=f'QQEorSSL L{sl} BB{bl}x{bm} T{thr}'
            if eq>CAP:
                print(f'  {nm:<35} {nl+nw:>4d}t ({sg}s) | WR {wr:>5.1f}% | R:R {rr:.2f}x | DD {dd:>6.1f}% | ${eq-1000:>+8.0f}')
            all_res.append((eq,wr,dd,rr,nl+nw,sg,aw,al,nm))

# === APPROACH 5: QQE only (no BB) + price momentum ===
print('\n--- QQE + MOMENTUM ---')
for thr in [1.5, 2.0, 2.5]:
    qb=(sz>thr)&(pz>0); qr=(sz<-thr)&(pz<0)
    le=np.zeros(n,bool); se=np.zeros(n,bool)
    for i in range(w,n):
        if qb[i] and c[i]>c[i-1]: le[i]=True
        elif qr[i] and c[i]<c[i-1]: se[i]=True
    tr,cv=sim(c,h,l,le,se)
    mr=mets(tr,cv)
    if mr:
        wr,rr,dd,eq,nw,nl,aw,al=mr
        sg=le.sum()+se.sum(); nm=f'QQE+mom T{thr}'
        print(f'  {nm:<20} {nl+nw:>4d}t ({sg}s) | WR {wr:>5.1f}% | R:R {rr:.2f}x | DD {dd:>6.1f}% | ${eq-1000:>+8.0f}')
        all_res.append((eq,wr,dd,rr,nl+nw,sg,aw,al,nm))

# === FINAL RANKING ===
print(f'\n{\"=\"*80}')
print('TOP 20 BY WIN RATE:')
for i,x in enumerate(sorted(all_res, key=lambda x: x[1], reverse=True)[:20]):
    eq,wr,dd,rr,n,sg,aw,al,nm=x
    ico='1st' if i==0 else '2nd' if i==1 else '3rd' if i==2 else f'{i+1}th'
    print(f'{ico:>3} {nm:<38} {n:>4d}t ({sg}s) | WR {wr:>5.1f}% | R:R {rr:.2f}x | DD {dd:>5.1f}% | ${eq-1000:>+7.0f}')

print(f'\nTOP 20 BY RETURN:')
for i,x in enumerate(sorted(all_res, key=lambda x: x[0], reverse=True)[:20]):
    eq,wr,dd,rr,n,sg,aw,al,nm=x
    ico='\U0001F947' if i==0 else '\U0001F948' if i==1 else '\U0001F949' if i==2 else f'{i+1}.'
    print(f'{ico:>3} ${eq-1000:>+7.0f} | WR {wr:>5.1f}% | {n:>4d}t | R:R {rr:.2f}x | DD {dd:>5.1f}% | {nm}')
