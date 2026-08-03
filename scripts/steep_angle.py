#!/usr/bin/env python3
"""
Steep angle + pullback continuation — FET 1h/4h
صعود بزاوية 90° ثم تصحيح بسيط ثم يكمل
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

def sim(c, h, l, le, se, tp, sl):
    n=len(c); w=200; trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
    for i in range(w, n):
        if pos==1:
            ex=False; xp=c[i]
            if h[i]>=ep*(1+tp/100): ex=True; xp=ep*(1+tp/100)
            elif c[i]<=ep*(1-sl/100): ex=True; xp=c[i]
            elif se[i]: ex=True
            if ex:
                pnl=(xp/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
                if se[i] and se[i]: pos=-1; ep=c[i]
        elif pos==-1:
            ex=False; xp=c[i]
            if l[i]<=ep*(1-tp/100): ex=True; xp=ep*(1-tp/100)
            elif c[i]>=ep*(1+sl/100): ex=True; xp=c[i]
            elif le[i]: ex=True
            if ex:
                pnl=(1-xp/ep)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
                if le[i]: pos=1; ep=c[i]
        if pos==0:
            if le[i]: pos=1; ep=c[i]
            elif se[i]: pos=-1; ep=c[i]
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

def slope(c, period):
    """Linear regression slope in % per bar"""
    y = c[-period:]
    x = np.arange(period)
    slope = np.polyfit(x, y, 1)[0]
    return slope / np.mean(y) * 100  # % per bar

def compute_slopes_and_pullbacks(c, h, l, n):
    """Pre-compute angle and pullback metrics for all bars"""
    slope_arr = np.full(n, np.nan)
    peak_arr = np.full(n, np.nan)
    pullback_pct = np.full(n, np.nan)
    
    for i in range(50, n):
        lookback = 15  # bars
        if i < lookback: continue
        slope_arr[i] = slope(c[i-lookback+1:i+1], lookback)
        # Peak over last 5 bars
        peak_arr[i] = h[i-5:i+1].max()
        # Pullback from peak
        if peak_arr[i] > 0:
            pullback_pct[i] = (peak_arr[i] - c[i]) / peak_arr[i] * 100
    
    return slope_arr, peak_arr, pullback_pct

for tf in ['1h', '4h']:
    print(f'\n{"="*70}')
    print(f'FET {tf} — Steep Angle + Pullback')
    print(f'{"="*70}')
    
    df = fetch(tf, DAYS)
    c=df['close'].values; h=df['high'].values; l=df['low'].values; o=df['open'].values; n=len(c); w=200
    
    slope_arr, peak_arr, pullback = compute_slopes_and_pullbacks(c, h, l, n)
    
    # Grid search: angle threshold, pullback %, entry trigger
    best = None
    for angle_min in [0.3, 0.4, 0.5, 0.6, 0.8]:  # % per bar slope
        for pb_min in [0.3, 0.5, 0.8, 1.0, 1.5]:  # min pullback %
            for pb_max in [2.0, 2.5, 3.0, 4.0, 5.0]:  # max pullback % (small correction)
                for tp in [2.0, 2.5, 3.0, 4.0]:  # TP %
                    for sl in [1.0, 1.5, 2.0]:  # SL %
                        le=np.zeros(n,bool); se=np.zeros(n,bool)
                        # Detect: steep angle was recent (last 5 bars), now pulling back slightly, breaking back up
                        for i in range(w, n):
                            if np.isnan(slope_arr[i]): continue
                            # LONG: recent steep rise + small pullback + breaking back above the pullback low
                            steep_recent = False
                            for j in range(max(0,i-5), i+1):
                                if not np.isnan(slope_arr[j]) and slope_arr[j] > angle_min:
                                    steep_recent = True; break
                            
                            if steep_recent and pullback[i] > pb_min and pullback[i] < pb_max:
                                # Entry: price breaks above previous bar's high (resume up)
                                if c[i] > h[i-1] and c[i] > o[i]:
                                    le[i] = True
                            
                            # SHORT: recent steep drop + small bounce + breaking back down
                            steep_down = False
                            for j in range(max(0,i-5), i+1):
                                if not np.isnan(slope_arr[j]) and slope_arr[j] < -angle_min:
                                    steep_down = True; break
                            
                            pullback_up = (c[i] - l[i-5:i+1].min()) / l[i-5:i+1].min() * 100
                            if steep_down and pullback_up > pb_min and pullback_up < pb_max:
                                if c[i] < l[i-1] and c[i] < o[i]:
                                    se[i] = True
                        
                        if le.sum() + se.sum() < 5: continue
                        tr,cv = sim(c,h,l,le,se,tp,sl)
                        mr = mets(tr,cv)
                        if not mr: continue
                        wr,rr,dd,eq,nw,nl,aw,al = mr
                        score = wr * 0.3 + (eq/CAP-1)*0.5 + (rr*0.2)
                        if best is None or (wr > 30 and eq > CAP and score > best.get('score', 0)):
                            best = {'score':score,'wr':wr,'rr':rr,'dd':dd,'eq':eq,'n':nw+nl,'aw':aw,'al':al,
                                    'a':angle_min,'pb_min':pb_min,'pb_max':pb_max,'tp':tp,'sl':sl,'sig':le.sum()+se.sum()}
    
    if best:
        print(f'  Best: angle>{best["a"]:.1f}% PB {best["pb_min"]:.1f}-{best["pb_max"]:.1f}% TP{best["tp"]} SL{best["sl"]}')
        print(f'  {best["n"]}t ({best["sig"]}s) WR {best["wr"]:.1f}% R:R {best["rr"]:.2f}x DD {best["dd"]:.1f}% ${best["eq"]-1000:+.0f}')
        print(f'  aW +{best["aw"]:.2f}% aL {best["al"]:.2f}%')

# Also test on FET 1h with the best params found
print(f'\n{"="*70}')
print('BEST PARAMS — Quick test on 4h with multiple exit types')
print(f'{"="*70}')
df = fetch('4h', DAYS)
c=df['close'].values; h=df['high'].values; l=df['low'].values; o=df['open'].values; n=len(c)
slope_arr, peak_arr, pullback = compute_slopes_and_pullbacks(c, h, l, n)

for tp,sl,trail,label in [(3.0,1.5,None,'TP3/SL1.5'),(4.0,1.5,None,'TP4/SL1.5'),(None,None,0.5,'Trail0.5')]:
    ext = 'trail' if trail else 'tp_sl'
    le=np.zeros(n,bool); se=np.zeros(n,bool)
    for i in range(200, n):
        if np.isnan(slope_arr[i]): continue
        steep = any(not np.isnan(slope_arr[j]) and slope_arr[j]>0.5 for j in range(max(0,i-5),i+1))
        steep_down = any(not np.isnan(slope_arr[j]) and slope_arr[j]<-0.5 for j in range(max(0,i-5),i+1))
        if steep and pullback[i]>0.5 and pullback[i]<2.5 and c[i]>h[i-1] and c[i]>o[i]: le[i]=True
        pb_up = (c[i]-l[max(0,i-5):i+1].min())/l[max(0,i-5):i+1].min()*100
        if steep_down and pb_up>0.5 and pb_up<2.5 and c[i]<l[i-1] and c[i]<o[i]: se[i]=True
    
    tr,cv = sim(c,h,l,le,se,tp,sl) if not trail else None
    if trail:
        # trail sim
        tr=[]; eq=CAP; curve=[CAP]; pos=0; ep=0; peak=0
        for i in range(200,n):
            if pos==1:
                ex=False; xp=c[i]; peak=max(peak,h[i])
                if c[i]<=peak*(1-trail/100): ex=True; xp=c[i]
                elif se[i]: ex=True
                if ex:
                    pnl=(xp/ep-1)*100-COMM*100; tr.append(pnl); eq*=(1+pnl/100); pos=0; peak=0
                    if se[i]: pos=-1; ep=c[i]; peak=l[i]
            elif pos==-1:
                ex=False; xp=c[i]; peak=min(peak,l[i]) if peak else l[i]
                if c[i]>=peak*(1+trail/100): ex=True; xp=c[i]
                elif le[i]: ex=True
                if ex:
                    pnl=(1-xp/ep)*100-COMM*100; tr.append(pnl); eq*=(1+pnl/100); pos=0; peak=0
                    if le[i]: pos=1; ep=c[i]; peak=h[i]
            if pos==0:
                if le[i]: pos=1; ep=c[i]; peak=h[i]
                elif se[i]: pos=-1; ep=c[i]; peak=l[i]
            curve.append(eq)
        if pos:
            pnl=((c[-1]/ep-1)*100-COMM*100) if pos==1 else ((1-c[-1]/ep)*100-COMM*100)
            tr.append(pnl); eq*=(1+pnl/100); curve.append(eq)
        cv=curve
    
    mr=mets(tr,cv)
    if mr:
        wr,rr,dd,eq,nw,nl,aw,al=mr
        ico='+' if eq>CAP else '-'
        print(f'  {label:<12} {nw+nl:>4d}t WR {wr:>5.1f}% R:R {rr:.2f}x DD {dd:>5.1f}% {ico}${eq-1000:>+7.0f}')
