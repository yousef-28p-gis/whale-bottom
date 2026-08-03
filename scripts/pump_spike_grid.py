#!/usr/bin/env python3
"""Pump24 + Spike — Threshold grid — FET 5m/15m"""
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

def ema(s, p): return s.ewm(span=p, adjust=False).mean()

for tf in ['5m','15m']:
    print(f'\n{"="*80}')
    print(f'FET {tf} — Pump24 thresholds + Spike strength')
    print(f'{"="*80}')
    df = fetch(tf, DAYS)
    c=df['close'].values; h=df['high'].values; l=df['low'].values; v=df['volume'].values
    n=len(c); w=200
    
    bars = {'5m':288,'15m':96}[tf]
    pump24 = np.full(n, np.nan)
    for i in range(bars, n):
        pump24[i] = (c[i] - c[i-bars]) / c[i-bars] * 100
    
    vsma = ema(pd.Series(v), 20).values
    
    print(f'{"Pump>":<8} {"Spike>":<8} {"Trades":>6} {"WR":>7} {"R:R":>6} {"DD":>7} {"Profit":>9} {"aW":>7} {"aL":>7} {"Best"}')
    print('-'*85)
    
    for pump_min in [0, 1.0, 2.0, 3.0, 5.0]:
     for spike_mul in [2.0, 2.5, 3.0, 4.0]:
        spike = (v > vsma*spike_mul)
        at_low = (l <= np.minimum(np.roll(l,1), np.minimum(np.roll(l,2), np.roll(l,3))))
        
        le = np.zeros(n, bool)
        for i in range(w, n):
            if np.isnan(pump24[i]): continue
            if pump24[i] > pump_min and spike[i] and at_low[i] and c[i] > c[i-1]:
                le[i] = True
        
        if le.sum() < 3: continue
        
        best = None
        for tp in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
         for sl in [0.5, 0.7, 1.0, 1.5, 2.0]:
          if sl >= tp: continue
          trades=[]; eq=CAP; curve=[CAP]; pos=0; ep=0
          for i in range(w, n):
            if pos==1:
                if h[i]>=ep*(1+tp/100):
                    pnl=(tp-COMM*100); trades.append(pnl); eq*=(1+pnl/100); pos=0
                elif c[i]<=ep*(1-sl/100):
                    pnl=(c[i]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); pos=0
            if pos==0 and le[i]: pos=1; ep=c[i]
            curve.append(eq)
          if pos:
            pnl=(c[-1]/ep-1)*100-COMM*100; trades.append(pnl); eq*=(1+pnl/100); curve.append(eq)
          if len(trades)<3: continue
          w2=[p for p in trades if p>0]; l2=[p for p in trades if p<=0]
          wr=len(w2)/len(trades)*100
          aw=np.mean(w2) if w2 else 0; al=np.mean(l2) if l2 else 0
          rr=abs(aw/al) if al else 99
          dd=((pd.Series(curve)-pd.Series(curve).expanding().max())/pd.Series(curve).expanding().max()*100).min()
          sc=eq/CAP * wr/100
          if best is None or (wr>35 and eq>CAP and (eq/CAP+wr/100)>best.get('sc',0)):
            best={'sc':sc,'wr':wr,'dd':dd,'eq':eq,'n':len(trades),'tp':tp,'sl':sl,'aw':aw,'al':al,'rr':rr}
        
        if best:
            ico='+' if best['eq']>CAP else '-'
            print(f'>{pump_min:<6.0f}% >{spike_mul:<5.1f}x  {best["n"]:>6} {best["wr"]:>6.1f}% {best["rr"]:>5.2f}x {best["dd"]:>6.1f}% {ico}${best["eq"]-1000:>+8.0f} {best["aw"]:>+6.2f}% {best["al"]:>+6.2f}% TP{best["tp"]}/SL{best["sl"]}')
        else:
            print(f'>{pump_min:<6.0f}% >{spike_mul:<5.1f}x  NO WINNERS')
