#!/usr/bin/env python3
"""Whale: 300 vs 400 bar comparison"""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m.csv', parse_dates=['ts'])
FEE=0.001; CAPITAL=1000

# Swings
lb=5; sh=np.zeros(len(df),dtype=bool); sl_arr=np.zeros(len(df),dtype=bool)
for i in range(lb*2,len(df)):
    w=df['high'].iloc[i-lb*2:i+1]; m=i-lb
    if df['high'].iloc[m]==w.max() and w.values.argmax()==lb: sh[i]=True
    w=df['low'].iloc[i-lb*2:i+1]
    if df['low'].iloc[m]==w.min() and w.values.argmin()==lb: sl_arr[i]=True
def nsl(idx):
    for j in range(idx-1,max(0,idx-100),-1):
        if sl_arr[j]: return df['low'].iloc[j]
    return df['low'].iloc[idx]*0.95
def nsh(idx):
    for j in range(idx-1,max(0,idx-100),-1):
        if sh[j]: return df['high'].iloc[j]
    return df['high'].iloc[idx]*1.05

df['atr']=(df['high']-df['low']).rolling(14).mean()
df['atr_ma20']=df['atr'].rolling(20).mean()
df['vol_ma20']=df['volume'].rolling(20).mean()

for BARS in [300, 400]:
    print(f"\n{'='*60}")
    print(f"🐋 WHALE {BARS}-BAR")
    print(f"{'='*60}")
    
    lowest=df['low'].rolling(BARS).min()
    at_low=(df['low']<=lowest).astype(float)
    lc=abs(df['low']-df['low'].shift(1))/df['low']*100
    sm=lc.ewm(span=3,adjust=False).mean()
    hi=sm.rolling(BARS).max()
    st=np.where(at_low>0,(sm+hi*2)/3,0)
    w=pd.Series(st).ewm(span=3,adjust=False).mean().fillna(0)
    spike=(w>w.shift(1))&(w.shift(1)<=0.02)
    wma50=w.rolling(50).mean(); wma200=w.rolling(200).mean()
    wpk=w.rolling(50).max(); wstr=w/wpk.replace(0,np.nan)*100
    
    print(f"  Spikes: {spike.sum()}", flush=True)
    
    for ws in [50,60]:
        for vm in [1.0]:
            lo=wma50>wma200; so=wma50<wma200
            le=spike&(wstr>ws)&lo&(df['volume']>df['vol_ma20']*vm)&(df['atr']>df['atr_ma20'])
            se=spike&(wstr>ws)&so&(df['volume']>df['vol_ma20']*vm)&(df['atr']>df['atr_ma20'])
            
            eis=np.where(le|se)[0]
            if len(eis)==0: continue
            
            trades=[]; it=False; ed=0; eq=CAPITAL
            cm=df['ts'].iloc[400].month; cy=df['ts'].iloc[400].year; ms=CAPITAL
            
            for ei in eis:
                if ei<500: continue
                if it and ei<ed: continue
                ts=df['ts'].iloc[ei]
                if ts.month!=cm or ts.year!=cy: cm,cy=ts.month,ts.year; ms=eq
                if (eq-ms)/ms*100<-7: continue
                
                il=le.iloc[ei]; e=df['close'].iloc[ei]
                if il: sl=nsl(ei)*0.998; tp=99999
                else: sl=nsh(ei)*1.002; tp=e-df['atr'].iloc[ei]*3
                
                end=min(ei+192,len(df)); r=None; ep=e; ex=ei
                for j in range(ei+1,end):
                    if il:
                        if df['low'].iloc[j]<=sl: r='SL'; ep=sl; ex=j; break
                        if se.iloc[j] and wstr.iloc[j]>ws: r='REV'; ep=df['close'].iloc[j]; ex=j; break
                    else:
                        if df['high'].iloc[j]>=sl: r='SL'; ep=sl; ex=j; break
                        if df['low'].iloc[j]<=tp: r='TP'; ep=tp; ex=j; break
                if r is None: r='TIME'; ep=df['close'].iloc[end-1]; ex=end-1
                
                pnl=(ep-e)/e*100
                if il: pnl-=0.2
                else: pnl=-pnl-0.2
                
                trades.append({'il':il,'r':r,'pnl':pnl,'ei':ei,'exi':ex})
                it=True; ed=ex; eq+=CAPITAL*(pnl/100)
            
            n=len(trades)
            if n==0: continue
            wins=[t for t in trades if t['pnl']>0]; wr=len(wins)/n*100
            pnls=[t['pnl'] for t in trades]
            sp=np.mean(pnls)/np.std(pnls)*np.sqrt(n) if np.std(pnls)>0 else 0
            eqs=[CAPITAL]
            for t in trades: eqs.append(eqs[-1]+CAPITAL*(t['pnl']/100))
            pk=np.maximum.accumulate(eqs); dd=(np.array(eqs)-pk)/pk*100
            lt=[t for t in trades if t['il']]; st=[t for t in trades if not t['il']]
            lwr=len([t for t in lt if t['pnl']>0])/len(lt)*100 if lt else 0
            swr=len([t for t in st if t['pnl']>0])/len(st)*100 if st else 0
            rev=sum(1 for t in trades if t['r']=='REV')
            tp=sum(1 for t in trades if t['r']=='TP')
            sl_c=sum(1 for t in trades if t['r']=='SL')
            
            print(f"  {ws}%/{vm}x: {n}T | WR:{wr:.0f}% | ${eq:,.0f} | L/S:{lwr:.0f}/{swr:.0f} | S:{sp:.2f} | DD:{dd.min():.1f}% | R/T/S:{rev}/{tp}/{sl_c}", flush=True)

print("\n✅ Done")
