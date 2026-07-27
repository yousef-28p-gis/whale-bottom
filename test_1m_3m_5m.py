#!/usr/bin/env python3
"""1m 3m 5m test"""
import json, numpy as np, pandas as pd, os
from collections import defaultdict

CACHE='data/1m_test'
TP=3.5;SL=1.5;PL=30;TRAIL=0.10;MH=6;COMM=0.20;WHALE_MIN=0.50;STR=50
BLOCK_HOURS={1,3,6,12,0,4}
COINS=['ADA','ETH','SOL','DOGE','AVAX','LINK','DOT','ATOM','GRT','SAND']

def resample_ohlcv(df, minutes):
    df=df.copy()
    df.set_index('ts',inplace=True)
    ohlcv=df.resample(f'{minutes}min').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna()
    ohlcv.reset_index(inplace=True)
    return ohlcv

results={}

for tf_name, tf_val, use_blocks in [('1m',1,False),('3m',3,True),('5m',5,True),('15m',15,True)]:
    all_pnl=[];all_exits=defaultdict(int)
    
    for sym in COINS:
        fp=f'{CACHE}/{sym}_1m.json'
        if not os.path.exists(fp): continue
        with open(fp) as f: data=json.load(f)
        df=pd.DataFrame(data)
        df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'},inplace=True)
        df['ts']=pd.to_datetime(df['ts'],unit='ms')
        df=df.sort_values('ts').reset_index(drop=True)
        if tf_val>1: df=resample_ohlcv(df,tf_val)
        
        LB=30
        df['lo']=df['low'].rolling(LB).min()
        df['lc']=abs(df['low']-df['low'].shift(1))/df['low']*100
        df['sm']=df['lc'].ewm(span=3,adjust=False).mean()
        df['hi']=df['sm'].rolling(LB).max()
        df['raw']=np.where(df['low']<=df['lo'],(df['sm']+df['hi']*2)/3,0)
        df['whale']=df['raw'].ewm(span=3,adjust=False).mean().fillna(0)
        df['spike']=(df['whale']>df['whale'].shift(1))&(df['whale'].shift(1)<=0.03)
        df['wf']=df['whale'].rolling(2).mean();df['ws']=df['whale'].rolling(5).mean()
        df['wp']=df['whale'].rolling(50).max()
        df['str']=(df['whale']/df['wp'].replace(0,np.nan)*100).fillna(0)
        df['vma']=df['volume'].rolling(20).mean()
        df['entry']=(df['spike']&(df['wf']>df['ws'])&(df['str']>STR)&(df['volume']>df['vma']*1.0))
        d=df['close'].diff();g=d.where(d>0,0).rolling(14).mean();l=(-d.where(d<0,0)).rolling(14).mean()
        rs=g/l.replace(0,np.nan);df['rsi']=100-(100/(1+rs))
        
        for i in range(max(50,LB+10),len(df)-5):
            row=df.iloc[i]
            if not row['entry']: continue
            if float(row['whale'])<WHALE_MIN: continue
            if i+1<len(df) and float(df.iloc[i+1]['whale'])>=0.35: continue
            rsi=float(row['rsi'])
            if np.isnan(rsi) or rsi>=25: continue
            if use_blocks and row['ts'].hour in BLOCK_HOURS: continue
            ps=max(0,i-50);pb=float(df.iloc[ps]['close']);ep=float(row['close'])
            if (ep-pb)/pb*100>=0: continue
            if i+1>=len(df): continue
            if float(df.iloc[i+1]['close'])<=float(df.iloc[i+1]['open']): continue
            
            tp_p=ep*(1+TP/100);sl_p=ep*(1-SL/100);pl_p=ep+(tp_p-ep)*(PL/100)
            pt=False;pk=ep;tr=0;pnl=None;ex='EOD'
            for k in range(i+1,len(df)):
                cur=float(df.iloc[k]['close']);h=(k-i)*(tf_val/60)
                if h>MH: pnl=round((cur-ep)/ep*100-COMM,4);ex='TIME';break
                if cur>=tp_p: pnl=round(TP-COMM,4);ex='TP';break
                if cur<=sl_p: pnl=round(-SL-COMM,4);ex='SL';break
                if not pt and cur>=pl_p: pt=True;pk=cur;tr=cur*(1-TRAIL/100)
                if pt:
                    if cur>pk: pk=cur;tr=cur*(1-TRAIL/100)
                    if cur<=tr: pnl=round((tr-ep)/ep*100-COMM,4);ex='TRAIL';break
            if pnl is None: pnl=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4)
            all_pnl.append(pnl);all_exits[ex]+=1
    
    arr=np.array(all_pnl)
    wa=arr[arr>0];la=arr[arr<0]
    wc=len(wa);lc=len(la)
    aw=wa.mean() if wc>0 else 0;al=la.mean() if lc>0 else 0
    wr=wc/len(arr)*100 if len(arr)>0 else 0
    results[tf_name]=(len(arr),wc,lc,wr,arr.sum(),aw,al,aw/abs(al) if al!=0 else 0,all_exits)

print(f'{"TF":<8} {"ت":>4} {"WR":>6} {"صافي":>8} {"متوسط":>7} {"R:R":>5} {"TP":>4} {"SL":>4} {"TRAIL":>5} {"TIME":>5}')
print('-'*65)
for tf in ['1m','3m','5m','15m']:
    t,wc,lc,wr,net,aw,al,rr,ex=results[tf]
    tpc=ex.get('TP',0);slc=ex.get('SL',0);trc=ex.get('TRAIL',0);tmc=ex.get('TIME',0)+ex.get('EOD',0)
    avg=net/t if t>0 else 0
    print(f'{tf:<8} {t:>4} {wr:>5.1f}% {net:>+7.1f}% {avg:>+6.3f}% {rr:>4.1f}x {tpc:>4} {slc:>4} {trc:>5} {tmc:>5}')
print('DONE')
