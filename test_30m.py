#!/usr/bin/env python3
"""15m vs 30m — 40 عملة"""
import json, numpy as np, pandas as pd, os
from collections import defaultdict
from datetime import timedelta

CACHE='data/5year_halal'
MH=6;COMM=0.20;WHALE_MIN=0.50;STR=50;BLOCK_HOURS={1,3,6,12,0,4}
BLACKLIST={'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','0G','ROBO','PYTH','ANKR'}

all_files=sorted([f for f in os.listdir(CACHE) if f.endswith('.json')])
coin_files=[]
for f in all_files:
    sym=f.replace('_15m.json','')
    if sym in BLACKLIST: continue
    coin_files.append(f)
coin_files=[coin_files[i] for i in range(0,len(coin_files),5)]

print(f'{len(coin_files)} coins', flush=True)
all_15m=[];all_30m=[]
ex_15m=defaultdict(int);ex_30m=defaultdict(int)

for fi,fname in enumerate(coin_files):
    sym=fname.replace('_15m.json','')
    with open(f'{CACHE}/{fname}') as f: data=json.load(f)
    df=pd.DataFrame(data,columns=['ts','open','high','low','close','volume'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms')
    df=df.sort_values('ts').reset_index(drop=True)
    if len(df)<500: continue
    
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
    
    for i in range(50,len(df)-10):
        row=df.iloc[i]
        if not row['entry']: continue
        if float(row['whale'])<WHALE_MIN: continue
        if i+1<len(df) and float(df.iloc[i+1]['whale'])>=0.35: continue
        rsi=float(row['rsi'])
        if np.isnan(rsi) or rsi>=25: continue
        if row['ts'].weekday()==3: continue
        if row['ts'].hour in BLOCK_HOURS: continue
        ps=max(0,i-96);pb=float(df.iloc[ps]['close']);ep=float(row['close'])
        if (ep-pb)/pb*100>=0: continue
        if i+1>=len(df): continue
        if float(df.iloc[i+1]['close'])<=float(df.iloc[i+1]['open']): continue
        
        tp_p=ep*(1+3.5/100);sl_p=ep*(1-1.5/100);pl_p=ep+(tp_p-ep)*(30/100)
        
        # 15m
        pt=False;pk=ep;tr=0;pn=None;ex='EOD'
        for k in range(i+1,len(df)):
            cur=float(df.iloc[k]['close']);h=(k-i)*0.25
            if h>MH: pn=round((cur-ep)/ep*100-COMM,4);ex='TIME';break
            if cur>=tp_p: pn=round(3.5-COMM,4);ex='TP';break
            if cur<=sl_p: pn=round(-1.5-COMM,4);ex='SL';break
            if not pt and cur>=pl_p: pt=True;pk=cur;tr=cur*(1-0.1/100)
            if pt:
                if cur>pk: pk=cur;tr=cur*(1-0.1/100)
                if cur<=tr: pn=round((tr-ep)/ep*100-COMM,4);ex='TRAIL';break
        if pn is None: pn=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4)
        all_15m.append({'sym':sym,'dt':row['ts'],'pnl':pn,'exit':ex});ex_15m[ex]+=1
        
        # 30m
        pt=False;pk=ep;tr=0;pn=None;ex='EOD'
        k=i+2
        while k<len(df):
            cur=float(df.iloc[k]['close']);h=(k-i)*0.25
            if h>MH: pn=round((cur-ep)/ep*100-COMM,4);ex='TIME';break
            if cur>=tp_p: pn=round(3.5-COMM,4);ex='TP';break
            if cur<=sl_p: pn=round(-1.5-COMM,4);ex='SL';break
            if not pt and cur>=pl_p: pt=True;pk=cur;tr=cur*(1-0.1/100)
            if pt:
                if cur>pk: pk=cur;tr=cur*(1-0.1/100)
                if cur<=tr: pn=round((tr-ep)/ep*100-COMM,4);ex='TRAIL';break
            k+=2
        if pn is None: pn=round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4)
        all_30m.append({'sym':sym,'dt':row['ts'],'pnl':pn,'exit':ex});ex_30m[ex]+=1
    
    if (fi+1)%10==0: print(f'{fi+1}/{len(coin_files)} done', flush=True)

def report(name,t,ex):
    arr=np.array([x['pnl'] for x in t])
    wa=arr[arr>0];la=arr[arr<0];wc=len(wa);lc=len(la)
    aw=wa.mean() if wc>0 else 0;al=la.mean() if lc>0 else 0
    rr=aw/abs(al) if al!=0 else 0
    sh=arr.mean()/arr.std()*np.sqrt(365*24*2) if arr.std()>0 else 0
    tp=wa.sum();tl=la.sum()
    
    ts=sorted(t,key=lambda x:x['dt'])
    cap=1000.0;pk=1000.0;mdd=0.0;ac=[];sk=0;tk=0
    for x in ts:
        dt=x['dt']
        sa=[]
        for ed,c,p in ac:
            if dt>=ed: cap+=c+p
            else: sa.append((ed,c,p))
        ac=sa
        if len(ac)>=2: sk+=1;continue
        ps2=cap*0.50
        if cap<ps2: sk+=1;continue
        pa=ps2*x['pnl']/100;cap-=ps2
        ac.append((dt+timedelta(hours=MH),ps2,pa));tk+=1
        eq=cap+sum(pc+pd for _,pc,pd in ac)
        if eq>pk: pk=eq
        dd=(eq-pk)/pk*100
        if dd<mdd: mdd=dd
    for _,c,p in ac: cap+=c+p
    ann=(cap/1000)**(1/5)-1 if cap>0 else -1
    
    tpc=ex.get('TP',0);slc=ex.get('SL',0)
    trc=ex.get('TRAIL',0);tmc=ex.get('TIME',0)+ex.get('EOD',0)
    
    print(f'\n📅 2021-07-01 → 2026-07-16')
    print(f'📊 ~{len(coin_files)} عملة — ⏱️ {name}')
    print(f'🔍 Look-ahead bias: NONE (close-only)')
    print(f'\n📋 عدد الصفقات: {len(arr):,}')
    print(f'🟢 صفقات رابحة: {wc:,} | 🔴 صفقات خاسرة: {lc:,}')
    print(f'📈 Win Rate: {wc/len(arr)*100:.1f}%')
    print(f'💵 إجمالي الربح: +{tp:,.1f}%')
    print(f'💸 إجمالي الخسارة: {tl:,.1f}%')
    print(f'💰 صافي: {tp+tl:+,.1f}%')
    print(f'🟢 متوسط الربح: {aw:+.2f}%')
    print(f'🔴 متوسط الخسارة: {al:+.2f}%')
    print(f'📊 R:R: {rr:.1f}x')
    print(f'📊 شارپ: {sh:.1f}')
    print(f'📉 أقصى انخفاض: {mdd:.2f}%')
    print(f'\n🏦 المحفظة: $1,000 → ${cap:,.0f} (+{cap/10-100:,.0f}%)')
    print(f'📈 عائد سنوي: {ann*100:+.1f}%')
    print(f'\n✅ منفذة: {tk:,} | ⏭️ متخطية: {sk:,}')
    print(f'🎯 TP: {tpc} | 🛑 SL: {slc} | 🐌 TRAIL: {trc} | ⏱️ TIME: {tmc}')

report('خروج 15m (حالي)',all_15m,ex_15m)
report('خروج 30m (جديد)',all_30m,ex_30m)
