#!/usr/bin/env python3
"""6-month backtest — memory efficient version"""
import json, os, numpy as np, pandas as pd
from datetime import datetime
from collections import defaultdict
import gc

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'
STR=50; WHALE_MIN=0.35; MIN_VOL=200000; COMM=0.20
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6

STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
BLOCKED={'SUPER','ORCA','VANA','W','DOGS','MET','XLM','BB','COS','LUNA','S'}

with open(SIGNALS_FILE) as f: raw=json.load(f)
signals=[]
for s in raw:
    if s['symbol'] in STABLES or s['symbol'] in BLOCKED: continue
    if s.get('direction','LONG')!='LONG': continue
    if s.get('volume_usdt',0)<MIN_VOL: continue
    dt=datetime.fromisoformat(s['dt'])
    if dt.month not in range(1,7) or dt.year!=2026: continue
    signals.append({'symbol':s['symbol'],'dt':dt,'month':dt.strftime('%Y-%m')})

by_pair=defaultdict(list)
for sig in signals: by_pair[(sig['symbol'],sig['month'])].append(sig)

print(f'Total signals: {len(signals)}, pair-months: {len(by_pair)}', flush=True)

def load_and_process(sym, mon):
    fpath=f'{CACHE}/{sym}_{mon}.json'
    if not os.path.exists(fpath): return None
    try:
        with open(fpath) as f: data=json.load(f)
    except: return None
    df=pd.DataFrame(data); df['ts']=pd.to_datetime(df['ts'],unit='ms')
    df=df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'}).sort_values('ts').reset_index(drop=True)
    df=df.copy(); LB=30
    df['lo']=df['low'].rolling(LB).min()
    df['lc']=abs(df['low']-df['low'].shift(1))/df['low']*100
    df['sm']=df['lc'].ewm(span=3,adjust=False).mean()
    df['hi']=df['sm'].rolling(LB).max()
    df['raw']=np.where(df['low']<=df['lo'],(df['sm']+df['hi']*2)/3,0)
    df['whale']=df['raw'].ewm(span=3,adjust=False).mean().fillna(0)
    df['spike']=(df['whale']>df['whale'].shift(1))&(df['whale'].shift(1)<=0.03)
    df['wf']=df['whale'].rolling(2).mean(); df['ws']=df['whale'].rolling(5).mean()
    df['wp']=df['whale'].rolling(50).max()
    df['str']=(df['whale']/df['wp'].replace(0,np.nan)*100).fillna(0)
    df['vma']=df['volume'].rolling(20).mean()
    df['entry']=(df['spike']&(df['wf']>df['ws'])&(df['str']>STR)&(df['volume']>df['vma']*1.0))
    return df

# Process pair by pair
trades_all=[]  # {dt, net, exit}
with_pump=[]   # same
pairs_done=0; missed=0

for (sym,mon),sigs in sorted(by_pair.items()):
    df_w=load_and_process(sym,mon)
    if df_w is None:
        missed+=1; continue
    
    for sig in sigs:
        df_w['td']=abs((df_w['ts']-sig['dt']).dt.total_seconds())
        n=df_w['td'].idxmin()
        fwd=df_w.iloc[n:].reset_index(drop=True)
        for j,row in fwd.iterrows():
            if j*0.25>24: break
            if row['entry'] and float(row['whale'])>=WHALE_MIN:
                whale_next=float(fwd.iloc[j+1]['whale']) if j+1<len(fwd) else 0
                ep=float(row['close'])
                global_idx=n+j
                ps=max(0,global_idx-96)
                pb=df_w['close'].iloc[ps]
                pump24=(ep-pb)/pb*100 if pb!=0 else 0
                
                # Simulate
                tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
                pl_p=ep+(tp_p-ep)*(PL/100)
                pl_trig=False; peak=ep; trail_p=0
                for k in range(j+1,len(fwd)):
                    cur=fwd.iloc[k]['close']; h=(k-j)*0.25
                    if h>MH: pnl=round((cur-ep)/ep*100-COMM,4); exit_='TIME'; break
                    if cur>=tp_p: pnl=round(TP-COMM,4); exit_='TP'; break
                    if cur<=sl_p: pnl=round(-SL-COMM,4); exit_='SL'; break
                    if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
                    if pl_trig:
                        if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                        if cur<=trail_p: pnl=round((trail_p-ep)/ep*100-COMM,4); exit_='TRAIL'; break
                else: pnl=round((fwd.iloc[-1]['close']-ep)/ep*100-COMM,4); exit_='EOD'
                
                t={'dt':sig['dt'],'net':pnl,'exit':exit_,'sym':sym}
                trades_all.append(t)
                if whale_next<0.35 and pump24<0:
                    with_pump.append(t)
                break
    
    pairs_done+=1
    if pairs_done%100==0:
        print(f'  {pairs_done}/{len(by_pair)} pairs, {len(trades_all)} trades', flush=True)
    del df_w; gc.collect()

print(f'\nDone! Pairs: {pairs_done}, Missed: {missed}', flush=True)

def portfolio(ee):
    ee=sorted(ee, key=lambda x:x['dt'])
    cap=1000.0; peak=1000.0; max_dd=0.0; active=None; skipped=0
    for t in ee:
        if active is not None and t['dt']<active[0]:
            skipped+=1; continue
        if active is not None:
            et,amt,ec=active; cap+=amt+ec
        pos=cap; pnl_amt=pos*t['net']/100
        active=(t['dt']+pd.Timedelta(hours=MH),pos,pnl_amt)
        eq=cap
        if eq>peak: peak=eq
        dd=(eq-peak)/peak*100
        if dd<max_dd: max_dd=dd
        cap-=pos
    if active: cap+=active[1]+active[2]
    return cap,max_dd,skipped

def report(label, trades):
    if not trades: return
    nets=[t['net'] for t in trades]
    wins=sum(1 for n in nets if n>0)
    wr=wins/len(trades)*100
    cap,dd,skipped=portfolio(trades)
    exits=defaultdict(int)
    for t in trades: exits[t['exit']]+=1
    print(f'\n{"="*60}')
    print(f'🏆 {label}')
    print(f'{"="*60}')
    print(f'صفقات: {len(trades)} | تخطي: {skipped}')
    print(f'رابحة: {wins} | خاسرة: {len(trades)-wins}')
    print(f'WR: {wr:.1f}% | صافي: {sum(nets):+.1f}%')
    print(f'🎯TP={exits.get("TP",0)} 🛑SL={exits.get("SL",0)} 🐌TRAIL={exits.get("TRAIL",0)} ⏰TIME={exits.get("TIME",0)+exits.get("EOD",0)}')
    print(f'💼 محفظة: ${cap:.0f} | سحب: {dd:.2f}%')
    
    # Monthly
    months_ar={1:'يناير',2:'فبراير',3:'مارس',4:'أبريل',5:'مايو',6:'يونيو'}
    for m in range(1,7):
        mon=f'2026-{m:02d}'
        mt=[t for t in trades if t['dt'].strftime('%Y-%m')==mon]
        if not mt: continue
        mw=sum(1 for t in mt if t['net']>0)
        mn=sum(t['net'] for t in mt)
        m_sl=sum(1 for t in mt if t['exit']=='SL')
        print(f'  {months_ar[m]}: {len(mt):>3} | WR {mw/len(mt)*100:.0f}% | صافي {mn:+.1f}% | SL={m_sl}')

report('ALL (بدون فلاتر)', trades_all)
report('⭐ شمعة وحدة + Pump24 سالب', with_pump)
