import json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}
BLACKLIST = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','0G','ROBO','PYTH','ANKR'}
PRE_DROP_PCT = -2.0; PRE_DROP_CANDLES = 4

def compute_indicators(df):
    LB=30
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
    delta=df['close'].diff(); gain=delta.where(delta>0,0).rolling(14).mean()
    loss=(-delta.where(delta<0,0)).rolling(14).mean()
    df['rsi']=100-(100/(1+gain/loss.replace(0,np.nan)))
    return df

def sim_exit(df, i, ep):
    tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
    pl_p=ep+(tp_p-ep)*(PL/100); pl_trig=False; peak=ep; trail_p=0
    for k in range(i+1,len(df)):
        cur=float(df.iloc[k]['close']); h=(k-i)*0.25
        if h>MH: return round((cur-ep)/ep*100-COMM,4),'TIME'
        if cur>=tp_p: return round(TP-COMM,4),'TP'
        if cur<=sl_p: return round(-SL-COMM,4),'SL'
        if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
        if pl_trig:
            if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
            if cur<=trail_p: return round((trail_p-ep)/ep*100-COMM,4),'TRAIL'
    return round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4),'EOD'

def run_backtest(mode):
    """mode: 'base','combo'"""
    trades=[]; done=0; skipped_bl=0; skipped_sym=0
    files=sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    valid_files=[f for f in files if os.path.exists(f'{CACHE_DIR}/{f}')]
    
    for fname in valid_files:
        sym=fname.replace('_15m.json','')
        if sym in BLACKLIST:
            skipped_bl+=1; continue
        with open(f'{CACHE_DIR}/{fname}') as f: data=json.load(f)
        df=pd.DataFrame(data,columns=['ts','open','high','low','close','volume'])
        df['ts']=pd.to_datetime(df['ts'],unit='ms'); df=df.sort_values('ts').reset_index(drop=True)
        if len(df)<500:
            skipped_sym+=1; continue
        df=compute_indicators(df)
        for i in range(50,len(df)-10):
            row=df.iloc[i]
            if not row['entry']: continue
            if float(row['whale'])<WHALE_MIN: continue
            if i+1<len(df) and float(df.iloc[i+1]['whale'])>=0.35: continue
            rsi=float(row['rsi'])
            if np.isnan(rsi) or rsi>=25: continue
            if row['ts'].weekday()==3: continue
            if row['ts'].hour in BLOCK_HOURS: continue
            ps=max(0,i-96); pb=float(df.iloc[ps]['close']); ep=float(row['close'])
            if (ep-pb)/pb*100>=0: continue
            if mode=='combo':
                # Filter 2
                ps2=max(0,i-4); pp=float(df.iloc[ps2]['close'])
                if (ep-pp)/pp*100>-2.0: continue
                # Filter 3
                if i+1>=len(df): continue
                if float(df.iloc[i+1]['close'])<=float(df.iloc[i+1]['open']): continue
            pnl,ex=sim_exit(df,i,ep)
            trades.append({'sym':sym,'dt':str(row['ts']),'pnl':pnl,'exit':ex})
        done+=1
        if done%20==0: print(f'  [{mode}] {done} عملة | {len(trades)} صفقة', flush=True)
    return trades

def portfolio_stats(trades, label):
    if not trades: return {}
    nets=[t['pnl'] for t in trades]; wins=sum(1 for n in nets if n>0); exits=defaultdict(int)
    for t in trades: exits[t['exit']]+=1
    st=sorted(trades,key=lambda x:x['dt'])
    capital=1000.0; peak=1000.0; max_dd=0.0; active=[]; skipped=0; taken=0; exec_t=[]
    for t in st:
        dt=pd.to_datetime(t['dt']); still=[]
        for ed,cst,pnl in active:
            if dt>=ed: capital+=cst+pnl
            else: still.append((ed,cst,pnl))
        active=still
        if len(active)>=2: skipped+=1; continue
        ps=capital*0.50
        if capital<ps: skipped+=1; continue
        pa=ps*t['pnl']/100; capital-=ps
        active.append((dt+timedelta(hours=MH),ps,pa)); taken+=1; exec_t.append(t)
        eq=capital+sum(pc+pd for _,pc,pd in active)
        if eq>peak: peak=eq
        d=(eq-peak)/peak*100
        if d<max_dd: max_dd=d
    for _,cst,pnl in active: capital+=cst+pnl
    en=[t['pnl'] for t in exec_t]; ew=sum(1 for n in en if n>0)
    aw=sum(n for n in en if n>0)/max(1,sum(1 for n in en if n>0))
    al=sum(n for n in en if n<0)/max(1,sum(1 for n in en if n<0))
    return {'label':label,'signals':len(trades),'exec':taken,'skip':skipped,
            'wr':ew/taken*100 if taken else 0,'aw':aw,'al':al,'net':sum(en),
            'ev':sum(en)/taken if taken else 0,'pf':capital,'pf_ret':(capital/1000-1)*100,
            'dd':max_dd,'ann':((capital/1000)**(1/5)-1)*100,'exits':dict(exits)}

import sys
mode=sys.argv[1] if len(sys.argv)>1 else 'base'
print(f'🐋 وضع: {mode} | حلال فقط | 5 سنوات', flush=True)
trades=run_backtest(mode)
stats=portfolio_stats(trades,mode)
print(f'\n✅ [{mode}] {stats["signals"]} إشارة | {stats["exec"]} منفذة | WR {stats["wr"]:.1f}% | عائد {stats["pf_ret"]:.1f}% | DD {stats["dd"]:.2f}%', flush=True)

with open(f'/tmp/bt_{mode}.json','w') as f: json.dump(trades,f)
print(f'💾 /tmp/bt_{mode}.json', flush=True)
