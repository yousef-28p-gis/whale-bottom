import json, os, sys, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}
PRE_DROP_PCT = -2.0
PRE_DROP_CANDLES = 4

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

def one_trade(df, i, ep):
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

def run_and_save(mode, outfile):
    trades=[]; done=0; total=0
    files=[f for f in sorted(os.listdir(CACHE_DIR)) if f.endswith('.json')]
    valid_files = []
    for fname in files:
        fpath=f'{CACHE_DIR}/{fname}'
        if os.path.exists(fpath): valid_files.append(fname)
    total = len(valid_files)
    
    for fname in valid_files:
        sym=fname.replace('_15m.json','')
        with open(f'{CACHE_DIR}/{fname}') as f: data=json.load(f)
        df=pd.DataFrame(data,columns=['ts','open','high','low','close','volume'])
        df['ts']=pd.to_datetime(df['ts'],unit='ms'); df=df.sort_values('ts').reset_index(drop=True)
        if len(df)<500: continue
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
            if mode in ('f2','combo'):
                pre_start=max(0,i-PRE_DROP_CANDLES)
                pre_price=float(df.iloc[pre_start]['close'])
                if (ep-pre_price)/pre_price*100>PRE_DROP_PCT: continue
            if mode in ('f3','combo'):
                if i+1>=len(df): continue
                nc=df.iloc[i+1]
                if float(nc['close'])<=float(nc['open']): continue
            pnl,ex=one_trade(df,i,ep)
            trades.append({'sym':sym,'dt':str(row['ts']),'pnl':pnl,'exit':ex})
        done+=1
        if done%30==0: print(f'  [{mode}] {done}/{total} | {len(trades)} صفقة', flush=True)
    
    with open(outfile,'w') as f: json.dump(trades,f)
    print(f'✅ [{mode}] تم: {len(trades)} صفقة ← {outfile}', flush=True)
    return trades

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv)>1 else 'base'
    outfile = sys.argv[2] if len(sys.argv)>2 else f'/tmp/trades_{mode}.json'
    print(f'🐋 وضع: {mode} | حفظ: {outfile}', flush=True)
    run_and_save(mode, outfile)
