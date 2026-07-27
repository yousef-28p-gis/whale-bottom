"""اختبار TRAIL=0.15 + PL=40 مع فلتر 3"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}
BLACKLIST = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','0G','ROBO','PYTH','ANKR'}

def indicators(df):
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
    df['e']=(df['spike']&(df['wf']>df['ws'])&(df['str']>STR)&(df['volume']>df['vma']*1.0))
    d=df['close'].diff(); g=d.where(d>0,0).rolling(14).mean(); l=(-d.where(d<0,0)).rolling(14).mean()
    df['rsi']=100-(100/(1+g/l.replace(0,np.nan)))
    return df

def sim(df,i,ep,PL,TRAIL):
    tp=ep*(1+TP/100); sl=ep*(1-SL/100); pl_p=ep+(tp-ep)*(PL/100)
    pt=False; pk=ep; tr=0
    for k in range(i+1,len(df)):
        c=float(df.iloc[k]['close']); h=(k-i)*0.25
        if h>MH: return round((c-ep)/ep*100-COMM,4),'TIME'
        if c>=tp: return round(TP-COMM,4),'TP'
        if c<=sl: return round(-SL-COMM,4),'SL'
        if not pt and c>=pl_p: pt=True; pk=c; tr=c*(1-TRAIL/100)
        if pt:
            if c>pk: pk=c; tr=c*(1-TRAIL/100)
            if c<=tr: return round((tr-ep)/ep*100-COMM,4),'TRAIL'
    return round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4),'EOD'

def run(PL,TRAIL,label,max_coins=60):
    trades=[]; done=0
    files=sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    valid=[f for f in files if os.path.exists(f'{CACHE_DIR}/{f}')][:max_coins]
    for fname in valid:
        sym=fname.replace('_15m.json','')
        if sym in BLACKLIST: continue
        with open(f'{CACHE_DIR}/{fname}') as f: data=json.load(f)
        df=pd.DataFrame(data,columns=['ts','open','high','low','close','volume'])
        df['ts']=pd.to_datetime(df['ts'],unit='ms'); df=df.sort_values('ts').reset_index(drop=True)
        if len(df)<500: continue
        df=indicators(df)
        for i in range(50,len(df)-10):
            r=df.iloc[i]
            if not r['e']: continue
            if float(r['whale'])<WHALE_MIN: continue
            if i+1<len(df) and float(df.iloc[i+1]['whale'])>=0.35: continue
            rsi=float(r['rsi'])
            if np.isnan(rsi) or rsi>=25: continue
            if r['ts'].weekday()==3: continue
            if r['ts'].hour in BLOCK_HOURS: continue
            ps=max(0,i-96); pb=float(df.iloc[ps]['close']); ep=float(r['close'])
            if (ep-pb)/pb*100>=0: continue
            if i+1>=len(df): continue
            if float(df.iloc[i+1]['close'])<=float(df.iloc[i+1]['open']): continue
            pnl,ex=sim(df,i,ep,PL,TRAIL)
            trades.append({'sym':sym,'dt':str(r['ts']),'pnl':pnl,'exit':ex})
        done+=1
        if done%20==0: print(f'  [{label}] {done} | {len(trades)}ت', flush=True)
    return trades

def stats(tr):
    nets=[t['pnl'] for t in tr]
    st=sorted(tr,key=lambda x:x['dt'])
    cap=1000.0; peak=1000.0; dd=0.0; act=[]; skip=0; taken=0; et=[]
    for t in st:
        dt=pd.to_datetime(t['dt']); still=[]
        for ed,cst,pnl in act:
            if dt>=ed: cap+=cst+pnl
            else: still.append((ed,cst,pnl))
        act=still
        if len(act)>=2: skip+=1; continue
        ps=cap*0.50
        if cap<ps: skip+=1; continue
        pa=ps*t['pnl']/100; cap-=ps
        act.append((dt+timedelta(hours=MH),ps,pa)); taken+=1; et.append(t)
        eq=cap+sum(pc+pd for _,pc,pd in act)
        if eq>peak: peak=eq
        d=(eq-peak)/peak*100
        if d<dd: dd=d
    for _,cst,pnl in act: cap+=cst+pnl
    en=[t['pnl'] for t in et]; ew_t=sum(1 for n in en if n>0)
    aw=sum(n for n in en if n>0)/max(1,sum(1 for n in en if n>0))
    al=sum(n for n in en if n<0)/max(1,sum(1 for n in en if n<0))
    ex=defaultdict(int)
    for t in et: ex[t['exit']]+=1
    return {'signals':len(tr),'exec':taken,'wr':ew_t/taken*100 if taken else 0,
            'aw':aw,'al':al,'rr':abs(aw/al) if al!=0 else 0,
            'pf':cap,'pf_ret':(cap/1000-1)*100,'dd':dd,'exits':dict(ex)}

# Run all configs
configs = [
    (30, 0.10, 'OLD: PL30+TR0.10'),
    (30, 0.15, 'PL30+TR0.15'),
    (40, 0.10, 'PL40+TR0.10'),
    (40, 0.15, 'PL40+TR0.15'),
    (50, 0.15, 'PL50+TR0.15'),
]

results = {}
for PL,TR,label in configs:
    print(f'\n🐋 {label}...', flush=True)
    trades = run(PL, TR, label, 60)
    s = stats(trades)
    results[label] = s
    print(f'  ✅ WR {s["wr"]:.1f}% | R:R {s["rr"]:.2f}x | عائد {s["pf_ret"]:.1f}% | DD {s["dd"]:.2f}%', flush=True)

print('\n' + '='*65)
print(f'{"التكوين":<20} {"WR%":>7} {"متوسط ربح":>9} {"متوسط خسارة":>9} {"R:R":>6} {"عائد%":>8} {"سحب%":>7}')
print('-'*65)
for label in [c[2] for c in configs]:
    s = results[label]
    print(f'{label:<20} {s["wr"]:>7.1f} {s["aw"]:>9.2f} {s["al"]:>9.2f} {s["rr"]:>6.2f} {s["pf_ret"]:>8.1f} {s["dd"]:>7.2f}')

best_rr = max(results.items(), key=lambda x: x[1]['rr'])
best_pf = max(results.items(), key=lambda x: x[1]['pf_ret'])
print(f'\n🏆 أفضل R:R: {best_rr[0]} — R:R {best_rr[1]["rr"]:.2f}x | WR {best_rr[1]["wr"]:.1f}%')
print(f'🏆 أفضل عائد: {best_pf[0]} — عائد {best_pf[1]["pf_ret"]:.1f}% | WR {best_pf[1]["wr"]:.1f}%')
print('='*65, flush=True)
