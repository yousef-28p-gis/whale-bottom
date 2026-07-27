"""اختبار تريل بالإغلاق بدل النسبة — فلتر 3"""
import json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
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

def sim_new_trail(df, i, ep):
    """تريل جديد: خروج إذا إغلاق الشمعة أقل من إغلاق الشمعة السابقة"""
    tp=ep*(1+TP/100); sl=ep*(1-SL/100)
    pl_p=ep+(tp-ep)*(PL/100)
    pl_trig=False; peak=ep; peak_idx=i
    
    for k in range(i+1, len(df)):
        cur=float(df.iloc[k]['close']); h=(k-i)*0.25
        if h>MH: return round((cur-ep)/ep*100-COMM,4),'TIME'
        if cur>=tp: return round(TP-COMM,4),'TP'
        if cur<=sl: return round(-SL-COMM,4),'SL'
        
        if not pl_trig and cur>=pl_p:
            pl_trig=True; peak=cur; peak_idx=k
            continue
        
        if pl_trig:
            if cur>peak:
                peak=cur; peak_idx=k
                continue
            # 🔥 الجديد: خروج إذا إغلاق < إغلاق الشمعة السابقة (تأكيد انعكاس)
            prev_close = float(df.iloc[k-1]['close'])
            if cur < prev_close:
                exit_price = cur
                pnl = round((exit_price-ep)/ep*100-COMM,4)
                return pnl, 'TRAIL'
    
    return round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4),'EOD'

def run(label, max_coins=40):
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
            # فلتر 3
            if i+1>=len(df): continue
            if float(df.iloc[i+1]['close'])<=float(df.iloc[i+1]['open']): continue
            pnl,ex=sim_new_trail(df,i,ep)
            trades.append({'sym':sym,'dt':str(r['ts']),'pnl':pnl,'exit':ex})
        done+=1
        if done%10==0: print(f'  [{label}] {done} | {len(trades)}ت', flush=True)
    return trades

def stats(tr):
    nets=[t['pnl'] for t in tr]; ew=sum(1 for n in nets if n>0)
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
            'pf':cap,'pf_ret':(cap/1000-1)*100,'dd':dd,
            'exits':dict(ex),'annual':((cap/1000)**(1/5)-1)*100}

# Run
print('🐋 تريل جديد: خروج عند إغلاق < إغلاق سابق', flush=True)
tr=run('new_trail', 60)
s=stats(tr)
print(f'\n✅ {s["signals"]} إشارة | {s["exec"]} منفذة | WR {s["wr"]:.1f}%', flush=True)
print(f'💼 $1000 → ${s["pf"]:.0f} (+{s["pf_ret"]:.1f}%) | DD {s["dd"]:.2f}%', flush=True)
print(f'🟢 AvgWin: +{s["aw"]:.2f}% | 🔴 AvgLoss: {s["al"]:.2f}% | R:R = {s["rr"]:.2f}x', flush=True)
tp=s['exits'].get('TP',0); sl=s['exits'].get('SL',0); tr_ex=s['exits'].get('TRAIL',0)
tm=s['exits'].get('TIME',0)+s['exits'].get('EOD',0)
print(f'📤 TP={tp} SL={sl} TRAIL={tr_ex} TIME={tm}', flush=True)

# Compare with old
print('\n📊 مقارنة (60 عملة):', flush=True)
print(f'  القديم (تريل 0.10%): WR 77.1% | R:R ~1.02x | عائد مرتفع', flush=True)
print(f'  الجديد (إغلاق<سابق): WR {s["wr"]:.1f}% | R:R = {s["rr"]:.2f}x | عائد {s["pf_ret"]:.1f}%', flush=True)
