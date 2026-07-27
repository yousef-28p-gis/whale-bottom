import json, os, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK = {1,3,6,12,0,4}; N=20  # عدد العملات

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

def sim(df,i,ep):
    tp=ep*(1+TP/100); sl=ep*(1-SL/100); pl=ep+(tp-ep)*(PL/100); pt=False; pk=ep; tr=0
    for k in range(i+1,len(df)):
        c=float(df.iloc[k]['close']); h=(k-i)*0.25
        if h>MH: return round((c-ep)/ep*100-COMM,4),'TIME'
        if c>=tp: return round(TP-COMM,4),'TP'
        if c<=sl: return round(-SL-COMM,4),'SL'
        if not pt and c>=pl: pt=True; pk=c; tr=c*(1-TRAIL/100)
        if pt:
            if c>pk: pk=c; tr=c*(1-TRAIL/100)
            if c<=tr: return round((tr-ep)/ep*100-COMM,4),'TRAIL'
    return round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4),'EOD'

def run(mode):
    trades=[]; done=0
    files=[f for f in sorted(os.listdir(CACHE_DIR)) if f.endswith('.json')]
    valid=[fn for fn in files if os.path.exists(f'{CACHE_DIR}/{fn}')][:N]
    for fname in valid:
        sym=fname.replace('_15m.json','')
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
            if r['ts'].hour in BLOCK: continue
            ps=max(0,i-96); pb=float(df.iloc[ps]['close']); ep=float(r['close'])
            if (ep-pb)/pb*100>=0: continue
            if mode in ('f2','combo'):
                ps2=max(0,i-4); pp=float(df.iloc[ps2]['close'])
                if (ep-pp)/pp*100>-2.0: continue
            if mode in ('f3','combo'):
                if i+1>=len(df): continue
                if float(df.iloc[i+1]['close'])<=float(df.iloc[i+1]['open']): continue
            pnl,ex=sim(df,i,ep)
            trades.append({'sym':sym,'dt':str(r['ts']),'pnl':pnl,'exit':ex})
        done+=1
        if done%10==0: print(f'  {mode} {done}/{len(valid)} | {len(trades)}ت', flush=True)
    return trades

def pf_stats(tr,label):
    if not tr: return {}
    nets=[t['pnl'] for t in tr]; wins=sum(1 for n in nets if n>0); exits=defaultdict(int)
    for t in tr: exits[t['exit']]+=1
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
    en=[t['pnl'] for t in et]; ew=sum(1 for n in en if n>0)
    aw=sum(n for n in en if n>0)/max(1,sum(1 for n in en if n>0))
    al=sum(n for n in en if n<0)/max(1,sum(1 for n in en if n<0))
    return {'label':label,'signals':len(tr),'exec':taken,'skip':skip,
            'wr':ew/taken*100 if taken else 0,'net':sum(en),'aw':aw,'al':al,
            'ev':sum(en)/taken if taken else 0,'pf':cap,'pf_ret':(cap/1000-1)*100,
            'dd':dd,'exits':dict(exits),'ann':((cap/1000)**(1/5)-1)*100}

R={}
for mode,label in [('base','بدون'),('f2','فلتر2'),('f3','فلتر3'),('combo','2+3')]:
    print(f'\n🐋 {label}...', flush=True)
    tr=run(mode); s=pf_stats(tr,label); R[mode]=s
    print(f'  ✅ {s.get("exec",0)}ت | WR {s.get("wr",0):.1f}% | عائد {s.get("pf_ret",0):.1f}% | DD {s.get("dd",0):.2f}%', flush=True)

print('\n'+'='*65)
print(f'📊 مقارنة الفلاتر — {N} عملة حلال — 5 سنوات')
print('='*65)
print(f'{"":<20} {"بدون":>10} {"فلتر2":>10} {"فلتر3":>10} {"2+3":>10}')
print('-'*55)
for k,d in [('signals','إشارات'),('exec','منفذة'),('wr','WR%'),('aw','AvgWin%'),('al','AvgLoss%'),('pf_ret','عائد%'),('dd','سحب%'),('ann','سنوي%')]:
    vs=[f'{R[m][k]:.1f}' if isinstance(R[m].get(k),float) else str(R[m].get(k,'-')) for m in ['base','f2','f3','combo']]
    print(f'{d:<20} {vs[0]:>10} {vs[1]:>10} {vs[2]:>10} {vs[3]:>10}')
bw=max(R.items(),key=lambda x:x[1].get('wr',0))
bp=max(R.items(),key=lambda x:x[1].get('pf_ret',0))
print(f'\n🏆 WR: {bw[1]["label"]} {bw[1]["wr"]:.1f}% | عائد {bw[1]["pf_ret"]:.1f}% | DD {bw[1]["dd"]:.2f}%')
print(f'🏆 عائد: {bp[1]["label"]} WR {bp[1]["wr"]:.1f}% | عائد {bp[1]["pf_ret"]:.1f}% | DD {bp[1]["dd"]:.2f}%')
print('='*65, flush=True)
