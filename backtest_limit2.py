#!/usr/bin/env python3
"""أوامر حد — معالجة عملة عملة"""
import json, numpy as np, pandas as pd, os, time, gc

COMM=0.20; TF_MIN=3; CAPITAL=1000; MAX_POS=2
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

OFFSETS=[0.0,0.05,0.10,0.15,0.20,0.30]

def comp(df):
    df['low_lc']=df['low'].rolling(2).min()
    df['low_sm']=df['low_lc'].rolling(3).min()
    df['low_hi']=df['low_sm'].rolling(5).min()
    df['low_raw']=df['low_hi'].rolling(7).min()
    w=(df['low'].values-df['low_raw'].values)/np.where(df['low_raw'].values!=0,df['low_raw'].values,np.nan)*100
    df['whale']=np.clip(w,0,None)
    vm=df['volume'].rolling(20).mean().values
    df['spike']=df['volume'].values/np.where(vm!=0,vm,np.nan)
    d=df['close'].diff().values
    g=pd.Series(np.where(d>0,d,0)).rolling(14).mean().values
    l=pd.Series(np.where(d<0,-d,0)).rolling(14).mean().values
    df['rsi']=100-100/(1+g/np.where(l!=0,l,np.nan))
    return df

def sigs(df):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    wh=df['whale'].values; sp=df['spike'].values; rs=df['rsi'].values
    mask=(wh>=0.10)&(sp>=1.5)&(rs<35)&~np.isnan(wh)&~np.isnan(sp)&~np.isnan(rs)
    mask[:50]=False
    ng=np.zeros(n,dtype=bool); ng[:-1]=df['close'].values[1:]>df['open'].values[1:]; mask&=ng
    return np.where(mask)[0]

def sim_limit(df, sig_idxs, offset):
    ca=df['close'].values; la=df['low'].values; ha=df['high'].values
    ts=df['ts'].values.astype('datetime64[ns]').astype('int64')
    n=len(df); mb=80; tpr=1.013; slr=0.995; trr=0.9998
    trades=[]; missed=0; filled=0; active=[]
    for i in range(n):
        cur=ca[i]
        if i in sig_idxs:
            sp=cur; lp=sp*(1-offset/100)
            active.append({'ty':'p','sp':sp,'lp':lp,'tp':lp*tpr,'sl':lp*slr,'pok':False,'pk':lp,'tr':lp,'ei':i,'en':int(ts[i]),'bw':0})
        for j in range(len(active)-1,-1,-1):
            p=active[j]
            if p['ty']=='a':
                e=p['lp']; bh=i-p['ei']
                if bh>=mb:
                    p['pnl']=round((cur/e-1)*100-COMM,4); p['xt']='TIME'; trades.append(p); del active[j]; continue
                if ha[i]>=p['tp']:
                    p['pnl']=round(1.3-COMM,4); p['xt']='TP'; trades.append(p); del active[j]; continue
                if la[i]<=p['sl']:
                    p['pnl']=round(-0.5-COMM,4); p['xt']='SL'; trades.append(p); del active[j]; continue
                if p['pok']:
                    if ha[i]>p['pk']: p['pk']=ha[i]; p['tr']=p['pk']*trr
                    if la[i]<=p['tr']:
                        trp=round((p['tr']/e-1)*100-COMM,4); p['pnl']=trp; p['xt']='TRAIL'; trades.append(p); del active[j]
                else:
                    plp=e+(p['tp']-e)*0.12
                    if ha[i]>=plp: p['pok']=True; p['pk']=ha[i]; p['tr']=p['pk']*trr
            else:
                p['bw']+=1
                if la[i]<=p['lp']:
                    p['ty']='a'; p['ei']=i; p['en']=int(ts[i])
                    p['tp']=p['lp']*tpr; p['sl']=p['lp']*slr; p['pok']=False; p['pk']=p['lp']; p['tr']=p['lp']
                    filled+=1; continue
                if ha[i]>p['sp']*1.01 or p['bw']>20:
                    missed+=1; del active[j]
    return trades, missed, filled

with open('/data/trading28/config/shariah_coins.json') as f: shariah=json.load(f)
COINS=[c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]

all_trades_by_offset={o:[] for o in OFFSETS}
total_missed={o:0 for o in OFFSETS}
total_filled={o:0 for o in OFFSETS}
total_signals=0; t0=time.time()

for ci,coin in enumerate(COINS):
    fpath=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw
    df=comp(df); idxs=sigs(df); total_signals+=len(idxs)
    if len(idxs)==0: del df; continue
    sig_set=set(idxs)
    for offset in OFFSETS:
        trades,missed,filled=sim_limit(df,sig_set,offset)
        all_trades_by_offset[offset].extend(trades)
        total_missed[offset]+=missed
        total_filled[offset]+=filled
    del df; gc.collect()
    if (ci+1)%30==0: print('  ...', ci+1, '/', len(COINS), '|', int(time.time()-t0), 's', flush=True)

print('Done:', int(time.time()-t0), 's\n', flush=True)

# Simulation
sep = '=' * 90
print(sep)
print('Limit vs Market Orders | TP=1.3% SL=0.5% | MAX_POS=2 50%')
print(sep)
hdr = '  {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>9} {:>6}'.format(
    'Limit', 'Signals', 'Filled', 'Missed', '%Exec', 'WR', 'R:R', 'Win%', 'Loss%', 'Fixed$', 'DD')
print(hdr)
print('  ' + '-'*6 + ' ' + '-'*7 + ' ' + '-'*7 + ' ' + '-'*7 + ' ' + '-'*7 + ' ' + '-'*7 + ' ' + '-'*6 + ' ' + '-'*6 + ' ' + '-'*6 + ' ' + '-'*9 + ' ' + '-'*6)

for offset in OFFSETS:
    label = 'Market' if offset==0 else '{:.2f}%'.format(offset)
    trades=all_trades_by_offset[offset]
    trades.sort(key=lambda t:t.get('en',0))
    
    eq=float(CAPITAL); peak=float(CAPITAL); mdd=0.0
    slots=[None]*MAX_POS; epnls=[]; sk=0
    for t in trades:
        en=t.get('en',0)
        if 'pnl' not in t: continue
        pnl=t['pnl']
        for s in range(MAX_POS):
            if slots[s] is not None:
                sex,spnl=slots[s]
                if sex<=en:
                    eq+=eq*0.5*(spnl/100); slots[s]=None
                    if eq>peak: peak=eq
                    if eq<peak: mdd=min(mdd,(eq-peak)/peak*100)
        free=-1
        for s in range(MAX_POS): 
            if slots[s] is None: free=s; break
        if free==-1: sk+=1; continue
        epnls.append(pnl); slots[free]=(en,pnl)
    for s in range(MAX_POS):
        if slots[s] is not None: eq+=eq*0.5*(slots[s][1]/100)
    
    wins=sum(1 for p in epnls if p>0); losses=len(epnls)-wins
    wr=wins/len(epnls)*100 if epnls else 0
    aw=np.mean([p for p in epnls if p>0]) if wins else 0
    al=np.mean([p for p in epnls if p<=0]) if losses else 0
    rr=aw/abs(al) if al!=0 else 0
    fp=sum(p/100*500 for p in epnls)
    er=total_filled[offset]/total_signals*100 if total_signals else 0
    
    row = '  {:>6} {:>7,} {:>7,} {:>7,} {:>6.1f}% {:>6.1f}% {:>5.2f}x {:>+5.2f}% {:>+5.2f}% ${:>8,.0f} {:>5.1f}%'.format(
        label, total_signals, total_filled[offset], total_missed[offset], er, wr, rr, aw, al, 1000+fp, mdd)
    print(row)

print('\n  Market = close-only | Limit = High/Low fill')
