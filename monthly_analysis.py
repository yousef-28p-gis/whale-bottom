#!/usr/bin/env python3
"""تحليل شهري للاستراتيجية"""
import json, numpy as np, pandas as pd, os

COMM=0.20; DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

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

with open('/data/trading28/config/shariah_coins.json') as f: shariah=json.load(f)
COINS=[c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]

all_trades=[]
for coin in COINS:
    fpath=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df=comp(df); ca=df['close'].values; ta=df['ts'].values
    idxs=sigs(df)
    if len(idxs)==0: del df; continue
    mb=80; tpr=1.013; slr=0.995; trr=0.9998
    active=[]; sm=dict(zip(idxs,ca[idxs]))
    for i in range(len(df)):
        cur=ca[i]
        if i in sm:
            ts_ns = int(pd.Timestamp(ta[i], unit='ms').value)
            m = pd.Timestamp(ta[i], unit='ms').month
            active.append({'s':coin,'e':sm[i],'tp':sm[i]*tpr,'sl':sm[i]*slr,
                'pok':False,'pk':sm[i],'tr':sm[i],'ei':i,'en':ts_ns,'month':m})
        for j in range(len(active)-1,-1,-1):
            p=active[j]; e=p['e']; bh=i-p['ei']
            ts_ns = int(pd.Timestamp(ta[i], unit='ms').value)
            if bh>=mb:
                p['pnl']=round((cur/e-1)*100-COMM,4); p['xt']='TIME'; p['xn']=ts_ns; all_trades.append(p); del active[j]
            elif cur>=p['tp']:
                p['pnl']=round(1.3-COMM,4); p['xt']='TP'; p['xn']=ts_ns; all_trades.append(p); del active[j]
            elif cur<=p['sl']:
                p['pnl']=round(-0.5-COMM,4); p['xt']='SL'; p['xn']=ts_ns; all_trades.append(p); del active[j]
            elif p['pok']:
                if cur>p['pk']: p['pk']=cur; p['tr']=cur*trr
                if cur<=p['tr']:
                    p['pnl']=round((p['tr']/e-1)*100-COMM,4); p['xt']='TRAIL'; p['xn']=ts_ns; all_trades.append(p); del active[j]
            else:
                pl_p=e+(p['tp']-e)*0.12
                if cur>=pl_p: p['pok']=True; p['pk']=cur; p['tr']=cur*trr
    del df

all_trades.sort(key=lambda t:t['en'])

# Simulation with monthly tracking
slots=[None,None]; monthly={}; eq=1000.0
for t in all_trades:
    en=t['en']; pnl_pct=t['pnl']; m=t.get('month',0)
    for s in range(2):
        if slots[s] is not None:
            sex,spnl,smonth=slots[s]
            if sex<=en:
                eq+=eq*0.5*(spnl/100); slots[s]=None
    
    mn = monthly.get(m, {'wins':0, 'losses':0, 'pnl_sum':0.0, 'trades':0})
    
    free=-1
    for s in range(2):
        if slots[s] is None: free=s; break
    if free==-1: continue
    
    mn['trades'] += 1
    mn['pnl_sum'] += pnl_pct
    if pnl_pct > 0: mn['wins'] += 1
    else: mn['losses'] += 1
    monthly[m] = mn
    
    slots[free] = (t.get('xn',en), pnl_pct, m)

months_ar={3:'مارس',4:'أبريل',5:'مايو',6:'يونيو',7:'يوليو'}
print('=== تحليل شهري — TP1.3% SL0.5% TRAIL0.02% ===')
hdr = '  {:10} {:>6} {:>6} {:>6} {:>7} {:>9} {:>10}'.format('شهر','صفقات','ربح','خسارة','WR','صافي%','ربح/500$')
print(hdr)
print('  ' + '-'*10 + ' ' + '-'*6 + ' ' + '-'*6 + ' ' + '-'*6 + ' ' + '-'*7 + ' ' + '-'*9 + ' ' + '-'*10)
total_t=0; total_w=0; total_pnl=0.0
for m in sorted(monthly.keys()):
    d=monthly[m]; wr=d['wins']/d['trades']*100; pnl_dollar=d['pnl_sum']/100*500
    row = '  {:10} {:>6} {:>6} {:>6} {:>6.1f}% {:>+8.1f}% ${:>+9,.0f}'.format(
        months_ar.get(m,str(m)), d['trades'], d['wins'], d['losses'], wr, d['pnl_sum'], pnl_dollar)
    print(row)
    total_t+=d['trades']; total_w+=d['wins']; total_pnl+=d['pnl_sum']

print('  ' + '-'*10 + ' ' + '-'*6 + ' ' + '-'*6 + ' ' + '-'*6 + ' ' + '-'*7 + ' ' + '-'*9 + ' ' + '-'*10)
pnl_dollar_total = total_pnl/100*500
row = '  {:10} {:>6} {:>6} {:>6} {:>6.1f}% {:>+8.1f}% ${:>+9,.0f}'.format(
    'المجموع', total_t, total_w, total_t-total_w, total_w/total_t*100, total_pnl, pnl_dollar_total)
print(row)
