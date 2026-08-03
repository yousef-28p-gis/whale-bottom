#!/usr/bin/env python3
"""تدقيق DD — لماذا DD كبير مع SL صغير؟"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

COMM=0.002; CAP=1000; DATA='/data/trading28/data/whale_15m_1y'

def load(sym):
    with open(os.path.join(DATA, f'{sym}.json')) as f: d=json.load(f)
    return {'c':np.array(d['c'],float),'h':np.array(d['h'],float),
            'l':np.array(d['l'],float),'o':np.array(d['o'],float),
            'ts':pd.to_datetime(d['ts'],unit='ms')}

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def sim(le, c, h, l_, n, tp, sl):
    t=[]; eq=CAP; cv=[CAP]; pos=0; ep=0; cool=0; all_trades=[]
    for i in range(200,n):
        if pos:
            if h[i]>=ep*(1+tp/100):
                pnl=tp-COMM*100; t.append(pnl); eq*=(1+pnl/100); pos=0; cool=12
                all_trades.append({'pnl':pnl,'exit':'TP'})
            elif l_[i]<=ep*(1-sl/100):
                raw=(c[i]/ep-1)*100-COMM*100
                pnl=max(raw, -sl*1.5-COMM*100)
                t.append(pnl); eq*=(1+pnl/100); pos=0; cool=12
                all_trades.append({'pnl':pnl,'exit':'SL','raw':raw})
        if not pos and cool==0 and le[i]: pos=1; ep=c[i]
        if not pos and cool>0: cool-=1
        cv.append(eq)
    if pos:
        pnl=(c[-1]/ep-1)*100-COMM*100; t.append(pnl); eq*=(1+pnl/100)
        all_trades.append({'pnl':pnl,'exit':'OPEN'})
    return t,cv,eq,all_trades

# Test on KAITO
d=load('KAITO')
c=d['c']; h=d['h']; l_=d['l']; o=d['o']; n=len(c)

wp_sig=np.zeros(n,bool)  # simplified entry for audit
ssl_c=np.zeros(n,int)
sma_h=pd.Series(h).rolling(10).mean().values
sma_l=pd.Series(l_).rolling(10).mean().values
for i in range(10,n):
    if h[i-1]>sma_h[i-1]: ssl_c[i]=1
    else: ssl_c[i]=-1

ln=pd.Series(l_).shift(1).rolling(70).min().values
lc=np.zeros(n)
for i in range(1,n): lc[i]=abs(l_[i]-l_[i-1])/l_[i]*100
sc=pd.Series(lc).ewm(span=3,adjust=False).mean().values
hc=pd.Series(sc).rolling(70).max().values
sr=np.where(l_<=ln,(sc+hc*2)/3,0)
wp=pd.Series(sr).ewm(span=3,adjust=False).mean().values
wp_up=wp>np.roll(wp,1)
for i in range(200,n):
    if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0:
        wp_sig[i]=True

tr,cv,eq,at=sim(wp_sig,c,h,l_,n,5.0,2.5)
dd_series=pd.Series(cv).expanding().max()
dd_pct=(pd.Series(cv)-dd_series)/dd_series*100

# Find worst DD period
worst_dd_i=dd_pct.idxmin()
worst_dd=dd_pct.min()
dd_start=dd_series[:worst_dd_i].idxmax()

print(f'KAITO | {len(tr)} trades | WR: {len([t for t in at if t["pnl"]>0])/len(at)*100:.1f}% | DD: {worst_dd:.1f}%')
print(f'DD period: trade #{dd_series[:worst_dd_i].idxmax()} → #{worst_dd_i}')
print(f'\n📋 All trades:')
print(f'{"#":>3} {"Exit":>5} {"PnL":>8} {"Eq":>9} {"DD%":>7}')
print('-'*40)
eq_tmp=CAP; peak=CAP; dd_max=0
for i,trade in enumerate(at):
    eq_tmp*=(1+trade['pnl']/100)
    peak=max(peak,eq_tmp)
    dd_current=(eq_tmp-peak)/peak*100
    if dd_current<dd_max: dd_max=dd_current
    
    marker=''
    if dd_current<-5: marker='⚠️'
    if dd_current<-8: marker='🚩'
    if eq_tmp>peak*0.95 and dd_current==0: pass  # skip printing all good trades
    if abs(dd_current)>2 or i<5 or i>len(at)-5:
        print(f'{i+1:>3} {trade["exit"]:>5} {trade["pnl"]:>+7.2f}% ${eq_tmp:>8.1f} {dd_current:>+6.1f}% {marker}')

print(f'\n📊 تحليل DD:')
print(f'أقصى DD: {dd_max:.1f}%')
print(f'أكبر خسارة فردية: {min(t["pnl"] for t in at):+.2f}%')
print(f'عدد الخسائر المتتالية الأقصى: ...')

# Count consecutive losses
max_cons=0; curr_cons=0
for t in at:
    if t['pnl']<=0: curr_cons+=1; max_cons=max(max_cons,curr_cons)
    else: curr_cons=0
print(f'أقصى خسائر متتالية: {max_cons}')
if max_cons>0:
    max_loss_seq=max_cons* (abs(tr[0]) if tr else 2.9)
    print(f'≈ DD نظري من الخسائر المتتالية: -{max_loss_seq:.1f}%')

# Check: does compounding explain DD?
eq_after_losses=CAP
for i in range(max_cons):
    eq_after_losses*=(1-2.9/100)
print(f'المحفظة بعد {max_cons} خسارة متتالية (2.9%): ${eq_after_losses:.1f} = {(eq_after_losses/CAP-1)*100:.1f}%')

print('\n✅ Done')
