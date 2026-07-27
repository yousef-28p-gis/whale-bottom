#!/usr/bin/env python3
"""اختبار MAX_POS متنوع — TP1.3_SL0.5_TR2_لا_تباعد"""
import json, numpy as np, pandas as pd, os, time, gc

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# TP1.3%, SL0.5%, PL12%, TRAIL0.02%, MH4h, WHALE≥0.10, RSI<35, تأكيد, بدون تباعد
tp,sl,pl,trail,mh,wm,rm,cf = 1.3, 0.5, 12, 0.02, 4, 0.10, 35, True
spacing = False

MAX_POS_VALUES = [2, 3, 4, 5, 6, 8, 10]

def comp(df):
    df['low_lc']=df['low'].rolling(2).min()
    df['low_sm']=df['low_lc'].rolling(3).min()
    df['low_hi']=df['low_sm'].rolling(5).min()
    df['low_raw']=df['low_hi'].rolling(7).min()
    w=(df['low'].values-df['low_raw'].values)/np.where(df['low_raw'].values!=0,df['low_raw'].values,np.nan)*100
    df['whale']=np.clip(w,0,None)
    vm=df['volume'].rolling(20).mean().values
    df['spike']=df['volume'].values/np.where(vm!=0,vm,np.nan)
    delta=df['close'].diff().values
    gain=pd.Series(np.where(delta>0,delta,0)).rolling(14).mean().values
    loss=pd.Series(np.where(delta<0,-delta,0)).rolling(14).mean().values
    df['rsi']=100-100/(1+gain/np.where(loss!=0,loss,np.nan))
    return df

def sigs(df):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    wh=df['whale'].values; sp_=df['spike'].values; rs=df['rsi'].values
    mask=(wh>=wm)&(sp_>=1.5)&(rs<rm)&~np.isnan(wh)&~np.isnan(sp_)&~np.isnan(rs)
    mask[:50]=False
    if cf: ng=np.zeros(n,dtype=bool); ng[:-1]=df['close'].values[1:]>df['open'].values[1:]; mask&=ng
    return np.where(mask)[0]

print("⏳ جمع الصفقات...", flush=True)
with open('config/shariah_coins.json') as f: shariah=json.load(f)
COINS=[c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]

all_trades=[]
for coin in COINS:
    fpath=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df=comp(df)
    ca=df['close'].values; ta=df['ts'].values.astype('datetime64[ns]').astype('int64')
    idxs=sigs(df)
    if len(idxs)==0: del df; continue
    mb=int(mh*60/TF_MIN); tpr=1+tp/100; slr=1-sl/100; trr=1-trail/100
    active=[]; sm=dict(zip(idxs,ca[idxs]))
    for i in range(len(df)):
        cur=ca[i]
        if i in sm:
            active.append({'s':coin,'e':sm[i],'tp':sm[i]*tpr,'sl':sm[i]*slr,
                'pok':False,'pk':sm[i],'tr':sm[i],'ei':i,'en':int(ta[i])})
        for j in range(len(active)-1,-1,-1):
            p=active[j]; e=p['e']; bh=i-p['ei']
            if bh>=mb:
                p['pnl']=round((cur/e-1)*100-COMM,4); p['xt']='TIME'; p['xn']=int(ta[i]); all_trades.append(p); del active[j]
            elif cur>=p['tp']:
                p['pnl']=round(tp-COMM,4); p['xt']='TP'; p['xn']=int(ta[i]); all_trades.append(p); del active[j]
            elif cur<=p['sl']:
                p['pnl']=round(-sl-COMM,4); p['xt']='SL'; p['xn']=int(ta[i]); all_trades.append(p); del active[j]
            elif p['pok']:
                if cur>p['pk']: p['pk']=cur; p['tr']=cur*trr
                if cur<=p['tr']:
                    p['pnl']=round((p['tr']/e-1)*100-COMM,4); p['xt']='TRAIL'; p['xn']=int(ta[i]); all_trades.append(p); del active[j]
            else:
                pl_p=e+(p['tp']-e)*(pl/100)
                if cur>=pl_p: p['pok']=True; p['pk']=cur; p['tr']=cur*trr
    del df

all_trades.sort(key=lambda t:t['en'])
print(f"✅ {len(all_trades):,} صفقة\n", flush=True)

# ═══════════════ اختبار MAX_POS متنوع ═══════════════
print(f"{'='*85}")
print(f"📊 MAX_POS متنوع | TP={tp}% SL={sl}% TR={trail}% | بدون تباعد")
print(f"{'='*85}")
print(f"  {'MAX':>4} {'%ص':>4} {'✅نفذ':>7} {'⏭️تخطى':>7} {'%تنفيذ':>7} {'WR':>7} {'R:R':>6} {'🟢':>6} {'🔴':>6} {'ثابت\$':>10} {'تركيب\$':>11} {'DD':>6}")
print(f"  {'─'*4} {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*10} {'─'*11} {'─'*6}")

for mp in MAX_POS_VALUES:
    pos_pct = 100/mp
    equity=float(CAPITAL); peak=float(CAPITAL); max_dd=0.0
    slots=[None]*mp; executed=0; skipped=0
    executed_pnls=[]
    
    for t in all_trades:
        en=t['en']; xn=t['xn']; pnl_pct=t['pnl']
        for s in range(mp):
            if slots[s] is not None:
                sex,spnl=slots[s]
                if sex<=en:
                    pos_cap=equity*(pos_pct/100); pnl_d=pos_cap*(spnl/100); equity+=pnl_d; slots[s]=None
                    if equity>peak: peak=equity
                    dd=(equity-peak)/peak*100
                    if dd<max_dd: max_dd=dd
        free=-1
        for s in range(mp):
            if slots[s] is None: free=s; break
        if free==-1: skipped+=1; continue
        executed+=1; executed_pnls.append(pnl_pct)
        slots[free]=(xn,pnl_pct)
    
    for s in range(mp):
        if slots[s] is not None:
            sex,spnl=slots[s]; pos_cap=equity*(pos_pct/100); pnl_d=pos_cap*(spnl/100); equity+=pnl_d
    
    if equity>peak: peak=equity
    dd=(equity-peak)/peak*100
    if dd<max_dd: max_dd=dd
    
    wins=sum(1 for p in executed_pnls if p>0); losses=len(executed_pnls)-wins
    wr=wins/len(executed_pnls)*100 if executed_pnls else 0
    aw=np.mean([p for p in executed_pnls if p>0]) if wins else 0
    al=np.mean([p for p in executed_pnls if p<=0]) if losses else 0
    rr=aw/abs(al) if al!=0 else 0
    
    # Fixed PnL: each trade uses CAPITAL/mp (not current equity)
    fixed_pnl = sum(pnl/100*(CAPITAL/mp) for pnl in executed_pnls)
    
    exec_rate = executed/len(all_trades)*100
    
    print(f"  {mp:>4} {pos_pct:>4.0f}% {executed:>7,} {skipped:>7,} {exec_rate:>6.1f}% {wr:>6.1f}% {rr:>5.2f}x {aw:>+5.2f}% {al:>+5.2f}% ${1000+fixed_pnl:>9,.0f} ${equity:>10,.0f} {dd:>5.1f}%")
