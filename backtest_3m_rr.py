#!/usr/bin/env python3
"""🧪 تحسين R:R — C2 دخول + NoSL + وقت أقصر + SL طوارئ — عالمي"""
import json, numpy as np, pandas as pd, os, time, gc

COMM=0.20; TF_MIN=3; CAPITAL=1000
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

TESTS=[
    # Baseline
    ("B0_NoSL_TP2.5_TR08_6h_MP2",      2.5,99,30,0.08,6,2),
    # وقت أقصر
    ("T1_NoSL_TP2.5_TR08_4h_MP2",      2.5,99,30,0.08,4,2),
    ("T2_NoSL_TP2.5_TR08_3h_MP2",      2.5,99,30,0.08,3,2),
    ("T3_NoSL_TP2.0_TR05_4h_MP2",      2.0,99,30,0.05,4,2),
    ("T4_NoSL_TP2.0_TR05_3h_MP2",      2.0,99,30,0.05,3,2),
    # SL طوارئ واسع
    ("S1_SL3_TP2.5_TR08_6h_MP2",       2.5,3.0,30,0.08,6,2),
    ("S2_SL4_TP2.5_TR08_6h_MP2",       2.5,4.0,30,0.08,6,2),
    ("S3_SL3_TP2.5_TR08_4h_MP2",       2.5,3.0,30,0.08,4,2),
    # PL أبكر
    ("P1_PL20_TP2.5_TR08_6h_MP2",      2.5,99,20,0.08,6,2),
    ("P2_PL20_TP2.5_TR08_4h_MP2",      2.5,99,20,0.08,4,2),
    # دمج: وقت أقصر + PL أبكر
    ("M1_PL20_TP2.5_TR08_3h_MP2",      2.5,99,20,0.08,3,2),
    ("M2_PL20_TP2.5_TR05_3h_MP2",      2.5,99,20,0.05,3,2),
]

def compute_indicators(df):
    n=len(df); c=df['close'].values; h=df['high'].values; l=df['low'].values; v=df['volume'].values
    df['low_lc']=df['low'].rolling(2).min()
    df['low_sm']=df['low_lc'].rolling(3).min()
    df['low_hi']=df['low_sm'].rolling(5).min()
    df['low_raw']=df['low_hi'].rolling(7).min()
    w=(l-df['low_raw'].values)/np.where(df['low_raw'].values!=0,df['low_raw'].values,np.nan)*100
    df['whale']=np.clip(w,0,None)
    vm=df['volume'].rolling(20).mean().values
    df['spike']=v/np.where(vm!=0,vm,np.nan)
    d=c.copy(); d[1:]=c[1:]-c[:-1]; d[0]=0
    g=np.where(d>0,d,0); ls=np.where(d<0,-d,0)
    ag=pd.Series(g).rolling(14).mean().values; al=pd.Series(ls).rolling(14).mean().values
    rs=ag/np.where(al!=0,al,np.nan)
    df['rsi']=100-(100/(1+rs))
    return df

def find_signals(df):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    w=df['whale'].values; s=df['spike'].values; r=df['rsi'].values
    mask=(w>=0.25)&(s>=2.0)&(r<25)&~np.isnan(w)&~np.isnan(s)&~np.isnan(r)
    mask[:50]=False
    hp=np.zeros(n,dtype=bool)
    for sh in[1,2,3]:
        shf=np.zeros(n,dtype=bool); shf[sh:]=mask[:-sh]; hp|=shf
    mask&=~hp
    return np.where(mask)[0]

def compute_trade(close_arr,entry_idx,entry_price,tp,sl,pl,trail,max_h):
    n=len(close_arr); mb=int(max_h*60/TF_MIN)
    tpp=entry_price*(1+tp/100); tr=1-trail/100
    dsl=entry_price*(1-sl/100) if sl<90 else 0.0001
    pt=False; pk=entry_price; tlp=entry_price
    for i in range(entry_idx+1,n):
        cur=close_arr[i]; bh=i-entry_idx
        if bh>=mb:
            pnl=round((cur/entry_price-1)*100-COMM,4); return('TIME',pnl,i)
        if cur>=tpp:
            pnl=round((tpp/entry_price-1)*100-COMM,4); return('TP',pnl,i)
        if sl<90 and cur<=dsl:
            pnl=round((cur/entry_price-1)*100-COMM,4); return('SL',pnl,i)
        if pt:
            if cur>pk: pk=cur; tlp=cur*tr
            if cur<=tlp:
                pnl=round((tlp/entry_price-1)*100-COMM,4); return('TRAIL',pnl,i)
        else:
            plp=entry_price+(tpp-entry_price)*(pl/100)
            if cur>=plp: pt=True; pk=cur; tlp=cur*tr
    return('OPEN',0.0,n-1)

def global_sim(potential,max_pos):
    potential.sort(key=lambda x:x[0])
    active=[]; executed=[]; skipped=0
    for ets,coin,eidx,xts,xidx,pnl,etyp in potential:
        active=[a for a in active if a[0]>ets]
        if len(active)>=max_pos: skipped+=1; continue
        active.append((xts,ets,coin,pnl,etyp))
        executed.append((ets,coin,pnl,etyp,xts))
    return executed,skipped

def calc_portfolio(executed,max_pos):
    executed.sort(key=lambda x:x[0])
    eq=CAPITAL; peq=CAPITAL; mdd=0.0
    for _,_,pnl,_,_ in executed:
        pc=eq/max_pos; eq+=pc*(pnl/100)
        if eq>peq: peq=eq
        dd=(eq-peq)/peq*100
        if dd<mdd: mdd=dd
    return eq,mdd

print("⏳ تجهيز...",flush=True)
with open('/data/trading28/config/shariah_coins.json') as f: shariah=json.load(f)
COINS=[c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
print(f"📋 {len(COINS)} عملة | 🔬 {len(TESTS)} تكوين\n",flush=True)

all_pt={t[0]:[] for t in TESTS}; processed=0; t_total=time.time()

for ci,coin in enumerate(COINS):
    fpath=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw=json.load(f)
    if len(raw)<200: del raw; continue
    df=pd.DataFrame(raw)
    df=df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=compute_indicators(df)
    close_arr=df['close'].values; timestamps=df['ts'].values
    sidxs=find_signals(df)
    if len(sidxs)==0: del df; gc.collect(); processed+=1; continue
    
    for name,tp,sl,pl,trail,max_h,max_pos in TESTS:
        for idx in sidxs:
            ep=close_arr[idx]; ets=timestamps[idx]
            xtyp,pnl,xidx=compute_trade(close_arr,idx,ep,tp,sl,pl,trail,max_h)
            xts=timestamps[xidx]
            all_pt[name].append((ets,coin,idx,xts,xidx,pnl,xtyp))
    del df; gc.collect(); processed+=1
    if processed%40==0:
        el=time.time()-t_total; eta=el/processed*(len(COINS)-processed)
        print(f"  ⏳ {processed}/{len(COINS)} | {el:.0f}s | ETA {eta:.0f}s",flush=True)

print(f"\n✅ {processed} عملة | {time.time()-t_total:.0f}s\n",flush=True)

all_results=[]
for name,tp,sl,pl,trail,max_h,max_pos in TESTS:
    pot=all_pt[name]
    if not pot: print(f"📊 {name}: ❌ 0"); continue
    exc,skp=global_sim(pot,max_pos)
    if not exc: print(f"📊 {name}: ❌ 0"); continue
    
    pnls=[e[2] for e in exc]; wins=[p for p in pnls if p>0]; loss=[p for p in pnls if p<=0]
    wr=len(wins)/len(exc)*100; tn=sum(pnls)
    aw=np.mean(wins) if wins else 0; al=np.mean(loss) if loss else 0
    rr=aw/abs(al) if al!=0 else 0
    ec={}; [ec.update({e[3]:ec.get(e[3],0)+1}) for e in exc]
    eq,mdd=calc_portfolio(exc,max_pos)
    sh=np.mean(pnls)/np.std(pnls)*np.sqrt(len(pnls)) if len(pnls)>1 else 0
    days=122; ar=((eq/1000)**(365/days)-1)*100
    slt="بدون" if sl==99 else f"{sl}%"
    
    all_results.append({
        'name':name,'trades':len(exc),'skipped':skp,'pot':len(pot),
        'wins':len(wins),'losses':len(loss),'wr':wr,
        'net':tn,'aw':aw,'al':al,'rr':rr,
        'tpc':ec.get('TP',0),'slc':ec.get('SL',0),'trc':ec.get('TRAIL',0),'tmc':ec.get('TIME',0),
        'sh':sh,'dd':mdd,'eq':eq,'ar':ar,
    })

sr=sorted(all_results,key=lambda x:x['rr'],reverse=True)

print(f"{'='*80}")
print("📊 تحسين R:R — C2 دخول | MAX_POS عالمي")
print(f"{'='*80}")
print(f"  {'التجربة':<28} {'صفقات':>6} {'WR':>7} {'R:R':>5} {'م.ربح':>7} {'م.خسارة':>8} {'محفظة':>9} {'DD':>7} {'سنوي':>8} {'شارپ':>6}")
print(f"  {'─'*28} {'─'*6} {'─'*7} {'─'*5} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*6}")
for r in sr:
    print(f"  {r['name']:<28} {r['trades']:>6} {r['wr']:>6.1f}% {r['rr']:>4.1f}x {r['aw']:>+6.2f}% {r['al']:>+7.2f}% ${r['eq']:>8,.0f} {r['dd']:>6.1f}% {r['ar']:>+7.1f}% {r['sh']:>6.2f}")

print(f"\n{'─'*80}")
print("🔍 تفاصيل الصفقات:")
for r in sr:
    print(f"  {r['name']}: TP={r['tpc']} SL={r['slc']} TRAIL={r['trc']} TIME={r['tmc']} | منفذة={r['trades']} من {r['pot']}")
