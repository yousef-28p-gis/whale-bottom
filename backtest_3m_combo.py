#!/usr/bin/env python3
"""🧪 الدمج الذهبي — TIME=8h + MAX_POS=3 + تنويعات"""
import json, numpy as np, pandas as pd, os, time, gc

COMM=0.20; TF_MIN=3; CAPITAL=1000
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# (name, tp, trail, time_h, max_pos)
TESTS=[
    ("B0_TP2.5_TR08_6h_M2",  2.5,0.08,6,2),   # baseline
    ("T1_TP2.5_TR08_8h_M3",  2.5,0.08,8,3),   # 🏆 combo
    ("T2_TP2.5_TR05_8h_M3",  2.5,0.05,8,3),   # tighter trail
    ("T3_TP3.0_TR08_8h_M3",  3.0,0.08,8,3),   # higher TP
    ("T4_TP2.5_TR10_8h_M3",  2.5,0.10,8,3),   # wider trail
    ("T5_TP2.5_TR08_10h_M3", 2.5,0.08,10,3),  # longer time
    ("T6_TP2.5_TR08_8h_M4",  2.5,0.08,8,4),   # MAX_POS=4
    ("T7_TP3.0_TR05_8h_M3",  3.0,0.05,8,3),   # high TP + tight trail
]

def compute_indicators(df):
    n=len(df); c=df['close'].values; h=df['high'].values; l=df['low'].values; v=df['volume'].values
    df['low_lc']=df['low'].rolling(2).min(); df['low_sm']=df['low_lc'].rolling(3).min()
    df['low_hi']=df['low_sm'].rolling(5).min(); df['low_raw']=df['low_hi'].rolling(7).min()
    w=(l-df['low_raw'].values)/np.where(df['low_raw'].values!=0,df['low_raw'].values,np.nan)*100
    df['whale']=np.clip(w,0,None)
    vm=df['volume'].rolling(20).mean().values; df['spike']=v/np.where(vm!=0,vm,np.nan)
    d=c.copy(); d[1:]=c[1:]-c[:-1]; d[0]=0
    g=np.where(d>0,d,0); ls=np.where(d<0,-d,0)
    ag=pd.Series(g).rolling(14).mean().values; al=pd.Series(ls).rolling(14).mean().values
    rs=ag/np.where(al!=0,al,np.nan); df['rsi']=100-(100/(1+rs))
    return df

def find_alerts(df):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    w=df['whale'].values; s=df['spike'].values; r=df['rsi'].values
    mask=(w>=0.25)&(s>=2.0)&(r<25)&~np.isnan(w)&~np.isnan(s)&~np.isnan(r)
    mask[:50]=False
    hp=np.zeros(n,dtype=bool)
    for sh in[1,2,3]: shf=np.zeros(n,dtype=bool); shf[sh:]=mask[:-sh]; hp|=shf
    mask&=~hp
    return np.where(mask)[0]

def compute_trade(close_arr, ei, ep, tp, trail, max_h):
    n=len(close_arr); mb=int(max_h*60/TF_MIN)
    tpp=ep*(1+tp/100); tr=1-trail/100
    pt=False; pk=ep; tlp=ep
    for i in range(ei+1, n):
        cur=close_arr[i]; bh=i-ei
        if bh>=mb: pnl=round((cur/ep-1)*100-COMM,4); return('TIME',pnl,i)
        if cur>=tpp: pnl=round((tpp/ep-1)*100-COMM,4); return('TP',pnl,i)
        if pt:
            if cur>pk: pk=cur; tlp=cur*tr
            if cur<=tlp: pnl=round((tlp/ep-1)*100-COMM,4); return('TRAIL',pnl,i)
        else:
            plp=ep+(tpp-ep)*0.30
            if cur>=plp: pt=True; pk=cur; tlp=cur*tr
    return('OPEN',0.0,n-1)

def global_sim(potential, max_pos):
    potential.sort(key=lambda x:x[0])
    active=[]; executed=[]; skipped=0
    for ets,coin,eidx,xts,xidx,pnl,etyp in potential:
        active=[a for a in active if a[0]>ets]
        if len(active)>=max_pos: skipped+=1; continue
        active.append((xts,ets,coin,pnl,etyp))
        executed.append((ets,coin,pnl,etyp,xts))
    return executed,skipped

def calc_portfolio(executed, max_pos):
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
print(f"📋 {len(COINS)} عملة | 🎯 WHALE≥0.25 RSI<25 SPK≥2.0\n",flush=True)

all_pt={t[0]:[] for t in TESTS}; alerts_total=0
processed=0; t_total=time.time()

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
    
    alerts=find_alerts(df)
    if len(alerts)==0: del df; gc.collect(); processed+=1; continue
    alerts_total+=len(alerts)
    
    for name,tp,trail,time_h,mp in TESTS:
        for idx in alerts:
            ep=close_arr[idx]; ets=timestamps[idx]
            xtyp,pnl,xidx=compute_trade(close_arr,idx,ep,tp,trail,time_h)
            xts=timestamps[xidx]
            all_pt[name].append((ets,coin,idx,xts,xidx,pnl,xtyp))
    
    del df; gc.collect(); processed+=1
    if processed%40==0:
        el=time.time()-t_total; eta=el/processed*(len(COINS)-processed)
        print(f"  ⏳ {processed}/{len(COINS)} | {el:.0f}s | ETA {eta:.0f}s",flush=True)

print(f"\n✅ {processed} عملة | 🚨 {alerts_total} إنذار | {time.time()-t_total:.0f}s\n",flush=True)

all_results=[]
for name,tp,trail,time_h,mp in TESTS:
    pot=all_pt[name]
    if not pot: print(f"📊 {name}: ❌ 0"); continue
    exc,skp=global_sim(pot,mp)
    if not exc: print(f"📊 {name}: ❌ 0"); continue
    
    pnls=[e[2] for e in exc]; wins=[p for p in pnls if p>0]; loss=[p for p in pnls if p<=0]
    wr=len(wins)/len(exc)*100; tn=sum(pnls)
    aw=np.mean(wins) if wins else 0; al=np.mean(loss) if loss else 0
    rr=aw/abs(al) if al!=0 else 0
    ec={}; [ec.update({e[3]:ec.get(e[3],0)+1}) for e in exc]
    eq,mdd=calc_portfolio(exc,mp)
    sh=np.mean(pnls)/np.std(pnls)*np.sqrt(len(pnls)) if len(pnls)>1 else 0
    days=122; ar=((eq/1000)**(365/days)-1)*100
    
    all_results.append({
        'name':name,'trades':len(exc),'skipped':skp,'pot':len(pot),
        'wins':len(wins),'losses':len(loss),'wr':wr,'net':tn,'aw':aw,'al':al,'rr':rr,
        'tpc':ec.get('TP',0),'trc':ec.get('TRAIL',0),'tmc':ec.get('TIME',0),
        'sh':sh,'dd':mdd,'eq':eq,'ar':ar,'mp':mp,'tp':tp,'trail':trail,'mh':time_h,
    })

sr=sorted(all_results,key=lambda x:x['eq'],reverse=True)

print(f"{'='*95}")
print("📊 الدمج الذهبي — TIME أطول + MAX_POS أعلى")
print(f"{'='*95}")
print(f"  {'التكوين':<24} {'TP':>5} {'TRAIL':>6} {'TIME':>5} {'MP':>2} {'صفقات':>6} {'WR':>7} {'R:R':>5} {'م.ربح':>7} {'م.خسارة':>8} {'محفظة':>9} {'DD':>7} {'سنوي':>8} {'شارپ':>6} {'TP':>5} {'TRAIL':>6} {'TIME':>5}")
print(f"  {'─'*24} {'─'*5} {'─'*6} {'─'*5} {'──'} {'──────'} {'───────'} {'─────'} {'───────'} {'────────'} {'─────────'} {'───────'} {'────────'} {'──────'} {'─────'} {'──────'} {'─────'}")
for r in sr:
    print(f"  {r['name']:<24} {r['tp']:>4.1f}% {r['trail']:>5.2f}% {r['mh']:>4}h {r['mp']:>2}  {r['trades']:>6} {r['wr']:>6.1f}% {r['rr']:>4.1f}x {r['aw']:>+6.2f}% {r['al']:>+7.2f}% ${r['eq']:>8,.0f} {r['dd']:>6.1f}% {r['ar']:>+7.1f}% {r['sh']:>6.2f} {r['tpc']:>5} {r['trc']:>6} {r['tmc']:>5}")

print(f"\n{'─'*95}")
print("🔍 ملاحظات:")
for r in sr:
    pot_pct=r['trades']/r['pot']*100 if r['pot']>0 else 0
    skip_pct=r['skipped']/r['pot']*100 if r['pot']>0 else 0
    print(f"  {r['name']:<24} {r['pot']:>5} محتملة | {r['trades']:>4} منفذة ({pot_pct:.0f}%) | {r['skipped']:>4} متخطية ({skip_pct:.0f}%)")
