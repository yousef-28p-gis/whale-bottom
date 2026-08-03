#!/usr/bin/env python3
"""🧪 تجارب جديدة — PL متنوع + MACD خروج + BB + حجم — MAX_POS=2 عالمي"""
import json, numpy as np, pandas as pd, os, time, gc

COMM=0.20; TF_MIN=3; CAPITAL=1000
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# (name, tp, trail, time_h, max_pos, pl, exit_mode, bb_filter, vol_exit)
# exit_mode: 'nosl' (normal), 'macd' (MACD crossunder exit), 'vol' (volume spike exit)
# bb_filter=True → enter only if close < lower BB(20,2)
# vol_exit_mult: 0=disabled, >0 = exit when volume > vol_exit_mult * avg_volume

TESTS=[
    ("B0_base",              2.5,0.05,8,2, 30,'nosl',False,0),   # baseline
    ("P1_PL15",              2.5,0.05,8,2, 15,'nosl',False,0),   # trail sooner
    ("P2_PL20",              2.5,0.05,8,2, 20,'nosl',False,0),   # trail a bit sooner
    ("P3_PL40",              2.5,0.05,8,2, 40,'nosl',False,0),   # trail later
    ("P4_PL50",              2.5,0.05,8,2, 50,'nosl',False,0),   # trail much later
    ("M1_MACDexit",          2.5,0.05,8,2, 30,'macd',False,0),   # MACD exit
    ("V1_BBfilter",          2.5,0.05,8,2, 30,'nosl',True, 0),   # BB filter
    ("V2_VolExit",           2.5,0.05,8,2, 30,'vol', False,3.0), # exit on vol spike
]

def compute_indicators(df):
    n=len(df); c=df['close'].values; h=df['high'].values; l=df['low'].values; v=df['volume'].values
    # Whale
    df['low_lc']=df['low'].rolling(2).min(); df['low_sm']=df['low_lc'].rolling(3).min()
    df['low_hi']=df['low_sm'].rolling(5).min(); df['low_raw']=df['low_hi'].rolling(7).min()
    w=(l-df['low_raw'].values)/np.where(df['low_raw'].values!=0,df['low_raw'].values,np.nan)*100
    df['whale']=np.clip(w,0,None)
    vm=df['volume'].rolling(20).mean().values; df['spike']=v/np.where(vm!=0,vm,np.nan)
    df['vol_ma20']=vm
    # RSI
    d=c.copy(); d[1:]=c[1:]-c[:-1]; d[0]=0
    g=np.where(d>0,d,0); ls=np.where(d<0,-d,0)
    ag=pd.Series(g).rolling(14).mean().values; al=pd.Series(ls).rolling(14).mean().values
    rs=ag/np.where(al!=0,al,np.nan); df['rsi']=100-(100/(1+rs))
    # MACD
    e12=df['close'].ewm(span=12,adjust=False).mean().values
    e26=df['close'].ewm(span=26,adjust=False).mean().values
    df['macd']=e12-e26
    df['macd_sig']=pd.Series(df['macd']).ewm(span=9,adjust=False).mean().values
    # Bollinger
    df['bb_mid']=df['close'].rolling(20).mean().values
    df['bb_std']=df['close'].rolling(20).std().values
    df['bb_low']=df['bb_mid']-2*df['bb_std']
    return df

def find_alerts(df, bb_filter):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    w=df['whale'].values; s=df['spike'].values; r=df['rsi'].values; c=df['close'].values
    mask=(w>=0.25)&(s>=2.0)&(r<25)&~np.isnan(w)&~np.isnan(s)&~np.isnan(r)
    if bb_filter:
        bb=df['bb_low'].values
        mask&= (c<=bb) & ~np.isnan(bb)
    mask[:50]=False
    hp=np.zeros(n,dtype=bool)
    for sh in[1,2,3]: shf=np.zeros(n,dtype=bool); shf[sh:]=mask[:-sh]; hp|=shf
    mask&=~hp
    return np.where(mask)[0]

def compute_trade(close_arr, vol_arr, vol_ma, macd_arr, macd_sig, ei, ep, tp, trail, max_h, pl, exit_mode, vol_exit_mult):
    n=len(close_arr); mb=int(max_h*60/TF_MIN)
    
    if exit_mode=='macd':
        # Exit on MACD bearish crossover — no TP/TRAIL/TIME
        for i in range(ei+2, n):
            if np.isnan(macd_arr[i]) or np.isnan(macd_sig[i]): continue
            if macd_arr[i]<macd_sig[i] and macd_arr[i-1]>=macd_sig[i-1]:
                cur=close_arr[i]; pnl=round((cur/ep-1)*100-COMM,4)
                return('MACD',pnl,i)
        cur=close_arr[-1]; pnl=round((cur/ep-1)*100-COMM,4)
        return('EOD',pnl,n-1)
    
    if exit_mode=='vol':
        # Exit on volume spike — but also keep TP/TRAIL/TIME
        tpp=ep*(1+tp/100); tr=1-trail/100; pt=False; pk=ep; tlp=ep
        for i in range(ei+1, n):
            cur=close_arr[i]; bh=i-ei
            if bh>=mb: pnl=round((cur/ep-1)*100-COMM,4); return('TIME',pnl,i)
            if cur>=tpp: pnl=round((tpp/ep-1)*100-COMM,4); return('TP',pnl,i)
            # Volume exit
            if vol_exit_mult>0 and not np.isnan(vol_arr[i]) and vol_ma[i]>0:
                if vol_arr[i] > vol_exit_mult*vol_ma[i] and bh>3:
                    pnl=round((cur/ep-1)*100-COMM,4); return('VOL',pnl,i)
            if pt:
                if cur>pk: pk=cur; tlp=cur*tr
                if cur<=tlp: pnl=round((tlp/ep-1)*100-COMM,4); return('TRAIL',pnl,i)
            else:
                plp=ep+(tpp-ep)*(pl/100)
                if cur>=plp: pt=True; pk=cur; tlp=cur*tr
        return('OPEN',0.0,n-1)
    
    # Normal NoSL+TRAIL
    tpp=ep*(1+tp/100); tr=1-trail/100; pt=False; pk=ep; tlp=ep
    for i in range(ei+1, n):
        cur=close_arr[i]; bh=i-ei
        if bh>=mb: pnl=round((cur/ep-1)*100-COMM,4); return('TIME',pnl,i)
        if cur>=tpp: pnl=round((tpp/ep-1)*100-COMM,4); return('TP',pnl,i)
        if pt:
            if cur>pk: pk=cur; tlp=cur*tr
            if cur<=tlp: pnl=round((tlp/ep-1)*100-COMM,4); return('TRAIL',pnl,i)
        else:
            plp=ep+(tpp-ep)*(pl/100)
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
print(f"📋 {len(COINS)} عملة | 🔬 {len(TESTS)} تكوين\n",flush=True)

all_pt={t[0]:[] for t in TESTS}; alerts_total={t[0]:0 for t in TESTS}
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
    vol_arr=df['volume'].values; vol_ma=df['vol_ma20'].values
    macd_arr=df['macd'].values; macd_sig=df['macd_sig'].values
    
    for name,tp,trail,time_h,mp,pl,exit_mode,bb_filter,vol_exit in TESTS:
        alerts=find_alerts(df,bb_filter)
        alerts_total[name]+=len(alerts)
        if len(alerts)==0: continue
        for idx in alerts:
            ep=close_arr[idx]; ets=timestamps[idx]
            xtyp,pnl,xidx=compute_trade(close_arr,vol_arr,vol_ma,macd_arr,macd_sig,idx,ep,tp,trail,time_h,pl,exit_mode,vol_exit)
            xts=timestamps[xidx]
            all_pt[name].append((ets,coin,idx,xts,xidx,pnl,xtyp))
    
    del df; gc.collect(); processed+=1
    if processed%40==0:
        el=time.time()-t_total; eta=el/processed*(len(COINS)-processed)
        print(f"  ⏳ {processed}/{len(COINS)} | {el:.0f}s | ETA {eta:.0f}s",flush=True)

print(f"\n✅ {processed} عملة | {time.time()-t_total:.0f}s\n",flush=True)

all_results=[]
for name,tp,trail,time_h,mp,pl,exit_mode,bb_filter,vol_exit in TESTS:
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
    
    desc=f"PL={pl}%"
    if exit_mode=='macd': desc='MACD خروج'
    elif exit_mode=='vol': desc=f'حجم>{vol_exit}x'
    if bb_filter: desc+=' +BB'
    
    all_results.append({
        'name':name,'trades':len(exc),'skipped':skp,'pot':len(pot),
        'wins':len(wins),'losses':len(loss),'wr':wr,'net':tn,'aw':aw,'al':al,'rr':rr,
        'ec':ec,'sh':sh,'dd':mdd,'eq':eq,'ar':ar,'desc':desc,
    })

sr=sorted(all_results,key=lambda x:x['eq'],reverse=True)

print(f"{'='*95}")
print("📊 تجارب جديدة — PL + MACD خروج + BB + حجم")
print(f"{'='*95}")
print(f"  {'التكوين':<20} {'الوصف':<20} {'إنذارات':>6} {'صفقات':>6} {'WR':>7} {'R:R':>5} {'م.ربح':>7} {'م.خسارة':>8} {'محفظة':>9} {'DD':>7} {'سنوي':>8} {'شارپ':>6}")
print(f"  {'─'*20} {'─'*20} {'─'*6} {'─'*6} {'─'*7} {'─'*5} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*6}")
for r in sr:
    print(f"  {r['name']:<20} {r['desc']:<20} {alerts_total[r['name']]:>6} {r['trades']:>6} {r['wr']:>6.1f}% {r['rr']:>4.1f}x {r['aw']:>+6.2f}% {r['al']:>+7.2f}% ${r['eq']:>8,.0f} {r['dd']:>6.1f}% {r['ar']:>+7.1f}% {r['sh']:>6.2f}")

print(f"\n{'─'*95}")
print("🔍 توزيع المخارج:")
for r in sr:
    parts=[f"{k}:{v}" for k,v in sorted(r['ec'].items())]
    print(f"  {r['name']:<20} {' | '.join(parts)}")
