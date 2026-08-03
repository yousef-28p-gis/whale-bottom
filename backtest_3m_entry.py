#!/usr/bin/env python3
"""🧪 تغيير جذري للدخول — فترات حوت مختلفة + SPK أعلى + دخول مختلف"""
import json, numpy as np, pandas as pd, os, time, gc

COMM=0.20; TF_MIN=3; CAPITAL=1000
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# (name, whale_periods, whale_min, rsi_period, rsi_max, spike_min, entry_mode, tp, trail, time_h, max_pos, pl)
# whale_periods: string like "2,3,5,7" or "3,5,8,13"
# entry_mode: 'close' (whale close), 'green' (first green candle after whale), 'breakout' (close > whale high)

TESTS=[
    ("B0_org_2357",        "2,3,5,7",   0.25,14,25,2.0, 'close',    2.5,0.05,8,2,30),
    ("W1_35813",           "3,5,8,13",  0.25,14,25,2.0, 'close',    2.5,0.05,8,2,30),
    ("W2_1245",            "1,2,4,5",   0.25,14,25,2.0, 'close',    2.5,0.05,8,2,30),
    ("W3_2468",            "2,4,6,8",   0.25,14,25,2.0, 'close',    2.5,0.05,8,2,30),
    ("S1_SPK25",           "2,3,5,7",   0.25,14,25,2.5, 'close',    2.5,0.05,8,2,30),
    ("S2_SPK30",           "2,3,5,7",   0.25,14,25,3.0, 'close',    2.5,0.05,8,2,30),
    ("R1_RSI7",            "2,3,5,7",   0.25,7, 25,2.0, 'close',    2.5,0.05,8,2,30),
    ("R2_RSI21",           "2,3,5,7",   0.25,21,25,2.0, 'close',    2.5,0.05,8,2,30),
    ("E1_green",           "2,3,5,7",   0.25,14,25,2.0, 'green',    2.5,0.05,8,2,30),
    ("E2_breakout",        "2,3,5,7",   0.25,14,25,2.0, 'breakout', 2.5,0.05,8,2,30),
    ("H1_whale30",         "2,3,5,7",   0.30,14,25,2.0, 'close',    2.5,0.05,8,2,30),
    ("H2_RSI20",           "2,3,5,7",   0.25,14,20,2.0, 'close',    2.5,0.05,8,2,30),
]

def compute_indicators(df, wp, rsi_p):
    n=len(df); c=df['close'].values; h=df['high'].values; l=df['low'].values; v=df['volume'].values
    # Whale with custom periods
    periods=[int(x) for x in wp.split(',')]
    df['low_raw']=l.copy()
    for p in periods:
        df['low_raw']=df['low_raw'].rolling(p).min()
    w=(l-df['low_raw'].values)/np.where(df['low_raw'].values!=0,df['low_raw'].values,np.nan)*100
    df['whale']=np.clip(w,0,None)
    # Spike
    vm=df['volume'].rolling(20).mean().values; df['spike']=v/np.where(vm!=0,vm,np.nan)
    # RSI custom
    d=c.copy(); d[1:]=c[1:]-c[:-1]; d[0]=0
    g=np.where(d>0,d,0); ls=np.where(d<0,-d,0)
    ag=pd.Series(g).rolling(rsi_p).mean().values; al=pd.Series(ls).rolling(rsi_p).mean().values
    rs=ag/np.where(al!=0,al,np.nan); df['rsi']=100-(100/(1+rs))
    return df

def find_alerts(df, wm, rm, sm):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    w=df['whale'].values; s=df['spike'].values; r=df['rsi'].values
    mask=(w>=wm)&(s>=sm)&(r<rm)&~np.isnan(w)&~np.isnan(s)&~np.isnan(r)
    mask[:50]=False
    hp=np.zeros(n,dtype=bool)
    for sh in[1,2,3]: shf=np.zeros(n,dtype=bool); shf[sh:]=mask[:-sh]; hp|=shf
    mask&=~hp
    return np.where(mask)[0]

def get_entries(df, alerts, mode):
    n=len(df); close=df['close'].values; opens=df['open'].values; high=df['high'].values
    entries=[]
    if mode=='close':
        for idx in alerts:
            entries.append((idx, close[idx]))
    elif mode=='green':
        for idx in alerts:
            if idx+1<n and close[idx+1]>opens[idx+1]:
                entries.append((idx+1, close[idx+1]))
    elif mode=='breakout':
        for idx in alerts:
            whale_high=high[idx]
            # Look forward up to 5 candles for close > whale high
            found=False
            for i in range(idx+1, min(idx+6, n)):
                if close[i]>whale_high:
                    entries.append((i, close[i])); found=True; break
    return entries

def compute_trade(close_arr, ei, ep, tp, trail, max_h, pl):
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

# Cache DataFrames by whale_periods+rsi_period combo to avoid recomputing
df_cache={}
for ci,coin in enumerate(COINS):
    fpath=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw=json.load(f)
    if len(raw)<200: del raw; continue
    df=pd.DataFrame(raw)
    df=df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    
    for name,wp,wm,rp,rm,sm,mode,tp,trail,mh,mp,pl in TESTS:
        key=(wp,rp)
        if key not in df_cache:
            dfc=df.copy(); dfc=compute_indicators(dfc,wp,rp)
            df_cache[key]=dfc
        dfc=df_cache[key]
        close_arr=dfc['close'].values; timestamps=dfc['ts'].values
        
        alerts=find_alerts(dfc,wm,rm,sm)
        alerts_total[name]+=len(alerts)
        if len(alerts)==0: continue
        
        entries=get_entries(dfc,alerts,mode)
        for eidx,ep in entries:
            ets=timestamps[eidx]
            xtyp,pnl,xidx=compute_trade(close_arr,eidx,ep,tp,trail,mh,pl)
            xts=timestamps[xidx] if xidx<len(timestamps) else timestamps[-1]
            all_pt[name].append((ets,coin,eidx,xts,xidx,pnl,xtyp))
    
    df_cache.clear(); del df; gc.collect(); processed+=1
    if processed%40==0:
        el=time.time()-t_total; eta=el/processed*(len(COINS)-processed)
        print(f"  ⏳ {processed}/{len(COINS)} | {el:.0f}s | ETA {eta:.0f}s",flush=True)

print(f"\n✅ {processed} عملة | {time.time()-t_total:.0f}s\n",flush=True)

all_results=[]
for name,wp,wm,rp,rm,sm,mode,tp,trail,mh,mp,pl in TESTS:
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
    
    desc=f"حوت{wp} RSI{rp} WHALE≥{wm} SPK≥{sm}"
    if mode!='close': desc+=f' {mode}'
    
    all_results.append({
        'name':name,'trades':len(exc),'skipped':skp,'pot':len(pot),
        'wins':len(wins),'losses':len(loss),'wr':wr,'net':tn,'aw':aw,'al':al,'rr':rr,
        'sh':sh,'dd':mdd,'eq':eq,'ar':ar,'desc':desc,'alerts':alerts_total[name],
    })

sr=sorted(all_results,key=lambda x:x['eq'],reverse=True)

print(f"{'='*105}")
print("📊 تغيير جذري للدخول — فترات حوت + SPK + RSI + نمط دخول")
print(f"{'='*105}")
print(f"  {'التكوين':<22} {'الوصف':<40} {'إنذارات':>7} {'صفقات':>6} {'WR':>7} {'R:R':>5} {'محفظة':>9} {'DD':>7} {'سنوي':>8} {'شارپ':>6}")
print(f"  {'─'*22} {'─'*40} {'─'*7} {'─'*6} {'─'*7} {'─'*5} {'─'*9} {'─'*7} {'─'*8} {'─'*6}")
for r in sr:
    print(f"  {r['name']:<22} {r['desc']:<40} {r['alerts']:>7} {r['trades']:>6} {r['wr']:>6.1f}% {r['rr']:>4.1f}x ${r['eq']:>8,.0f} {r['dd']:>6.1f}% {r['ar']:>+7.1f}% {r['sh']:>6.2f}")
