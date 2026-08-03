#!/usr/bin/env python3
"""🧪 إنذار قوي + MA/MACD تأكيد دخول + NoSL/TRAIL خروج — عالمي"""
import json, numpy as np, pandas as pd, os, time, gc

COMM=0.20; TF_MIN=3; CAPITAL=1000
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# (name, confirm_mode, ma_fast, ma_slow, macd_f, macd_s, macd_sig, wait, tp, trail)
TESTS=[
    ("B0_NoConfirm_TP2.5",       'none', 0,0,  0,0,0, 0,  2.5,0.08),
    ("MA5x20_w5_TP2.5",          'ma',   5,20, 0,0,0, 5,  2.5,0.08),
    ("MA5x20_w10_TP2.5",         'ma',   5,20, 0,0,0, 10, 2.5,0.08),
    ("MA5x20_w5_TP3.0",          'ma',   5,20, 0,0,0, 5,  3.0,0.08),
    ("MA5x13_w5_TP2.5",          'ma',   5,13, 0,0,0, 5,  2.5,0.08),
    ("MA10x20_w5_TP2.5",         'ma',  10,20, 0,0,0, 5,  2.5,0.08),
    ("MACD1269_w5_TP2.5",        'macd', 0,0, 12,26,9, 5,  2.5,0.08),
    ("MA5x20_w5_TP2.5_TR10",     'ma',   5,20, 0,0,0, 5,  2.5,0.10),
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
    # MACD 12-26-9
    e12=df['close'].ewm(span=12,adjust=False).mean().values
    e26=df['close'].ewm(span=26,adjust=False).mean().values
    df['macd']=e12-e26
    df['macd_sig']=pd.Series(df['macd']).ewm(span=9,adjust=False).mean().values
    # MAs
    for p in[5,10,13,20,50]:
        df[f'ma{p}']=df['close'].rolling(p).mean().values
    return df

def find_alerts(df):
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

def find_entries(df, alerts, mode, mf, ms, wait):
    n=len(df); close=df['close'].values; opens=df['open'].values
    
    if mode=='none':
        return [(idx, close[idx]) for idx in alerts]
    
    entries=[]; watch_until=-1
    
    for alert_idx in alerts:
        ws=max(alert_idx, watch_until); we=min(alert_idx+wait, n-1)
        if we<=ws: watch_until=max(watch_until, we); continue
        
        found=False
        for i in range(alert_idx, we):
            if mode=='ma':
                mf_arr=df[f'ma{mf}'].values; ms_arr=df[f'ma{ms}'].values
                if np.isnan(mf_arr[i]) or np.isnan(ms_arr[i]): continue
                if i>0 and mf_arr[i]>ms_arr[i] and mf_arr[i-1]<=ms_arr[i-1]:
                    if i+1<n and close[i+1]>opens[i+1]:
                        entries.append((i+1, close[i+1])); found=True; break
            elif mode=='macd':
                macd=df['macd'].values; sig=df['macd_sig'].values
                if np.isnan(macd[i]) or np.isnan(sig[i]): continue
                if i>0 and macd[i]>sig[i] and macd[i-1]<=sig[i-1]:
                    if i+1<n and close[i+1]>opens[i+1]:
                        entries.append((i+1, close[i+1])); found=True; break
        
        if found: watch_until=i+2
        else: watch_until=max(watch_until, we)
    return entries

def compute_trade(close_arr, entry_idx, entry_price, tp, trail):
    n=len(close_arr); mb=int(6*60/TF_MIN)
    tpp=entry_price*(1+tp/100); tr=1-trail/100
    pt=False; pk=entry_price; tlp=entry_price
    for i in range(entry_idx+1, n):
        cur=close_arr[i]; bh=i-entry_idx
        if bh>=mb: pnl=round((cur/entry_price-1)*100-COMM,4); return('TIME',pnl,i)
        if cur>=tpp: pnl=round((tpp/entry_price-1)*100-COMM,4); return('TP',pnl,i)
        if pt:
            if cur>pk: pk=cur; tlp=cur*tr
            if cur<=tlp: pnl=round((tlp/entry_price-1)*100-COMM,4); return('TRAIL',pnl,i)
        else:
            plp=entry_price+(tpp-entry_price)*0.30
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
print(f"📋 {len(COINS)} عملة | 🎯 WHALE≥0.25 RSI<25 SPK≥2.0 | NoSL+TRAIL\n",flush=True)

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
    
    for name,mode,mf,ms,mf_m,ms_m,sig_m,wait,tp,trail in TESTS:
        entries=find_entries(df,alerts,mode,mf,ms,wait)
        for eidx,ep in entries:
            ets=timestamps[eidx]
            xtyp,pnl,xidx=compute_trade(close_arr,eidx,ep,tp,trail)
            xts=timestamps[xidx]
            all_pt[name].append((ets,coin,eidx,xts,xidx,pnl,xtyp))
    
    del df; gc.collect(); processed+=1
    if processed%40==0:
        el=time.time()-t_total; eta=el/processed*(len(COINS)-processed)
        print(f"  ⏳ {processed}/{len(COINS)} | {el:.0f}s | ETA {eta:.0f}s",flush=True)

print(f"\n✅ {processed} عملة | 🚨 {alerts_total} إنذار | {time.time()-t_total:.0f}s\n",flush=True)

all_results=[]
for name,mode,mf,ms,mf_m,ms_m,sig_m,wait,tp,trail in TESTS:
    pot=all_pt[name]
    if not pot: print(f"📊 {name}: ❌ 0"); continue
    exc,skp=global_sim(pot,2)  # MAX_POS=2 for all
    if not exc: print(f"📊 {name}: ❌ 0"); continue
    
    pnls=[e[2] for e in exc]; wins=[p for p in pnls if p>0]; loss=[p for p in pnls if p<=0]
    wr=len(wins)/len(exc)*100; tn=sum(pnls)
    aw=np.mean(wins) if wins else 0; al=np.mean(loss) if loss else 0
    rr=aw/abs(al) if al!=0 else 0
    ec={}; [ec.update({e[3]:ec.get(e[3],0)+1}) for e in exc]
    eq,mdd=calc_portfolio(exc,2)
    sh=np.mean(pnls)/np.std(pnls)*np.sqrt(len(pnls)) if len(pnls)>1 else 0
    days=122; ar=((eq/1000)**(365/days)-1)*100
    
    if mode=='none': desc="بدون تأكيد"
    elif mode=='ma': desc=f"MA({mf},{ms}) w{wait}"
    else: desc=f"MACD w{wait}"
    
    all_results.append({
        'name':name,'trades':len(exc),'skipped':skp,'pot':len(pot),
        'wins':len(wins),'losses':len(loss),'wr':wr,'net':tn,'aw':aw,'al':al,'rr':rr,
        'tpc':ec.get('TP',0),'slc':ec.get('SL',0),'trc':ec.get('TRAIL',0),'tmc':ec.get('TIME',0),
        'sh':sh,'dd':mdd,'eq':eq,'ar':ar,'desc':desc,'tp':tp,'trail':trail,
    })

sr=sorted(all_results,key=lambda x:x['eq'],reverse=True)

print(f"{'='*90}")
print("📊 إنذار قوي + تأكيد MA/MACD + NoSL/TRAIL خروج | MAX_POS=2 عالمي")
print(f"{'='*90}")
print(f"  {'التجربة':<24} {'تأكيد':<18} {'دخول':>5} {'منفذة':>6} {'WR':>7} {'R:R':>5} {'م.ربح':>7} {'م.خسارة':>8} {'محفظة':>9} {'DD':>7} {'سنوي':>8} {'شارپ':>6}")
print(f"  {'─'*24} {'─'*18} {'─'*5} {'─'*6} {'─'*7} {'─'*5} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*6}")
for r in sr:
    print(f"  {r['name']:<24} {r['desc']:<18} {r['pot']:>5} {r['trades']:>6} {r['wr']:>6.1f}% {r['rr']:>4.1f}x {r['aw']:>+6.2f}% {r['al']:>+7.2f}% ${r['eq']:>8,.0f} {r['dd']:>6.1f}% {r['ar']:>+7.1f}% {r['sh']:>6.2f}")

print(f"\n{'─'*90}")
print("🔍 تفاصيل المخارج (TP/TRAIL/TIME):")
for r in sr:
    print(f"  {r['name']:<24} TP={r['tpc']:>3} | TRAIL={r['trc']:>4} | TIME={r['tmc']:>4} | إجمالي={r['trades']} | TP={r['tp']}% TRAIL={r['trail']}%")
