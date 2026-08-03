#!/usr/bin/env python3
"""🧪 حوت + MACD/MA دخول وخروج — المؤشر يحدد متى نشتري ومتى نبيع — عالمي"""
import json, numpy as np, pandas as pd, os, time, gc

COMM=0.20; TF_MIN=3; CAPITAL=1000
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# (name, exit_mode, macd_fast, macd_slow, macd_sig, wait_bars, ma_fast, ma_slow, max_pos)
# exit_mode: 'macd'=MACD exit, 'ma'=MA exit, 'nosl'=NoSL+TRAIL (hybrid)
# Entry alert: WHALE≥0.20 RSI<30 SPK≥1.5
# sl=99 means NoSL; only used for nosl mode

TESTS=[
    # MACD دخول + MACD خروج
    ("MACD_M1269_w5_MP2",   'macd', 12,26,9, 5,  0,0, 2),
    ("MACD_M1269_w10_MP2",  'macd', 12,26,9, 10, 0,0, 2),
    ("MACD_M5135_w5_MP2",   'macd', 5,13,5,  5,  0,0, 2),
    ("MACD_M5135_w10_MP2",  'macd', 5,13,5,  10, 0,0, 2),
    # MA دخول + MA خروج
    ("MA_5x20_w5_MP2",      'ma',   0,0,0,  5,  5,20, 2),
    ("MA_5x20_w10_MP2",     'ma',   0,0,0,  10, 5,20, 2),
    ("MA_10x50_w5_MP2",     'ma',   0,0,0,  5,  10,50,2),
    ("MA_10x50_w10_MP2",    'ma',   0,0,0,  10, 10,50,2),
    # MACD دخول + NoSL/TRAIL خروج (هجين)
    ("MACD1269_w5_NoSL_MP2",'nosl', 12,26,9, 5,  0,0, 2),
    ("MA5x20_w5_NoSL_MP2",  'nosl', 0,0,0,  5,  5,20, 2),
]

def compute_indicators(df):
    n=len(df); c=df['close'].values; h=df['high'].values; l=df['low'].values; v=df['volume'].values
    # Whale
    df['low_lc']=df['low'].rolling(2).min()
    df['low_sm']=df['low_lc'].rolling(3).min()
    df['low_hi']=df['low_sm'].rolling(5).min()
    df['low_raw']=df['low_hi'].rolling(7).min()
    w=(l-df['low_raw'].values)/np.where(df['low_raw'].values!=0,df['low_raw'].values,np.nan)*100
    df['whale']=np.clip(w,0,None)
    vm=df['volume'].rolling(20).mean().values
    df['spike']=v/np.where(vm!=0,vm,np.nan)
    # RSI
    d=c.copy(); d[1:]=c[1:]-c[:-1]; d[0]=0
    g=np.where(d>0,d,0); ls=np.where(d<0,-d,0)
    ag=pd.Series(g).rolling(14).mean().values; al=pd.Series(ls).rolling(14).mean().values
    rs=ag/np.where(al!=0,al,np.nan)
    df['rsi']=100-(100/(1+rs))
    # MACD 12-26-9
    e12=df['close'].ewm(span=12,adjust=False).mean().values
    e26=df['close'].ewm(span=26,adjust=False).mean().values
    df['macd12269']=e12-e26
    df['macd12269_sig']=pd.Series(df['macd12269']).ewm(span=9,adjust=False).mean().values
    # MACD 5-13-5
    e5=df['close'].ewm(span=5,adjust=False).mean().values
    e13=df['close'].ewm(span=13,adjust=False).mean().values
    df['macd5135']=e5-e13
    df['macd5135_sig']=pd.Series(df['macd5135']).ewm(span=5,adjust=False).mean().values
    # MAs
    df['ma5']=df['close'].rolling(5).mean().values
    df['ma10']=df['close'].rolling(10).mean().values
    df['ma20']=df['close'].rolling(20).mean().values
    df['ma50']=df['close'].rolling(50).mean().values
    return df

def find_whale_alerts(df):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    w=df['whale'].values; s=df['spike'].values; r=df['rsi'].values
    mask=(w>=0.20)&(s>=1.5)&(r<30)&~np.isnan(w)&~np.isnan(s)&~np.isnan(r)
    mask[:50]=False
    hp=np.zeros(n,dtype=bool)
    for sh in[1,2,3]:
        shf=np.zeros(n,dtype=bool); shf[sh:]=mask[:-sh]; hp|=shf
    mask&=~hp
    return np.where(mask)[0]

def find_entries(df, alerts, entry_mode, macd_fast, macd_slow, macd_sig, wait_bars, ma_fast, ma_slow):
    """Returns list of (entry_idx, entry_price)"""
    n=len(df); close=df['close'].values; opens=df['open'].values
    
    if entry_mode=='macd':
        macd_col=f'macd{macd_fast}{macd_slow}{macd_sig}'
        sig_col=f'{macd_col}_sig'
        macd=df[macd_col].values; sig=df[sig_col].values
    elif entry_mode=='ma':
        macd=None; sig=None
    
    entries=[]; watch_until=-1
    
    for alert_idx in alerts:
        watch_start=max(alert_idx, watch_until)
        watch_end=min(alert_idx+wait_bars, n-1)
        if watch_end<=watch_start:
            watch_until=max(watch_until, watch_end); continue
        
        found=False
        for i in range(alert_idx, watch_end):
            if entry_mode=='macd':
                if np.isnan(macd[i]) or np.isnan(sig[i]): continue
                if i>0 and macd[i]>sig[i] and macd[i-1]<=sig[i-1]:
                    if i+1<n and close[i+1]>opens[i+1]:
                        entries.append((i+1, close[i+1])); found=True; break
            elif entry_mode=='ma':
                ma_f=df[f'ma{ma_fast}'].values; ma_s=df[f'ma{ma_slow}'].values
                if np.isnan(ma_f[i]) or np.isnan(ma_s[i]): continue
                if i>0 and ma_f[i]>ma_s[i] and ma_f[i-1]<=ma_s[i-1]:
                    if i+1<n and close[i+1]>opens[i+1]:
                        entries.append((i+1, close[i+1])); found=True; break
        
        if found: watch_until=i+2
        else: watch_until=max(watch_until, watch_end)
    
    return entries

def compute_trade_macd_exit(close_arr, entry_idx, entry_price, df):
    """Exit on MACD bearish crossover"""
    n=len(close_arr)
    macd=df['macd12269'].values; sig=df['macd12269_sig'].values
    for i in range(entry_idx+2, n):
        if np.isnan(macd[i]) or np.isnan(sig[i]): continue
        if macd[i]<sig[i] and macd[i-1]>=sig[i-1]:
            cur=close_arr[i]; pnl=round((cur/entry_price-1)*100-COMM,4)
            return('MACD_EXIT', pnl, i)
    cur=close_arr[-1]; pnl=round((cur/entry_price-1)*100-COMM,4)
    return('EOD', pnl, n-1)

def compute_trade_macd5135_exit(close_arr, entry_idx, entry_price, df):
    n=len(close_arr)
    macd=df['macd5135'].values; sig=df['macd5135_sig'].values
    for i in range(entry_idx+2, n):
        if np.isnan(macd[i]) or np.isnan(sig[i]): continue
        if macd[i]<sig[i] and macd[i-1]>=sig[i-1]:
            cur=close_arr[i]; pnl=round((cur/entry_price-1)*100-COMM,4)
            return('MACD_EXIT', pnl, i)
    cur=close_arr[-1]; pnl=round((cur/entry_price-1)*100-COMM,4)
    return('EOD', pnl, n-1)

def compute_trade_ma_exit(close_arr, entry_idx, entry_price, df, ma_fast, ma_slow):
    n=len(close_arr)
    mf=df[f'ma{ma_fast}'].values; ms=df[f'ma{ma_slow}'].values
    for i in range(entry_idx+2, n):
        if np.isnan(mf[i]) or np.isnan(ms[i]): continue
        if mf[i]<ms[i] and mf[i-1]>=ms[i-1]:
            cur=close_arr[i]; pnl=round((cur/entry_price-1)*100-COMM,4)
            return('MA_EXIT', pnl, i)
    cur=close_arr[-1]; pnl=round((cur/entry_price-1)*100-COMM,4)
    return('EOD', pnl, n-1)

def compute_trade_nosl(close_arr, entry_idx, entry_price):
    """NoSL + TRAIL 0.08% + TIME 6h"""
    n=len(close_arr); mb=int(6*60/TF_MIN)
    tpp=entry_price*1.025; tr=0.9992  # 0.08% trail
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
print(f"📋 {len(COINS)} عملة | 🎯 إنذار: WHALE≥0.20 RSI<30 SPK≥1.5\n",flush=True)

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
    
    alerts=find_whale_alerts(df)
    if len(alerts)==0: del df; gc.collect(); processed+=1; continue
    alerts_total+=len(alerts)
    
    for name,mode,fst,slw,sig,wait,mf,ms,mp in TESTS:
        # Step 1: Find entries based on mode
        if mode in ('macd','nosl') and fst>0:
            entries=find_entries(df,alerts,'macd',fst,slw,sig,wait,0,0)
        elif mode in ('ma','nosl') and mf>0:
            entries=find_entries(df,alerts,'ma',0,0,0,wait,mf,ms)
        else:
            entries=[]
        
        for eidx,ep in entries:
            ets=timestamps[eidx]
            # Step 2: Compute trade exit based on mode
            if mode=='macd':
                if fst==5: xtyp,pnl,xidx=compute_trade_macd5135_exit(close_arr,eidx,ep,df)
                else: xtyp,pnl,xidx=compute_trade_macd_exit(close_arr,eidx,ep,df)
            elif mode=='ma':
                xtyp,pnl,xidx=compute_trade_ma_exit(close_arr,eidx,ep,df,mf,ms)
            else:  # nosl
                xtyp,pnl,xidx=compute_trade_nosl(close_arr,eidx,ep)
            
            xts=timestamps[xidx]
            all_pt[name].append((ets,coin,eidx,xts,xidx,pnl,xtyp))
    
    del df; gc.collect(); processed+=1
    if processed%40==0:
        el=time.time()-t_total; eta=el/processed*(len(COINS)-processed)
        print(f"  ⏳ {processed}/{len(COINS)} | {el:.0f}s | ETA {eta:.0f}s",flush=True)

print(f"\n✅ {processed} عملة | 🚨 {alerts_total} إنذار | {time.time()-t_total:.0f}s\n",flush=True)

all_results=[]
for name,mode,fst,slw,sig,wait,mf,ms,mp in TESTS:
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
    
    if mode=='macd': desc=f"MACD({fst},{slw},{sig}) w{wait}"
    elif mode=='ma': desc=f"MA({mf},{ms}) w{wait}"
    else: desc=f"MACD/MA+NoSL w{wait}"
    
    all_results.append({
        'name':name,'mode':mode,'trades':len(exc),'skipped':skp,'pot':len(pot),
        'wins':len(wins),'losses':len(loss),'wr':wr,'net':tn,'aw':aw,'al':al,'rr':rr,
        'exit_counts':ec,'sh':sh,'dd':mdd,'eq':eq,'ar':ar,'desc':desc,
    })

sr=sorted(all_results,key=lambda x:x['eq'],reverse=True)

print(f"{'='*90}")
print("📊 دخول وخروج بالمؤشرات | WHALE≥0.20 RSI<30 SPK≥1.5 | MAX_POS عالمي")
print(f"{'='*90}")
print(f"  {'التجربة':<24} {'نمط':>16} {'صفقات':>6} {'WR':>7} {'R:R':>5} {'م.ربح':>7} {'م.خسارة':>8} {'محفظة':>9} {'DD':>7} {'سنوي':>8} {'شارپ':>6}")
print(f"  {'─'*24} {'─'*16} {'─'*6} {'─'*7} {'─'*5} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*6}")
for r in sr:
    print(f"  {r['name']:<24} {r['desc']:<16} {r['trades']:>6} {r['wr']:>6.1f}% {r['rr']:>4.1f}x {r['aw']:>+6.2f}% {r['al']:>+7.2f}% ${r['eq']:>8,.0f} {r['dd']:>6.1f}% {r['ar']:>+7.1f}% {r['sh']:>6.2f}")

print(f"\n{'─'*90}")
print("🔍 توزيع المخارج:")
for r in sr:
    ec=r['exit_counts']
    parts=[f"{k}:{v}" for k,v in sorted(ec.items())]
    print(f"  {r['name']:<24} {', '.join(parts)}")
