#!/usr/bin/env python3
"""🧪 المزيد — حوت بدون تأكيد + مخارج محسّنة + MACD بديل + MAX_POS=3"""
import json, numpy as np, pandas as pd, os, time, gc

COMM=0.20; TF_MIN=3; CAPITAL=1000
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# (name, entry_mode, tp, sl, pl, trail, max_h, max_pos, macd_f, macd_s, macd_sig, wait, exit_mode, atr_mult)
# entry_mode: 'whale' or 'macd'
# exit_mode: 'nosl' (no sl+trail), 'fixedsl' (fixed sl+tp), 'atrtrail' (atr-based trail), 'rsiexit' (rsi exit)
# sl=99 → no sl

TESTS=[
    # === Baseline ===
    ("B0_Whale_NoSL_TP2.5_TR08_6h_M2",   'whale',2.5,99,30,0.08,6,2, 0,0,0,0,  'nosl',0),
    
    # === SL طوارئ واسع + تريل ===
    ("W1_Whale_SL3_TP2.5_TR08_6h_M2",    'whale',2.5,3.0,30,0.08,6,2, 0,0,0,0,  'nosl',0),
    ("W2_Whale_SL4_TP2.5_TR08_6h_M2",    'whale',2.5,4.0,30,0.08,6,2, 0,0,0,0,  'nosl',0),
    ("W3_Whale_SL2_TP2.5_TR08_6h_M2",    'whale',2.5,2.0,30,0.08,6,2, 0,0,0,0,  'nosl',0),
    
    # === SL + TP بدون تريل ===
    ("W4_Whale_SL3_TP3_notrail_6h_M2",   'whale',3.0,3.0,0,0,    6,2, 0,0,0,0,  'fixedsl',0),
    ("W5_Whale_SL2_TP3_notrail_6h_M2",   'whale',3.0,2.0,0,0,    6,2, 0,0,0,0,  'fixedsl',0),
    ("W6_Whale_SL3_TP4_notrail_8h_M2",   'whale',4.0,3.0,0,0,    8,2, 0,0,0,0,  'fixedsl',0),
    
    # === ATR تريل ديناميكي ===
    ("W7_Whale_ATRtrail_TP2.5_6h_M2",    'whale',2.5,99,30,0,    6,2, 0,0,0,0,  'atrtrail',2.0),
    ("W8_Whale_ATRtrail_TP2.5_6h_M2x1.5",'whale',2.5,99,30,0,    6,2, 0,0,0,0,  'atrtrail',1.5),
    
    # === وقت مختلف ===
    ("W9_Whale_NoSL_TP2.5_TR08_4h_M2",   'whale',2.5,99,30,0.08,4,2, 0,0,0,0,  'nosl',0),
    ("W10_Whale_NoSL_TP2.5_TR08_8h_M2",  'whale',2.5,99,30,0.08,8,2, 0,0,0,0,  'nosl',0),
    
    # === MAX_POS=3 ===
    ("W11_Whale_NoSL_TP2.5_TR08_6h_M3",  'whale',2.5,99,30,0.08,6,3, 0,0,0,0,  'nosl',0),
    
    # === MACD(8,21,5) دخول أسرع + NoSL ===
    ("M1_MACD8215_w3_NoSL_M2",           'macd', 2.5,99,30,0.08,6,2, 8,21,5,3, 'nosl',0),
    ("M2_MACD8215_w5_NoSL_M2",           'macd', 2.5,99,30,0.08,6,2, 8,21,5,5, 'nosl',0),
    
    # === RSI خروج ===
    ("W12_Whale_RSIexit_TP2.5_6h_M2",    'whale',2.5,99,30,0.08,6,2, 0,0,0,0,  'rsiexit',0),
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
    # ATR
    tr1=h-l; tr2=np.abs(h-np.roll(c,1)); tr3=np.abs(l-np.roll(c,1))
    tr=np.maximum(np.maximum(tr1,tr2),tr3)
    df['atr']=pd.Series(tr).rolling(14).mean().values
    # MACDs
    for(fa,sl,sg) in[(12,26,9),(8,21,5)]:
        e1=df['close'].ewm(span=fa,adjust=False).mean().values
        e2=df['close'].ewm(span=sl,adjust=False).mean().values
        col=f'macd{fa}{sl}{sg}'; df[col]=e1-e2
        df[f'{col}_sig']=pd.Series(df[col]).ewm(span=sg,adjust=False).mean().values
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

def find_macd_entries(df, alerts, fast, slow, sig, wait):
    n=len(df); close=df['close'].values; opens=df['open'].values
    col=f'macd{fast}{slow}{sig}'; sc=f'{col}_sig'
    macd=df[col].values; sig_line=df[sc].values
    entries=[]; wu=-1
    for ai in alerts:
        ws=max(ai,wu); we=min(ai+wait,n-1)
        if we<=ws: wu=max(wu,we); continue
        found=False
        for i in range(ai,we):
            if np.isnan(macd[i]) or np.isnan(sig_line[i]): continue
            if i>0 and macd[i]>sig_line[i] and macd[i-1]<=sig_line[i-1]:
                if i+1<n and close[i+1]>opens[i+1]:
                    entries.append((i+1,close[i+1])); found=True; break
        if found: wu=i+2
        else: wu=max(wu,we)
    return entries

def compute_trade(close_arr, atr_arr, rsi_arr, entry_idx, entry_price, tp, sl, pl, trail, max_h, exit_mode, atr_mult):
    n=len(close_arr); mb=int(max_h*60/TF_MIN)
    if exit_mode=='fixedsl':
        return compute_fixedsl(close_arr, entry_idx, entry_price, tp, sl, mb)
    elif exit_mode=='atrtrail':
        return compute_atrtrail(close_arr, atr_arr, entry_idx, entry_price, tp, atr_mult, mb)
    elif exit_mode=='rsiexit':
        return compute_rsiexit(close_arr, rsi_arr, entry_idx, entry_price, mb)
    else:
        return compute_nosl(close_arr, entry_idx, entry_price, tp, sl, pl, trail, mb)

def compute_nosl(close_arr, ei, ep, tp, sl, pl, trail, mb):
    tpp=ep*(1+tp/100); tr=1-trail/100
    dsl=ep*(1-sl/100) if sl<90 else 0.0001
    pt=False; pk=ep; tlp=ep
    for i in range(ei+1, len(close_arr)):
        cur=close_arr[i]; bh=i-ei
        if bh>=mb: pnl=round((cur/ep-1)*100-COMM,4); return('TIME',pnl,i)
        if cur>=tpp: pnl=round((tpp/ep-1)*100-COMM,4); return('TP',pnl,i)
        if sl<90 and cur<=dsl: pnl=round((cur/ep-1)*100-COMM,4); return('SL',pnl,i)
        if pt:
            if cur>pk: pk=cur; tlp=cur*tr
            if cur<=tlp: pnl=round((tlp/ep-1)*100-COMM,4); return('TRAIL',pnl,i)
        else:
            plp=ep+(tpp-ep)*(pl/100)
            if cur>=plp: pt=True; pk=cur; tlp=cur*tr
    return('OPEN',0.0,len(close_arr)-1)

def compute_fixedsl(close_arr, ei, ep, tp, sl, mb):
    tpp=ep*(1+tp/100); slp=ep*(1-sl/100)
    for i in range(ei+1, len(close_arr)):
        cur=close_arr[i]; bh=i-ei
        if bh>=mb: pnl=round((cur/ep-1)*100-COMM,4); return('TIME',pnl,i)
        if cur>=tpp: pnl=round((tpp/ep-1)*100-COMM,4); return('TP',pnl,i)
        if cur<=slp: pnl=round((cur/ep-1)*100-COMM,4); return('SL',pnl,i)
    return('TIME',round((close_arr[-1]/ep-1)*100-COMM,4),len(close_arr)-1)

def compute_atrtrail(close_arr, atr_arr, ei, ep, tp, atr_mult, mb):
    tpp=ep*(1+tp/100); n=len(close_arr)
    pt=False; pk=ep; trail_price=ep
    for i in range(ei+1, n):
        cur=close_arr[i]; bh=i-ei
        if bh>=mb: pnl=round((cur/ep-1)*100-COMM,4); return('TIME',pnl,i)
        if cur>=tpp: pnl=round((tpp/ep-1)*100-COMM,4); return('TP',pnl,i)
        atr_v=atr_arr[i] if not np.isnan(atr_arr[i]) else ep*0.01
        if pt:
            if cur>pk: pk=cur
            trail_price=pk-atr_mult*atr_v
            if cur<=trail_price: pnl=round((trail_price/ep-1)*100-COMM,4); return('ATR_TRAIL',pnl,i)
        else:
            plp=ep+(tpp-ep)*0.30
            if cur>=plp: pt=True; pk=cur; trail_price=pk-atr_mult*atr_v
    return('TIME',round((close_arr[-1]/ep-1)*100-COMM,4),n-1)

def compute_rsiexit(close_arr, rsi_arr, ei, ep, mb):
    n=len(close_arr)
    for i in range(ei+1, n):
        cur=close_arr[i]; bh=i-ei
        if bh>=mb: pnl=round((cur/ep-1)*100-COMM,4); return('TIME',pnl,i)
        if not np.isnan(rsi_arr[i]) and rsi_arr[i]>70 and i>ei+3:
            pnl=round((cur/ep-1)*100-COMM,4); return('RSI_EXIT',pnl,i)
    return('TIME',round((close_arr[-1]/ep-1)*100-COMM,4),n-1)

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
    atr_arr=df['atr'].values; rsi_arr=df['rsi'].values
    
    alerts=find_alerts(df)
    if len(alerts)==0: del df; gc.collect(); processed+=1; continue
    alerts_total+=len(alerts)
    
    for name,em,tp,sl,pl,trail,mh,mp,mf,ms,sg,wait,xm,am in TESTS:
        if em=='whale':
            entries=[(idx,close_arr[idx]) for idx in alerts]
        else:
            entries=find_macd_entries(df,alerts,mf,ms,sg,wait)
        
        for eidx,ep in entries:
            ets=timestamps[eidx]
            xtyp,pnl,xidx=compute_trade(close_arr,atr_arr,rsi_arr,eidx,ep,tp,sl,pl,trail,mh,xm,am)
            xts=timestamps[xidx]
            all_pt[name].append((ets,coin,eidx,xts,xidx,pnl,xtyp))
    
    del df; gc.collect(); processed+=1
    if processed%40==0:
        el=time.time()-t_total; eta=el/processed*(len(COINS)-processed)
        print(f"  ⏳ {processed}/{len(COINS)} | {el:.0f}s | ETA {eta:.0f}s",flush=True)

print(f"\n✅ {processed} عملة | 🚨 {alerts_total} إنذار | {time.time()-t_total:.0f}s\n",flush=True)

all_results=[]
for name,em,tp,sl,pl,trail,mh,mp,mf,ms,sg,wait,xm,am in TESTS:
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
    
    if em=='macd': desc=f"MACD({mf},{ms},{sg})w{wait}"
    else: desc=f"حوت مباشر"
    ex_desc={'nosl':'NoSL+TRAIL','fixedsl':'SL+TP','atrtrail':'ATRتريل','rsiexit':'RSI>70'}[xm]
    
    all_results.append({
        'name':name,'trades':len(exc),'skipped':skp,'pot':len(pot),
        'wins':len(wins),'losses':len(loss),'wr':wr,'net':tn,'aw':aw,'al':al,'rr':rr,
        'ec':ec,'sh':sh,'dd':mdd,'eq':eq,'ar':ar,'desc':desc,'ex':ex_desc,'mp':mp,
        'tp':tp,'sl':sl,'trail':trail,'mh':mh,
    })

sr=sorted(all_results,key=lambda x:x['eq'],reverse=True)

print(f"{'='*95}")
print("📊 اختبارات موسعة — دخول + مخارج متنوعة | MAX_POS عالمي")
print(f"{'='*95}")
print(f"  {'التجربة':<30} {'دخول':<16} {'خروج':<12} {'MP':>2} {'صفقات':>6} {'WR':>7} {'R:R':>5} {'م.ربح':>7} {'م.خسارة':>8} {'محفظة':>9} {'DD':>7} {'سنوي':>8} {'شارپ':>6}")
print(f"  {'─'*30} {'─'*16} {'─'*12} {'──'} {'──────'} {'───────'} {'─────'} {'───────'} {'────────'} {'─────────'} {'───────'} {'────────'} {'──────'}")
for r in sr:
    print(f"  {r['name']:<30} {r['desc']:<16} {r['ex']:<12} {r['mp']:>2} {r['trades']:>6} {r['wr']:>6.1f}% {r['rr']:>4.1f}x {r['aw']:>+6.2f}% {r['al']:>+7.2f}% ${r['eq']:>8,.0f} {r['dd']:>6.1f}% {r['ar']:>+7.1f}% {r['sh']:>6.2f}")

print(f"\n{'─'*95}")
print("🔍 توزيع المخارج:")
for r in sr:
    parts=[f"{k}:{v}" for k,v in sorted(r['ec'].items())]
    print(f"  {r['name']:<30} {' | '.join(parts)}")
