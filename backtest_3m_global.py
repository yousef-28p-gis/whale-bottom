#!/usr/bin/env python3
"""🧪 باك تيست عالمي حقيقي — MAX_POS عالمي + ترتيب زمني — close-only (إصدار خفيف)"""
import json, numpy as np, pandas as pd, os, time, gc

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# ═══════════════ تكوينات ═══════════════
# (name, tp, sl, pl, trail, max_h, max_pos, entry_type)
# entry_type: 'C2' (WHALE≥0.25 RSI<25 SPK≥2.0) or 'Cl' (WHALE≥0.10 RSI<50 SPK≥1.5)

TESTS = [
    ("G1_C2_TP2_SL1.5_TR05_6h_1",     2.0, 1.5, 30, 0.05, 6, 1, 'C2'),
    ("A1_NoSL_TP2.5_TR08_6h_1",        2.5, 99,  30, 0.08, 6, 1, 'C2'),
    ("A2_NoSL_TP2.0_TR05_6h_1",        2.0, 99,  30, 0.05, 6, 1, 'C2'),
    ("A3_NoSL_TP2.0_TR10_8h_1",        2.0, 99,  30, 0.10, 8, 1, 'C2'),
    ("B1_TP2.5_SL2.0_TR05_6h_1",       2.5, 2.0, 30, 0.05, 6, 1, 'C2'),
    ("B2_TP3.0_SL2.0_TR05_8h_1",       3.0, 2.0, 30, 0.05, 8, 1, 'C2'),
    ("C1_NoSL_TP2.5_TR08_6h_2",        2.5, 99,  30, 0.08, 6, 2, 'C2'),
    ("C2_NoSL_TP2.0_TR05_6h_2",        2.0, 99,  30, 0.05, 6, 2, 'C2'),
    ("C3_TP2.5_SL2.0_TR05_6h_2",       2.5, 2.0, 30, 0.05, 6, 2, 'C2'),
    ("D1_Cl_NoSL_TP1.5_TR05_4h_1",     1.5, 99,  30, 0.05, 4, 1, 'Cl'),
    ("D2_Cl_TP1.5_SL1.0_TR05_6h_1",    1.5, 1.0, 30, 0.05, 6, 1, 'Cl'),
]

def compute_indicators(df):
    n = len(df)
    c=df['close'].values; h=df['high'].values; l=df['low'].values; v=df['volume'].values
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    w = (l - df['low_raw'].values) / np.where(df['low_raw'].values!=0, df['low_raw'].values, np.nan)*100
    df['whale'] = np.clip(w,0,None)
    vm=df['volume'].rolling(20).mean().values
    df['spike'] = v / np.where(vm!=0, vm, np.nan)
    d=c.copy(); d[1:]=c[1:]-c[:-1]; d[0]=0
    g=np.where(d>0,d,0); ls=np.where(d<0,-d,0)
    ag=pd.Series(g).rolling(14).mean().values; al=pd.Series(ls).rolling(14).mean().values
    rs=ag/np.where(al!=0,al,np.nan)
    df['rsi']=100-(100/(1+rs))
    return df

def find_signals(df, whale_m, rsi_m, spike_m):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    w=df['whale'].values; s=df['spike'].values; r=df['rsi'].values
    mask=(w>=whale_m)&(s>=spike_m)&(r<rsi_m)&~np.isnan(w)&~np.isnan(s)&~np.isnan(r)
    mask[:50]=False
    hp=np.zeros(n,dtype=bool)
    for sh in [1,2,3]:
        shf=np.zeros(n,dtype=bool); shf[sh:]=mask[:-sh]; hp|=shf
    mask&=~hp
    return np.where(mask)[0]

def compute_trade(close_arr, entry_idx, entry_price, tp, sl, pl, trail, max_h):
    n=len(close_arr); mb=int(max_h*60/TF_MIN)
    tpp=entry_price*(1+tp/100); tr=1-trail/100
    dsl=entry_price*(1-sl/100) if sl<90 else 0.0001
    pt=False; pk=entry_price; tlp=entry_price
    for i in range(entry_idx+1,n):
        cur=close_arr[i]; bh=i-entry_idx
        if bh>=mb:
            pnl=round((cur/entry_price-1)*100-COMM,4); return ('TIME',pnl,i)
        if cur>=tpp:
            pnl=round((tpp/entry_price-1)*100-COMM,4); return ('TP',pnl,i)
        if sl<90 and cur<=dsl:
            pnl=round((cur/entry_price-1)*100-COMM,4); return ('SL',pnl,i)
        if pt:
            if cur>pk: pk=cur; tlp=cur*tr
            if cur<=tlp:
                pnl=round((tlp/entry_price-1)*100-COMM,4); return ('TRAIL',pnl,i)
        else:
            plp=entry_price+(tpp-entry_price)*(pl/100)
            if cur>=plp: pt=True; pk=cur; tlp=cur*tr
    return ('OPEN',0.0,n-1)

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

# ═══════════════ MAIN ═══════════════
print("⏳ تجهيز ومعالجة عملة عملة...", flush=True)
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah=json.load(f)
COINS=[c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
print(f"📋 {len(COINS)} عملة | 🔬 {len(TESTS)} تكوين\n", flush=True)

all_pt={t[0]:[] for t in TESTS}
total_candles=0; processed=0; t_total=time.time()

for ci,coin in enumerate(COINS):
    fpath=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw=json.load(f)
    total_candles+=len(raw)
    if len(raw)<200: del raw; continue
    
    df=pd.DataFrame(raw)
    df=df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True)
    df=compute_indicators(df)
    close_arr=df['close'].values; timestamps=df['ts'].values
    
    # Find signals for each entry type
    sigs={}
    sigs['C2']=find_signals(df,0.25,25,2.0)
    sigs['Cl']=find_signals(df,0.10,50,1.5)
    
    for name,tp,sl,pl,trail,max_h,max_pos,etyp in TESTS:
        sidxs=sigs[etyp]
        if len(sidxs)==0: continue
        for idx in sidxs:
            ep=close_arr[idx]; ets=timestamps[idx]
            xtyp,pnl,xidx=compute_trade(close_arr,idx,ep,tp,sl,pl,trail,max_h)
            xts=timestamps[xidx]
            all_pt[name].append((ets,coin,idx,xts,xidx,pnl,xtyp))
    
    del df; gc.collect(); processed+=1
    if processed%30==0:
        el=time.time()-t_total; eta=el/processed*(len(COINS)-processed)
        print(f"  ⏳ {processed}/{len(COINS)} | {el:.0f}s | ETA {eta:.0f}s", flush=True)

el=time.time()-t_total
print(f"\n✅ {processed} عملة | {total_candles:,} شمعة | {el:.0f}s\n", flush=True)

# Phase 2: Global MAX_POS
print(f"{'='*90}")
print("📊 نتائج المحاكاة العالمية — MAX_POS عالمي + ترتيب زمني حقيقي")
print(f"{'='*90}")

all_results=[]
for name,tp,sl,pl,trail,max_h,max_pos,etyp in TESTS:
    pot=all_pt[name]
    if not pot: print(f"\n📊 {name}: ❌ 0 صفقات"); continue
    exc,skp=global_sim(pot,max_pos)
    if not exc: print(f"\n📊 {name}: ❌ 0 منفذة"); continue
    
    pnls=[e[2] for e in exc]; wins=[p for p in pnls if p>0]; loss=[p for p in pnls if p<=0]
    wr=len(wins)/len(exc)*100; tn=sum(pnls); tp_=sum(wins) if wins else 0; tl_=sum(loss) if loss else 0
    aw=np.mean(wins) if wins else 0; al=np.mean(loss) if loss else 0
    rr=aw/abs(al) if al!=0 else 0
    ec={}; [ec.update({e[3]:ec.get(e[3],0)+1}) for e in exc]
    eq,mdd=calc_portfolio(exc,max_pos)
    sh=np.mean(pnls)/np.std(pnls)*np.sqrt(len(pnls)) if len(pnls)>1 else 0
    days=122; ar=((eq/1000)**(365/days)-1)*100
    ent="C2(0.25/25/2.0)" if etyp=='C2' else "Cl(0.10/50/1.5)"
    slt="بدون" if sl==99 else f"{sl}%"
    
    all_results.append({
        'name':name,'trades':len(exc),'skipped':skp,'pot':len(pot),
        'wins':len(wins),'losses':len(loss),'wr':wr,
        'tp__':tp_,'tl__':tl_,'net':tn,'aw':aw,'al':al,'rr':rr,
        'tpc':ec.get('TP',0),'slc':ec.get('SL',0),'trc':ec.get('TRAIL',0),'tmc':ec.get('TIME',0),
        'sh':sh,'dd':mdd,'eq':eq,'ar':ar,'mp':max_pos,'slt':slt,'tpv':tp,'trl':trail,'mh':max_h,'ent':ent,
    })

sr=sorted(all_results,key=lambda x:x['eq'],reverse=True)

for r in sr:
    print(f"\n{'─'*70}")
    print(f"📊 {r['name']}")
    print(f"   🎯 {r['ent']} | TP={r['tpv']}% SL={r['slt']} TR={r['trl']}% TIME={r['mh']}h MP={r['mp']}")
    print(f"📋 محتملة: {r['pot']} | منفذة: {r['trades']} | متخطية: {r['skipped']} ({r['skipped']/r['pot']*100:.0f}%)")
    print(f"🟢 ربح: {r['wins']} | 🔴 خسارة: {r['losses']} | WR: {r['wr']:.1f}%")
    print(f"💵 ربح: +{r['tp__']:.1f}% | 💸 خسارة: {r['tl__']:.1f}% | صافي: {r['net']:+.1f}%")
    print(f"🟢 م.ربح: +{r['aw']:.2f}% | 🔴 م.خسارة: {r['al']:.2f}% | R:R: {r['rr']:.1f}x")
    print(f"📊 شارپ: {r['sh']:.2f} | 📉 سحب: {r['dd']:.1f}%")
    print(f"🏦 ${r['eq']:,.0f} (+{(r['eq']/10-100):.1f}%) | سنوي: {r['ar']:+.1f}%")
    print(f"🎯 TP:{r['tpc']} 🛑 SL:{r['slc']} 🐌 TRAIL:{r['trc']} ⏰ TIME:{r['tmc']}")

print(f"\n{'='*90}")
print("⚖️ ملخص مضغوط (MAX_POS عالمي)")
print(f"{'='*90}")
print(f"  {'التجربة':<32} {'محتملة':>6} {'منفذة':>6} {'WR':>7} {'صافي':>8} {'محفظة':>9} {'DD':>7} {'سنوي':>8} {'شارپ':>6}")
print(f"  {'─'*32} {'─'*6} {'─'*6} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*6}")
for r in sr:
    print(f"  {r['name']:<32} {r['pot']:>6} {r['trades']:>6} {r['wr']:>6.1f}% {r['net']:>+7.1f}% ${r['eq']:>8,.0f} {r['dd']:>6.1f}% {r['ar']:>+7.1f}% {r['sh']:>6.2f}")
