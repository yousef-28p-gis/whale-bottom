#!/usr/bin/env python3
"""آخر دفعة اختبارات: TP أعلى + تريل 0.02 + بدون تباعد 3 شمعات"""
import json, numpy as np, pandas as pd, os, time, gc

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000; MAX_POS = 2
POS_PCT = 50
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

CONFIGS = [
    # --- TP أعلى ---
    ("TP1.4_SL0.5_TR3",   1.4, 0.5, 12, 0.03, 4, 0.10, 35, True, True),
    ("TP1.5_SL0.5_TR3",   1.5, 0.5, 12, 0.03, 4, 0.10, 35, True, True),
    # --- TP1.3 مع تريل 0.02 ---
    ("TP1.3_SL0.5_TR2",   1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, True),
    # --- بدون تباعد 3 شمعات ---
    ("TP1.3_SL0.5_TR3_لا_تباعد", 1.3, 0.5, 12, 0.03, 4, 0.10, 35, True, False),
    ("TP1.0_SL0.5_TR3_لا_تباعد", 1.0, 0.5, 12, 0.03, 4, 0.10, 35, True, False),
    # --- المرجع ---
    ("TP1.3_SL0.5_TR3_مرجع", 1.3, 0.5, 12, 0.03, 4, 0.10, 35, True, True),
]

def compute_indicators(df):
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    w = (df['low'].values-df['low_raw'].values)/np.where(df['low_raw'].values!=0, df['low_raw'].values, np.nan)*100
    df['whale'] = np.clip(w,0,None)
    vm = df['volume'].rolling(20).mean().values
    df['spike'] = df['volume'].values/np.where(vm!=0, vm, np.nan)
    delta = df['close'].diff().values
    gain = pd.Series(np.where(delta>0,delta,0)).rolling(14).mean().values
    loss = pd.Series(np.where(delta<0,-delta,0)).rolling(14).mean().values
    df['rsi'] = 100-100/(1+gain/np.where(loss!=0,loss,np.nan))
    return df

def find_signals(df, whale_min, rsi_max, confirm, spacing):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    whale=df['whale'].values; spike=df['spike'].values; rsi=df['rsi'].values
    mask=(whale>=whale_min)&(spike>=1.5)&(rsi<rsi_max)&~np.isnan(whale)&~np.isnan(spike)&~np.isnan(rsi)
    mask[:50]=False
    if spacing:
        has_prev=np.zeros(n,dtype=bool)
        for shift in[1,2,3]:
            s=np.zeros(n,dtype=bool); s[shift:]=mask[:-shift]; has_prev|=s
        mask&=~has_prev
    if confirm:
        ng=np.zeros(n,dtype=bool); ng[:-1]=df['close'].values[1:]>df['open'].values[1:]
        mask&=ng
    return np.where(mask)[0]

print("⏳ جمع الصفقات + تحليل شهري + تحليل عملات...", flush=True)
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]

all_trades = {c[0]: [] for c in CONFIGS}
coin_stats = {c[0]: {} for c in CONFIGS}  # config -> coin -> [pnls]
monthly_trades = {c[0]: [] for c in CONFIGS}  # config -> [(month, pnl)]
processed=0; t0=time.time()

for coin in COINS:
    fpath=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw=json.load(f)
    if len(raw)<200: del raw; continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw
    df=compute_indicators(df)
    close_arr=df['close'].values
    ts_arr=df['ts'].values.astype('datetime64[ns]').astype('int64')
    # Convert to months for monthly analysis
    ts_dt = pd.to_datetime(df['ts'].values)
    
    for name,tp,sl,pl,trail,mh,whale,rsi,confirm,spacing in CONFIGS:
        idxs=find_signals(df,whale,rsi,confirm,spacing)
        if len(idxs)==0: continue
        max_bars=int(mh*60/TF_MIN); tp_r=1+tp/100; sl_r=1-sl/100; tr_r=1-trail/100
        active=[]; sig_map=dict(zip(idxs,close_arr[idxs]))
        coin_pnls = []
        for i in range(len(df)):
            cur=close_arr[i]
            if i in sig_map:
                active.append({'symbol':coin,'entry':sig_map[i],'tp':sig_map[i]*tp_r,'sl':sig_map[i]*sl_r,
                    'pl_ok':False,'peak':sig_map[i],'trail':sig_map[i],'entry_i':i,
                    'entry_ns':int(ts_arr[i]),'entry_month':ts_dt[i].month})
            for j in range(len(active)-1,-1,-1):
                p=active[j]; e=p['entry']; bh=i-p['entry_i']
                if bh>=max_bars:
                    pnl=round((cur/e-1)*100-COMM,4); p['pnl']=pnl; p['exit_type']='TIME'
                    p['exit_ns']=int(ts_arr[i]); p['month']=p['entry_month']
                    all_trades[name].append(p); coin_pnls.append(pnl); del active[j]
                elif cur>=p['tp']:
                    pnl=round(tp-COMM,4); p['pnl']=pnl; p['exit_type']='TP'
                    p['exit_ns']=int(ts_arr[i]); p['month']=p['entry_month']
                    all_trades[name].append(p); coin_pnls.append(pnl); del active[j]
                elif cur<=p['sl']:
                    pnl=round(-sl-COMM,4); p['pnl']=pnl; p['exit_type']='SL'
                    p['exit_ns']=int(ts_arr[i]); p['month']=p['entry_month']
                    all_trades[name].append(p); coin_pnls.append(pnl); del active[j]
                elif p['pl_ok']:
                    if cur>p['peak']: p['peak']=cur; p['trail']=cur*tr_r
                    if cur<=p['trail']:
                        pnl=round((p['trail']/e-1)*100-COMM,4); p['pnl']=pnl; p['exit_type']='TRAIL'
                        p['exit_ns']=int(ts_arr[i]); p['month']=p['entry_month']
                        all_trades[name].append(p); coin_pnls.append(pnl); del active[j]
                else:
                    pl_p=e+(p['tp']-e)*(pl/100)
                    if cur>=pl_p: p['pl_ok']=True; p['peak']=cur; p['trail']=cur*tr_r
        
        if coin_pnls:
            coin_stats[name][coin] = coin_pnls
    
    del df; gc.collect(); processed+=1
    if processed%50==0: print(f"  ⏳ {processed}/{len(COINS)} | {time.time()-t0:.0f}s", flush=True)

print(f"✅ {time.time()-t0:.0f}s\n", flush=True)

# ═══════════════ 1-3: نتائج التجارب ═══════════════
print(f"{'='*75}")
print(f"📊 نتائج التجارب الجديدة | MAX_POS=2 | 50%")
print(f"{'='*75}")

results = []
for name,tp,sl,pl,trail,mh,whale,rsi,confirm,spacing in CONFIGS:
    trades=all_trades[name]
    trades.sort(key=lambda t:t['entry_ns'])
    equity=float(CAPITAL); peak=float(CAPITAL); max_dd=0.0
    slots=[None]*MAX_POS; executed=0; skipped=0
    
    for t in trades:
        en=t['entry_ns']; ex=t['exit_ns']; pnl_pct=t['pnl']
        for s in range(MAX_POS):
            if slots[s] is not None:
                sex,spnl=slots[s]
                if sex<=en:
                    pos_cap=equity*(POS_PCT/100); pnl_dollar=pos_cap*(spnl/100); equity+=pnl_dollar; slots[s]=None
                    if equity>peak: peak=equity
                    dd=(equity-peak)/peak*100
                    if dd<max_dd: max_dd=dd
        free=-1
        for s in range(MAX_POS):
            if slots[s] is None: free=s; break
        if free==-1: skipped+=1; continue
        executed+=1; slots[free]=(ex,pnl_pct)
    
    for s in range(MAX_POS):
        if slots[s] is not None:
            sex,spnl=slots[s]; pos_cap=equity*(POS_PCT/100); pnl_dollar=pos_cap*(spnl/100); equity+=pnl_dollar
    
    if equity>peak: peak=equity
    dd=(equity-peak)/peak*100
    if dd<max_dd: max_dd=dd
    
    wins=sum(1 for t in trades if t['pnl']>0); losses=len(trades)-wins
    wr=wins/len(trades)*100
    avg_win=np.mean([t['pnl'] for t in trades if t['pnl']>0]) if wins else 0
    avg_loss=np.mean([t['pnl'] for t in trades if t['pnl']<=0]) if losses else 0
    rr=avg_win/abs(avg_loss) if avg_loss!=0 else 0
    tp_c=sum(1 for t in trades if t['exit_type']=='TP')
    sl_c=sum(1 for t in trades if t['exit_type']=='SL')
    tr_c=sum(1 for t in trades if t['exit_type']=='TRAIL')
    tm_c=sum(1 for t in trades if t['exit_type']=='TIME')
    
    fixed_pnl = sum(t['pnl']/100*500 for t in trades)
    
    results.append({
        'name':name,'tp':tp,'sl':sl,'trail':trail,'spacing':spacing,
        'signals':len(trades),'exec':executed,'skip':skipped,
        'wins':wins,'losses':losses,'wr':wr,'rr':rr,
        'avg_win':avg_win,'avg_loss':avg_loss,'dd':max_dd,
        'equity':equity,'fixed_eq':1000+fixed_pnl,
        'tp_c':tp_c,'sl_c':sl_c,'tr_c':tr_c,'tm_c':tm_c,
    })

for r in results:
    print(f"\n{'─'*60}")
    spacing_txt = "✅تباعد" if r['spacing'] else "❌بدون_تباعد"
    print(f"📊 {r['name']} | TP={r['tp']}% SL={r['sl']}% TR={r['trail']}% | {spacing_txt}")
    print(f"📋 إشارات: {r['signals']:,} | ✅ منفذة: {r['exec']:,} | ⏭️ متخطية: {r['skip']:,}")
    print(f"🟢 ربح: {r['wins']:,} | 🔴 خسارة: {r['losses']:,}")
    print(f"📈 WR: {r['wr']:.1f}%")
    print(f"🟢 متوسط ربح: +{r['avg_win']:.2f}% | 🔴 متوسط خسارة: {r['avg_loss']:.2f}%")
    print(f"📊 R:R: {r['rr']:.2f}x | 📉 سحب: {r['dd']:.1f}%")
    print(f"💰 بدون تركيب: $1,000 → ${r['fixed_eq']:,.0f} (+{(r['fixed_eq']/10-100):.1f}%)")
    print(f"🎯 TP:{r['tp_c']:,} 🛑 SL:{r['sl_c']:,} 🐌 TRAIL:{r['tr_c']:,} ⏱️ TIME:{r['tm_c']:,}")

# ═══════════════ 4: تحليل شهري ═══════════════
print(f"\n{'='*75}")
print(f"📅 تحليل شهري — TP1.3_SL0.5_TR3 (البطل)")
print(f"{'='*75}")

ref_trades = all_trades['TP1.3_SL0.5_TR3_مرجع']
monthly = {}  # month -> [pnls]
for t in ref_trades:
    m = t.get('month', 0)
    if m not in monthly: monthly[m] = []
    monthly[m].append(t['pnl'])

for m in sorted(monthly.keys()):
    pnls = monthly[m]
    wins_m = sum(1 for p in pnls if p > 0)
    wr_m = wins_m/len(pnls)*100
    net_m = sum(pnls)
    months_ar = {3:'مارس',4:'أبريل',5:'مايو',6:'يونيو',7:'يوليو'}
    print(f"  📅 {months_ar.get(m, m)}: {len(pnls):,} إشارة | WR {wr_m:.1f}% | صافي {net_m:+.1f}% | 🟢{wins_m} 🔴{len(pnls)-wins_m}")

# ═══════════════ 5: تحليل حسب العملة ═══════════════
print(f"\n{'='*75}")
print(f"🪙 أفضل وأسوأ 10 عملات — TP1.3_SL0.5_TR3")
print(f"{'='*75}")

cs = coin_stats.get('TP1.3_SL0.5_TR3_مرجع', {})
coin_ranking = []
for coin, pnls in cs.items():
    if len(pnls) < 5: continue
    wins_c = sum(1 for p in pnls if p > 0)
    wr_c = wins_c/len(pnls)*100
    net_c = sum(pnls)
    coin_ranking.append((coin, len(pnls), wr_c, net_c))

# Top 10 by net PnL
print(f"\n  🟢 أفضل 10 عملات (حسب الصافي):")
print(f"  {'عملة':<12} {'صفقات':>6} {'WR':>7} {'صافي%':>9}")
for coin, n, wr, net in sorted(coin_ranking, key=lambda x: x[3], reverse=True)[:10]:
    print(f"  {coin:<12} {n:>6} {wr:>6.1f}% {net:>+8.1f}%")

print(f"\n  🔴 أسوأ 10 عملات:")
for coin, n, wr, net in sorted(coin_ranking, key=lambda x: x[3])[:10]:
    print(f"  {coin:<12} {n:>6} {wr:>6.1f}% {net:>+8.1f}%")

# Summary stats
all_wr = [c[2] for c in coin_ranking]
all_net = [c[3] for c in coin_ranking]
print(f"\n  📊 إحصائيات الـ {len(coin_ranking)} عملة:")
print(f"  متوسط WR: {np.mean(all_wr):.1f}% | وسيط: {np.median(all_wr):.1f}%")
print(f"  عملات رابحة (صافي>0): {sum(1 for n in all_net if n>0)}/{len(all_net)} ({sum(1 for n in all_net if n>0)/len(all_net)*100:.0f}%)")
