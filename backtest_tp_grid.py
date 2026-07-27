#!/usr/bin/env python3
"""TP صغير — شبكة تجارب موسعة"""
import json, numpy as np, pandas as pd, os, time, gc

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000; MAX_POS = 2
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

CONFIGS = [
    # TP حول 0.6% مع SL متنوع
    ("TP0.4_SL0.3", 0.4, 0.3, 8,  0.05, 4, 0.10, 35, True),
    ("TP0.4_SL0.5", 0.4, 0.5, 10, 0.05, 4, 0.10, 35, True),
    ("TP0.5_SL0.3", 0.5, 0.3, 8,  0.05, 4, 0.10, 35, True),
    ("TP0.5_SL0.5", 0.5, 0.5, 10, 0.05, 4, 0.10, 35, True),
    ("TP0.5_SL0.7", 0.5, 0.7, 12, 0.05, 4, 0.10, 35, True),
    ("TP0.6_SL0.3", 0.6, 0.3, 8,  0.05, 4, 0.10, 35, True),
    ("TP0.6_SL0.4", 0.6, 0.4, 10, 0.05, 4, 0.10, 35, True),
    ("TP0.6_SL0.7", 0.6, 0.7, 12, 0.05, 4, 0.10, 35, True),
    ("TP0.7_SL0.5", 0.7, 0.5, 10, 0.05, 4, 0.10, 35, True),
    ("TP0.8_SL0.5", 0.8, 0.5, 12, 0.05, 4, 0.10, 35, True),
    # تريل مختلف
    ("TP0.6_SL0.5_TR3",  0.6, 0.5, 10, 0.03, 4, 0.10, 35, True),
    ("TP0.6_SL0.5_TR7",  0.6, 0.5, 10, 0.07, 4, 0.10, 35, True),
    # PL مختلف
    ("TP0.6_SL0.5_PL8",  0.6, 0.5, 8,  0.05, 4, 0.10, 35, True),
    ("TP0.6_SL0.5_PL12", 0.6, 0.5, 12, 0.05, 4, 0.10, 35, True),
    # MH مختلف
    ("TP0.6_SL0.5_MH3",  0.6, 0.5, 10, 0.05, 3, 0.10, 35, True),
    ("TP0.6_SL0.5_MH6",  0.6, 0.5, 10, 0.05, 6, 0.10, 35, True),
    # فلتر أقوى
    ("TP0.6_SL0.5_RSI30", 0.6, 0.5, 10, 0.05, 4, 0.10, 30, True),
    ("TP0.6_SL0.5_حوت15", 0.6, 0.5, 10, 0.05, 4, 0.15, 35, True),
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

def find_signals(df, whale_min, rsi_max, confirm):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    whale=df['whale'].values; spike=df['spike'].values; rsi=df['rsi'].values
    mask=(whale>=whale_min)&(spike>=1.5)&(rsi<rsi_max)&~np.isnan(whale)&~np.isnan(spike)&~np.isnan(rsi)
    mask[:50]=False
    has_prev=np.zeros(n,dtype=bool)
    for shift in[1,2,3]:
        s=np.zeros(n,dtype=bool); s[shift:]=mask[:-shift]; has_prev|=s
    mask&=~has_prev
    if confirm:
        ng=np.zeros(n,dtype=bool); ng[:-1]=df['close'].values[1:]>df['open'].values[1:]
        mask&=ng
    return np.where(mask)[0]

print("⏳ جمع الصفقات...", flush=True)
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]

all_trades = {c[0]: [] for c in CONFIGS}
processed=0; t0=time.time()

for coin in COINS:
    fpath=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath): continue
    with open(fpath) as f: raw=json.load(f)
    if len(raw)<200: del raw; continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw
    df=compute_indicators(df)
    close_arr=df['close'].values; ts_arr=df['ts'].values.astype('datetime64[ns]').astype('int64')
    
    for name,tp,sl,pl,trail,mh,whale,rsi,confirm in CONFIGS:
        idxs=find_signals(df,whale,rsi,confirm)
        if len(idxs)==0: continue
        max_bars=int(mh*60/TF_MIN); tp_r=1+tp/100; sl_r=1-sl/100; tr_r=1-trail/100
        active=[]; sig_map=dict(zip(idxs,close_arr[idxs]))
        for i in range(len(df)):
            cur=close_arr[i]
            if i in sig_map:
                active.append({'symbol':coin,'entry':sig_map[i],'tp':sig_map[i]*tp_r,'sl':sig_map[i]*sl_r,
                    'pl_ok':False,'peak':sig_map[i],'trail':sig_map[i],'entry_i':i,'entry_ns':int(ts_arr[i])})
            for j in range(len(active)-1,-1,-1):
                p=active[j]; e=p['entry']; bh=i-p['entry_i']
                if bh>=max_bars:
                    p['pnl']=round((cur/e-1)*100-COMM,4); p['exit_type']='TIME'; p['exit_ns']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                elif cur>=p['tp']:
                    p['pnl']=round(tp-COMM,4); p['exit_type']='TP'; p['exit_ns']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                elif cur<=p['sl']:
                    p['pnl']=round(-sl-COMM,4); p['exit_type']='SL'; p['exit_ns']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                elif p['pl_ok']:
                    if cur>p['peak']: p['peak']=cur; p['trail']=cur*tr_r
                    if cur<=p['trail']:
                        p['pnl']=round((p['trail']/e-1)*100-COMM,4); p['exit_type']='TRAIL'; p['exit_ns']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                else:
                    pl_p=e+(p['tp']-e)*(pl/100)
                    if cur>=pl_p: p['pl_ok']=True; p['peak']=cur; p['trail']=cur*tr_r
    
    del df; gc.collect(); processed+=1
    if processed%50==0: print(f"  ⏳ {processed}/{len(COINS)} | {time.time()-t0:.0f}s", flush=True)

print(f"✅ {time.time()-t0:.0f}s\n", flush=True)

# ═══════════════ محاكاة ═══════════════
print(f"{'='*70}")
print(f"📊 18 تجربة — TP صغير مع فلاتر مركب | MAX_POS=2 عالمي")
print(f"{'='*70}")

results = []
for name,tp,sl,pl,trail,mh,whale,rsi,confirm in CONFIGS:
    trades=all_trades[name]
    trades.sort(key=lambda t:t['entry_ns'])
    equity=float(CAPITAL); peak=float(CAPITAL); max_dd=0.0
    slots=[None,None]; executed=0; skipped=0
    
    for t in trades:
        en=t['entry_ns']; ex=t['exit_ns']; pnl_pct=t['pnl']
        for s in range(MAX_POS):
            if slots[s] is not None:
                sex,spnl=slots[s]
                if sex<=en:
                    pos_cap=equity*0.5; pnl_dollar=pos_cap*(spnl/100); equity+=pnl_dollar; slots[s]=None
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
            sex,spnl=slots[s]; pos_cap=equity*0.5; pnl_dollar=pos_cap*(spnl/100); equity+=pnl_dollar
    
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
    
    results.append({
        'name':name,'tp':tp,'sl':sl,'pl':pl,'trail':trail,'mh':mh,
        'signals':len(trades),'exec':executed,'skip':skipped,
        'wins':wins,'losses':losses,'wr':wr,'rr':rr,
        'avg_win':avg_win,'avg_loss':avg_loss,'dd':max_dd,
        'equity':equity,'tp_c':tp_c,'sl_c':sl_c,'tr_c':tr_c,'tm_c':tm_c,
    })

# Sort by WR
sorted_r = sorted(results, key=lambda x: x['wr'], reverse=True)

print(f"\n{'─'*80}")
print(f"  {'التجربة':<22} {'إشارات':>6} {'✅نفذ':>6} {'⏭️تخطى':>6} {'WR':>7} {'R:R':>6} {'محفظة':>9} {'DD':>6}")
print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*9} {'─'*6}")
for r in sorted_r:
    print(f"  {r['name']:<22} {r['signals']:>6,} {r['exec']:>6,} {r['skip']:>6,} {r['wr']:>6.1f}% {r['rr']:>5.2f}x ${r['equity']:>8,.0f} {r['dd']:>5.1f}%")

# Now highlight best by portfolio
print(f"\n{'─'*80}")
print(f"  ⚖️ مرتب حسب المحفظة:")
print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*9} {'─'*6}")
for r in sorted(results, key=lambda x: x['equity'], reverse=True):
    print(f"  {r['name']:<22} {r['signals']:>6,} {r['exec']:>6,} {r['skip']:>6,} {r['wr']:>6.1f}% {r['rr']:>5.2f}x ${r['equity']:>8,.0f} {r['dd']:>5.1f}%")
