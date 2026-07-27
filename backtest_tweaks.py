#!/usr/bin/env python3
"""3 تعديلات: فلتر ساعات + spike أعلى + تبريد بعد خسارة"""
import json, numpy as np, pandas as pd, os, time, gc

COMM = 0.20; TF_MIN = 3; CAPITAL = 1000; MAX_POS = 2
DATA_DIR = '/data/trading28/data/3m_4months'
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# Base: TP1.3 SL0.5 PL12 TR0.02 MH4 WHALE≥0.10 RSI<35 تأكيد بدون تباعد
# + متغيرات جديدة
CONFIGS = [
    # المرجع
    ("REF_الأساسي",        1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, False, 0, 1.5),
    # Spike أعلى
    ("SPIKE_2.0x",        1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, False, 0, 2.0),
    ("SPIKE_2.5x",        1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, False, 0, 2.5),
    # تبريد 30د / 60د بعد خسارة
    ("COOL_15m",          1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, False, 15, 1.5),
    ("COOL_30m",          1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, False, 30, 1.5),
    ("COOL_60m",          1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, False, 60, 1.5),
    # فلتر ساعات (منع 00-06 UTC)
    ("HOUR_block_0to6",   1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, True,  0, 1.5),
    # مركب: أفضل التعديلات
    ("COMBO_SP2_COOL30",  1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, False, 30, 2.0),
    ("COMBO_HOUR_SP2",    1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, True,  0, 2.0),
    ("COMBO_ALL",         1.3, 0.5, 12, 0.02, 4, 0.10, 35, True, True,  30, 2.0),
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

def find_signals(df, whale_min, rsi_max, confirm, spike_min, block_hours_0to6):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    whale=df['whale'].values; spike=df['spike'].values; rsi=df['rsi'].values
    mask=(whale>=whale_min)&(spike>=spike_min)&(rsi<rsi_max)&~np.isnan(whale)&~np.isnan(spike)&~np.isnan(rsi)
    mask[:50]=False
    if confirm:
        ng=np.zeros(n,dtype=bool); ng[:-1]=df['close'].values[1:]>df['open'].values[1:]; mask&=ng
    if block_hours_0to6:
        hours = pd.to_datetime(df['ts'].values, unit='ms').hour
        mask &= ~((hours >= 0) & (hours < 6))
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
    
    for name,tp,sl,pl,trail,mh,whale,rsi,confirm,block_hours,cool_min,spike_min in CONFIGS:
        idxs=find_signals(df,whale,rsi,confirm,spike_min,block_hours)
        if len(idxs)==0: continue
        max_bars=int(mh*60/TF_MIN); tp_r=1+tp/100; sl_r=1-sl/100; tr_r=1-trail/100
        active=[]; sig_map=dict(zip(idxs,close_arr[idxs]))
        for i in range(len(df)):
            cur=close_arr[i]
            if i in sig_map:
                active.append({'s':coin,'e':sig_map[i],'tp':sig_map[i]*tp_r,'sl':sig_map[i]*sl_r,
                    'pok':False,'pk':sig_map[i],'tr':sig_map[i],'ei':i,'en':int(ts_arr[i])})
            for j in range(len(active)-1,-1,-1):
                p=active[j]; e=p['e']; bh=i-p['ei']
                if bh>=max_bars:
                    p['pnl']=round((cur/e-1)*100-COMM,4); p['xt']='TIME'; p['xn']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                elif cur>=p['tp']:
                    p['pnl']=round(tp-COMM,4); p['xt']='TP'; p['xn']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                elif cur<=p['sl']:
                    p['pnl']=round(-sl-COMM,4); p['xt']='SL'; p['xn']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                elif p['pok']:
                    if cur>p['pk']: p['pk']=cur; p['tr']=cur*tr_r
                    if cur<=p['tr']:
                        p['pnl']=round((p['tr']/e-1)*100-COMM,4); p['xt']='TRAIL'; p['xn']=int(ts_arr[i]); all_trades[name].append(p); del active[j]
                else:
                    pl_p=e+(p['tp']-e)*(pl/100)
                    if cur>=pl_p: p['pok']=True; p['pk']=cur; p['tr']=cur*tr_r
    
    del df; gc.collect(); processed+=1
    if processed%50==0: print(f"  ⏳ {processed}/{len(COINS)} | {time.time()-t0:.0f}s", flush=True)

print(f"✅ {time.time()-t0:.0f}s\n", flush=True)

# ═══════════════ محاكاة ═══════════════
print(f"{'='*85}")
print(f"📊 9 تجارب — تعديلات إضافية | MAX_POS=2 | 50%")
print(f"{'='*85}")
print(f"  {'التجربة':<25} {'إشارات':>7} {'✅نفذ':>6} {'WR':>7} {'R:R':>6} {'🟢':>6} {'🔴':>6} {'ثابت$':>9} {'سحب':>6}")
print(f"  {'─'*25} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*9} {'─'*6}")

results = []
for name,tp,sl,pl,trail,mh,whale,rsi,confirm,block_hours,cool_min,spike_min in CONFIGS:
    trades=all_trades[name]
    trades.sort(key=lambda t:t['en'])
    equity=float(CAPITAL); peak=float(CAPITAL); max_dd=0.0
    slots=[None]*MAX_POS; executed=0; skipped=0; executed_pnls=[]
    
    last_loss_time = 0  # for cooldown
    
    for t in trades:
        en=t['en']; xn=t['xn']; pnl_pct=t['pnl']
        
        # Cooldown filter: skip if within cooldown after a loss
        if cool_min > 0 and last_loss_time > 0:
            if (en - last_loss_time) < cool_min * 60 * 1e9:
                continue  # skip this signal
        
        for s in range(MAX_POS):
            if slots[s] is not None:
                sex,spnl=slots[s]
                if sex<=en:
                    equity+=equity*0.5*(spnl/100); slots[s]=None
                    if spnl <= 0:
                        last_loss_time = sex  # track last loss
                    if equity>peak: peak=equity
                    dd=(equity-peak)/peak*100
                    if dd<max_dd: max_dd=dd
        free=-1
        for s in range(MAX_POS):
            if slots[s] is None: free=s; break
        if free==-1: skipped+=1; continue
        executed+=1; executed_pnls.append(pnl_pct)
        slots[free]=(xn,pnl_pct)
    
    for s in range(MAX_POS):
        if slots[s] is not None:
            sex,spnl=slots[s]; equity+=equity*0.5*(spnl/100)
    
    if equity>peak: peak=equity
    dd=(equity-peak)/peak*100
    if dd<max_dd: max_dd=dd
    
    wins=sum(1 for p in executed_pnls if p>0); losses=len(executed_pnls)-wins
    wr=wins/len(executed_pnls)*100 if executed_pnls else 0
    aw=np.mean([p for p in executed_pnls if p>0]) if wins else 0
    al=np.mean([p for p in executed_pnls if p<=0]) if losses else 0
    rr=aw/abs(al) if al!=0 else 0
    
    fixed_pnl = sum(p/100*500 for p in executed_pnls)
    
    results.append({
        'name':name,'signals':len(trades),'exec':executed,
        'wins':wins,'losses':losses,'wr':wr,'rr':rr,
        'aw':aw,'al':al,'dd':max_dd,'fixed':1000+fixed_pnl,'equity':equity,
    })

# Sort by WR
for r in sorted(results, key=lambda x: x['wr'], reverse=True):
    print(f"  {r['name']:<25} {r['signals']:>7,} {r['exec']:>6,} {r['wr']:>6.1f}% {r['rr']:>5.2f}x {r['aw']:>+5.2f}% {r['al']:>+5.2f}% ${r['fixed']:>8,.0f} {r['dd']:>5.1f}%")
