#!/usr/bin/env python3
"""تحليل الساعات + تبريد بعد خسارتين - fixed"""
import json, numpy as np, os, time, gc, pandas as pd
from collections import defaultdict

COMM=0.20; TF_MIN=3; MAX_POS=2
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
TP=1.3; SL=0.5; PL=12; TRAIL=0.02; MAX_H=4; WHALE_MIN=0.10; RSI_MAX=35

def ci(df):
    df['low_lc']=df['low'].rolling(2).min(); df['low_sm']=df['low_lc'].rolling(3).min()
    df['low_hi']=df['low_sm'].rolling(5).min(); df['low_raw']=df['low_hi'].rolling(7).min()
    w=(df['low'].values-df['low_raw'].values)/np.where(df['low_raw'].values!=0,df['low_raw'].values,np.nan)*100
    df['whale']=np.clip(w,0,None)
    vm=df['volume'].rolling(20).mean().values; df['spike']=df['volume'].values/np.where(vm!=0,vm,np.nan)
    delta=df['close'].diff().values
    gain=pd.Series(np.where(delta>0,delta,0)).rolling(14).mean().values
    loss=pd.Series(np.where(delta<0,-delta,0)).rolling(14).mean().values
    df['rsi']=100-100/(1+gain/np.where(loss!=0,loss,np.nan))
    return df

def fs(df):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    wh=df['whale'].values; sp=df['spike'].values; rs=df['rsi'].values
    mask=(wh>=WHALE_MIN)&(sp>=1.5)&(rs<RSI_MAX)&~np.isnan(wh)&~np.isnan(sp)&~np.isnan(rs)
    mask[:50]=False
    hp=np.zeros(n,dtype=bool)
    for sh in[1,2,3]: s=np.zeros(n,dtype=bool); s[sh:]=mask[:-sh]; hp|=s
    mask&=~hp
    ng=np.zeros(n,dtype=bool); ng[:-1]=df['close'].values[1:]>df['open'].values[1:]; mask&=ng
    return np.where(mask)[0]

with open('/data/trading28/config/shariah_coins.json') as f: shariah=json.load(f)
COINS=[c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
EX={'ETH','BTC','TRX','XRP','QI','LSK','GLMR','XTZ','YFI',
    'TLM','0G','LA','DYM','VANRY','SENT','VET','COOKIE','HEI','ACT','CKB','AR','RSR','AXS','XEC',
    'LPT','KNC','LTC','SFP','IOST','KAVA','VTHO','ZRX','1INCH','CVX','WAXP','ZIL','VANA','YGG','SUI','STEEM'}
COINS=[c for c in COINS if c not in EX]

print(f"⏳ جمع الصفقات ({len(COINS)} عملة)...", flush=True)
all_trades=[]
processed=0; t0=time.time()

for coin in COINS:
    fp=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw; df=ci(df)
    ca=df['close'].values; ta=df['ts'].values  # keep as ms timestamps
    ts_ms = df['ts'].values  # milliseconds
    idxs=fs(df)
    if len(idxs)==0: del df; continue
    mb=int(MAX_H*60/TF_MIN); tpr=1+TP/100; slr=1-SL/100; trr=1-TRAIL/100
    active=[]; sm=dict(zip(idxs,ca[idxs]))
    for i in range(len(df)):
        cur=ca[i]
        if i in sm: active.append({'symbol':coin,'entry':sm[i],'tp':sm[i]*tpr,'sl':sm[i]*slr,
            'pl_ok':False,'peak':sm[i],'trail':sm[i],'entry_i':i,'entry_ms':int(ts_ms[i]),
            'entry_hour':int(pd.to_datetime(ts_ms[i],unit='ms').hour)})
        for j in range(len(active)-1,-1,-1):
            p=active[j]; e=p['entry']; bh=i-p['entry_i']
            if bh>=mb:
                p['pnl']=round((cur/e-1)*100-COMM,4); p['exit_type']='TIME'
                p['exit_ms']=int(ts_ms[i]); p['exit_hour']=int(pd.to_datetime(ts_ms[i],unit='ms').hour)
                all_trades.append(p); del active[j]
            elif cur>=p['tp']:
                p['pnl']=round(TP-COMM,4); p['exit_type']='TP'
                p['exit_ms']=int(ts_ms[i]); p['exit_hour']=int(pd.to_datetime(ts_ms[i],unit='ms').hour)
                all_trades.append(p); del active[j]
            elif cur<=p['sl']:
                p['pnl']=round(-SL-COMM,4); p['exit_type']='SL'
                p['exit_ms']=int(ts_ms[i]); p['exit_hour']=int(pd.to_datetime(ts_ms[i],unit='ms').hour)
                all_trades.append(p); del active[j]
            elif p['pl_ok']:
                if cur>p['peak']: p['peak']=cur; p['trail']=cur*trr
                if cur<=p['trail']:
                    p['pnl']=round((p['trail']/e-1)*100-COMM,4); p['exit_type']='TRAIL'
                    p['exit_ms']=int(ts_ms[i]); p['exit_hour']=int(pd.to_datetime(ts_ms[i],unit='ms').hour)
                    all_trades.append(p); del active[j]
            else:
                pl_p=e+(p['tp']-e)*(PL/100)
                if cur>=pl_p: p['pl_ok']=True; p['peak']=cur; p['trail']=cur*trr
    del df; gc.collect(); processed+=1
    if processed%50==0: print(f"  ⏳ {processed}/{len(COINS)}", flush=True)

print(f"✅ {time.time()-t0:.0f}s | {len(all_trades):,} signals\n")
all_trades.sort(key=lambda t:t['entry_ms'])

# ═══════════════ تحليل الساعات ═══════════════
def simulate(trades, cooldown_min=0):
    active_slots=[None,None]
    executed=[]
    consecutive_losses=0
    last_loss_ms=0
    
    for t in trades:
        en=t['entry_ms']; ex=t['exit_ms']; pp=t['pnl']
        
        # Cooldown check
        if cooldown_min > 0 and consecutive_losses >= 2:
            if en < last_loss_ms + cooldown_min * 60 * 1000:
                continue
            else:
                consecutive_losses = 0
        
        # Free completed slots
        for s in range(2):
            if active_slots[s] and active_slots[s][0]<=en:
                if active_slots[s][1] <= 0:
                    consecutive_losses += 1
                    last_loss_ms = active_slots[s][0]
                else:
                    consecutive_losses = 0
                active_slots[s]=None
        
        free=-1
        for s in range(2):
            if not active_slots[s]: free=s; break
        if free==-1: continue
        
        active_slots[free]=(ex,pp)
        executed.append(t)
    
    return executed

executed = simulate(all_trades, 0)

# Hour analysis
hour_stats = defaultdict(lambda: {'wins':0,'losses':0,'pnl_sum':0.0,'count':0})
for t in executed:
    h = t['entry_hour']
    hour_stats[h]['count']+=1; hour_stats[h]['pnl_sum']+=t['pnl']
    if t['pnl']>0: hour_stats[h]['wins']+=1
    else: hour_stats[h]['losses']+=1

print("="*70)
print("📊 تحليل حسب ساعات اليوم (UTC) — 172 عملة — MAX_POS=2")
print("="*70)
print(f"{'ساعة':>5s} | {'صفقات':>6s} | {'ربح':>5s} {'خسارة':>5s} | {'WR':>6s} | {'مجموع%':>9s} | {'متوسط%':>9s}")
print("-"*70)

good_hours = []; bad_hours = []
for h in range(24):
    s = hour_stats[h]
    if s['count'] == 0: continue
    wr = s['wins']/s['count']*100
    avg = s['pnl_sum']/s['count']
    bar = '█' * int(s['count']/200)
    print(f"  {h:02d}:00 | {s['count']:5d}  | {s['wins']:4d} {s['losses']:4d} | {wr:5.1f}% | {s['pnl_sum']:+9.2f} | {avg:+9.4f}  {bar}")
    if s['count'] >= 50:
        if avg > 0.15: good_hours.append(h)
        elif avg < 0.05: bad_hours.append(h)

print(f"\n✅ ساعات ممتازة (متوسط>0.15%): {good_hours}")
print(f"❌ ساعات ضعيفة (متوسط<0.05%): {bad_hours}")

# ═══════════════ تبريد ═══════════════
print("\n" + "="*70)
print("📊 تجربة التبريد بعد خسارتين متتاليتين")
print("="*70)

bl_wins = sum(1 for t in executed if t['pnl']>0)
bl_wr = bl_wins/len(executed)*100
bl_pnl = sum(t['pnl'] for t in executed)
print(f"\nبدون تبريد: {len(executed):,} ص | WR {bl_wr:.1f}% | مجموع {bl_pnl:+.1f}%")

print(f"\n{'تبريد':>8s} | {'صفقات':>6s} | {'WR':>6s} | {'مجموع%':>9s} | {'محذوفة':>7s}")
print("-"*50)

for cd in [15, 30, 45, 60, 90, 120]:
    ex2 = simulate(all_trades, cd)
    if len(ex2)==0: continue
    wr = sum(1 for t in ex2 if t['pnl']>0)/len(ex2)*100
    total = sum(t['pnl'] for t in ex2)
    skipped = len(executed) - len(ex2)
    print(f"  {cd:3d} دقيقة | {len(ex2):5d}  | {wr:5.1f}% | {total:+9.2f} | {skipped:5d}")
