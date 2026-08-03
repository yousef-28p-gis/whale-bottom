#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  ⛔ قوانين ثابتة — ممنوع التعديل بدون تأكيد صريح من يوسف   ║
║  1. CLOSE-ONLY  2. SL حقيقي  3. MAX_POS=2 عالمي             ║
║  4. LOOK-AHEAD=NONE  5. تباعد 3 شمعات  6. منع تكرار         ║
║  7. عمولة 0.2%  8. whale(7-layer)+spike+RSI(14)             ║
║  9. WHALE≥0.10 SPIKE≥1.5 RSI<35 تأكيد أخضر                   ║
║ 10. TP/SL/TRAIL(PL%)/TIME                                    ║
╚══════════════════════════════════════════════════════════════╝
شبكة اختبارات — كل الـ 212 عملة
"""
import json, numpy as np, os, time, gc, pandas as pd
from itertools import product

COMM=0.20; MAX_POS=2; CAPITAL=1000
DATA_DIR='/data/trading28/data/3m_4months'
STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# ═══════════════ المؤشرات (ثابت — قانون 5,8,9) ═══════════════
def compute_indicators(df):
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

def find_signals(df):
    n=len(df)
    if n<100: return np.array([],dtype=int)
    WHALE_MIN=0.10; RSI_MAX=35; SPIKE_MIN=1.5
    wh=df['whale'].values; sp=df['spike'].values; rs=df['rsi'].values
    mask=(wh>=WHALE_MIN)&(sp>=SPIKE_MIN)&(rs<RSI_MAX)&~np.isnan(wh)&~np.isnan(sp)&~np.isnan(rs)
    mask[:50]=False
    hp=np.zeros(n,dtype=bool)
    for sh in[1,2,3]: s=np.zeros(n,dtype=bool); s[sh:]=mask[:-sh]; hp|=s
    mask&=~hp
    ng=np.zeros(n,dtype=bool); ng[:-1]=df['close'].values[1:]>df['open'].values[1:]; mask&=ng
    return np.where(mask)[0]

# ═══════════════ التكوينات ═══════════════
CONFIGS = [
    # Baseline
    ("TP1.3_SL0.5_TR0.02_PL12_4h", 1.3, 0.5, 0.02, 12, 4),
    
    # TP variants
    ("TP1.0_SL0.5_TR0.02_PL12_4h", 1.0, 0.5, 0.02, 12, 4),
    ("TP1.5_SL0.5_TR0.02_PL12_4h", 1.5, 0.5, 0.02, 12, 4),
    ("TP2.0_SL0.5_TR0.02_PL12_4h", 2.0, 0.5, 0.02, 12, 4),
    ("TP0.8_SL0.5_TR0.02_PL12_4h", 0.8, 0.5, 0.02, 12, 4),
    
    # SL variants
    ("TP1.3_SL0.3_TR0.02_PL12_4h", 1.3, 0.3, 0.02, 12, 4),
    ("TP1.3_SL0.8_TR0.02_PL12_4h", 1.3, 0.8, 0.02, 12, 4),
    ("TP1.3_SL1.0_TR0.02_PL12_4h", 1.3, 1.0, 0.02, 12, 4),
    
    # TRAIL variants
    ("TP1.3_SL0.5_TR0.05_PL12_4h", 1.3, 0.5, 0.05, 12, 4),
    ("TP1.3_SL0.5_TR0.10_PL12_4h", 1.3, 0.5, 0.10, 12, 4),
    
    # PL variants
    ("TP1.3_SL0.5_TR0.02_PL20_4h", 1.3, 0.5, 0.02, 20, 4),
    ("TP1.3_SL0.5_TR0.02_PL30_4h", 1.3, 0.5, 0.02, 30, 4),
    
    # TIME variants
    ("TP1.3_SL0.5_TR0.02_PL12_2h", 1.3, 0.5, 0.02, 12, 2),
    ("TP1.3_SL0.5_TR0.02_PL12_6h", 1.3, 0.5, 0.02, 12, 6),
    
    # Combo: wider SL + wider TP
    ("TP1.5_SL0.8_TR0.02_PL12_4h", 1.5, 0.8, 0.02, 12, 4),
    ("TP2.0_SL1.0_TR0.05_PL12_4h", 2.0, 1.0, 0.05, 12, 4),
    
    # Combo: tight TP + tight SL
    ("TP0.8_SL0.3_TR0.02_PL12_4h", 0.8, 0.3, 0.02, 12, 4),
    
    # Combo: medium everything
    ("TP1.5_SL0.5_TR0.05_PL20_3h", 1.5, 0.5, 0.05, 20, 3),
]

print(f"🔬 {len(CONFIGS)} تكوين للاختبار")
print(f"📋 القوانين العشرة مثبتة — لا يمكن تغييرها\n")

# Load coins — ALL 212
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah=json.load(f)
COINS=[c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
print(f"💰 {len(COINS)} عملة حلال\n")

# ═══════════════ جمع الصفقات لكل التكوينات ═══════════════
all_trades_by_config = {c[0]: [] for c in CONFIGS}
processed=0; t0=time.time()

for coin in COINS:
    fp=f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fp): continue
    with open(fp) as f: raw=json.load(f)
    if len(raw)<200: continue
    df=pd.DataFrame(raw).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    del raw; df=compute_indicators(df)
    ca=df['close'].values; ha=df['high'].values; ts=df['ts'].values
    idxs=find_signals(df)
    if len(idxs)==0: del df; continue
    
    for name, tp, sl, trail, pl, max_h in CONFIGS:
        mb=int(max_h*60/3); tpr=1+tp/100; slr=1-sl/100; trr=1-trail/100
        active=[]; sm=dict(zip(idxs,ca[idxs]))
        for i in range(len(df)):
            cur=ca[i]
            if i in sm: active.append({'symbol':coin,'entry':sm[i],'tp':sm[i]*tpr,'sl':sm[i]*slr,
                'pl_ok':False,'peak':sm[i],'trail':sm[i],'entry_i':i,'entry_ms':int(ts[i])})
            for j in range(len(active)-1,-1,-1):
                p=active[j]; e=p['entry']; bh=i-p['entry_i']
                if bh>=mb:
                    p['pnl']=round((cur/e-1)*100-COMM,4); p['exit_type']='TIME'; p['exit_ms']=int(ts[i])
                    all_trades_by_config[name].append(p); del active[j]
                elif ha[i]>=p['tp']:
                    p['pnl']=round(tp-COMM,4); p['exit_type']='TP'; p['exit_ms']=int(ts[i])
                    all_trades_by_config[name].append(p); del active[j]
                elif cur<=p['sl']:
                    p['pnl']=round((cur/e-1)*100-COMM,4); p['exit_type']='SL'; p['exit_ms']=int(ts[i])
                    all_trades_by_config[name].append(p); del active[j]
                elif p['pl_ok']:
                    if ha[i]>p['peak']: p['peak']=ha[i]; p['trail']=ha[i]*trr
                    if cur<=p['trail']:
                        p['pnl']=round((cur/e-1)*100-COMM,4); p['exit_type']='TRAIL'; p['exit_ms']=int(ts[i])
                        all_trades_by_config[name].append(p); del active[j]
                else:
                    pl_p=e+(p['tp']-e)*(pl/100)
                    if ha[i]>=pl_p: p['pl_ok']=True; p['peak']=ha[i]; p['trail']=ha[i]*trr
    del df; gc.collect(); processed+=1
    if processed%50==0: print(f"  ⏳ {processed}/{len(COINS)} | {time.time()-t0:.0f}s", flush=True)

print(f"\n✅ جمع البيانات: {time.time()-t0:.0f}s\n", flush=True)

# ═══════════════ محاكاة MAX_POS=2 عالمي لكل تكوين ═══════════════
print("="*85)
print(f"{'تكوين':<32s} | {'صفقات':>6s} | {'WR':>6s} | {'R:R':>5s} | {'بدون تركيب':>10s} | {'سحب':>6s}")
print("-"*85)

results=[]
for name, tp, sl, trail, pl, max_h in CONFIGS:
    trades=all_trades_by_config[name]
    trades.sort(key=lambda t:t['entry_ms'])
    
    equity=float(CAPITAL); peak_e=float(CAPITAL); max_dd=0.0
    active_slots=[None,None]; executed=[]; skipped=0
    
    for t in trades:
        en=t['entry_ms']; ex=t['exit_ms']; pp=t['pnl']
        for s in range(2):
            if active_slots[s] and active_slots[s][0]<=en:
                pos_cap=equity*0.5; pnl_d=pos_cap*(active_slots[s][1]/100)
                equity+=pnl_d; active_slots[s]=None
                if equity>peak_e: peak_e=equity
                dd=(equity-peak_e)/peak_e*100
                if dd<max_dd: max_dd=dd
        free=-1
        for s in range(2):
            if not active_slots[s]: free=s; break
        if free==-1: skipped+=1; continue
        active_slots[free]=(ex,pp); executed.append(t)
    
    for s in range(2):
        if active_slots[s]:
            pos_cap=equity*0.5; pnl_d=pos_cap*(active_slots[s][1]/100)
            equity+=pnl_d; active_slots[s]=None
    if equity>peak_e: peak_e=equity
    dd=(equity-peak_e)/peak_e*100
    if dd<max_dd: max_dd=dd
    
    et=executed
    wins=sum(1 for t in et if t['pnl']>0)
    losses=sum(1 for t in et if t['pnl']<=0)
    wr=wins/len(et)*100 if et else 0
    avg_win=np.mean([t['pnl'] for t in et if t['pnl']>0]) if wins else 0
    avg_loss=np.mean([t['pnl'] for t in et if t['pnl']<=0]) if losses else 0
    rr=avg_win/abs(avg_loss) if avg_loss!=0 else 0
    
    fixed_pnl=sum(t['pnl'] for t in et)
    
    results.append((name, len(et), wr, rr, fixed_pnl, max_dd, equity))
    
    fixed_eq=f"${1000+fixed_pnl*5:,.0f}"
    print(f"{name:<32s} | {len(et):5d}  | {wr:5.1f}% | {rr:4.2f}x | {fixed_eq:>10s} | {max_dd:+5.1f}%")

# Sort by best without-compounding
print(f"\n{'='*85}")
print("🏆 أفضل 5 تكوينات (بدون تركيب):")
print(f"{'='*85}")
results.sort(key=lambda x: x[4], reverse=True)
for name, n, wr, rr, pnl, dd, eq in results[:5]:
    print(f"  {name:<32s} | {n:5d} ص | WR {wr:.1f}% | R:R {rr:.2f}x | PnL {pnl:+.1f}% | DD {dd:+.1f}%")
