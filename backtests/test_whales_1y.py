#!/usr/bin/env python3
"""اختبار استراتيجيات الحوت على سنة كاملة — 198 عملة 15m"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/data/whale_15m_1y'
COMM = 0.2; CAP = 1000; POS_PCT = 50

# Load
print("📦 تحميل سنة كاملة...")
all_data = {}
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.json') or fname.startswith('_'): continue
    with open(f'{DATA_DIR}/{fname}') as f: raw = json.load(f)
    all_data[fname.replace('.json','')] = {
        'c': np.array(raw['c']), 'h': np.array(raw['h']),
        'l': np.array(raw['l']), 'v': np.array(raw['v']),
        'ts': [datetime.fromtimestamp(t/1000, tz=timezone.utc) for t in raw['ts']],
    }

n_coins = len(all_data)
total_bars = sum(len(d['c']) for d in all_data.values())
first_coin = list(all_data.keys())[0]
d0 = all_data[first_coin]
print(f"  {n_coins} عملة | {total_bars:,} شمعة")
print(f"  {d0['ts'][0].date()} → {d0['ts'][-1].date()}")
print(f"  مثال {first_coin}: {d0['c'][0]:.4f} → {d0['c'][-1]:.4f}\n")

# ═══════ 🐋 حوت القاع (نسخة مطابقة للدايمون) ═══════
def compute_whale(df):
    df = df.copy(); LB = 30
    df['lo'] = df['low'].rolling(LB).min()
    df['lc'] = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
    df['sm'] = df['lc'].ewm(span=3, adjust=False).mean()
    df['hi'] = df['sm'].rolling(LB).max()
    df['raw'] = np.where(df['low'] <= df['lo'], (df['sm'] + df['hi'] * 2) / 3, 0)
    df['whale'] = df['raw'].ewm(span=3, adjust=False).mean().fillna(0)
    df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.03)
    df['wf'] = df['whale'].rolling(2).mean(); df['ws'] = df['whale'].rolling(5).mean()
    df['wp'] = df['whale'].rolling(50).max()
    df['str'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
    df['vma'] = df['volume'].rolling(20).mean()
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) & (df['str'] > 50) & (df['volume'] > df['vma'] * 1.0))
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    return df

def run_whale_bottom():
    TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6
    WHALE_MIN=0.50; RSI_MAX=25
    BLOCK_HOURS={1,3,6,12,0,4}; BLOCK_WDAY=3
    trades=[]
    done=0
    for coin, d in all_data.items():
        df = pd.DataFrame({'open':d['c'],'high':d['h'],'low':d['l'],'close':d['c'],'volume':d['v']})
        df.index=d['ts']; df_w=compute_whale(df); n=len(df_w)
        for i in range(100,n-2):
            row=df_w.iloc[i]
            if not row['entry']: continue
            if float(row['whale'])<WHALE_MIN: continue
            if i+1<n and float(df_w.iloc[i+1]['whale'])>=0.35: continue
            rsi=float(row['rsi'])
            if np.isnan(rsi) or rsi>=RSI_MAX: continue
            ti=df_w.index[i]
            if ti.weekday()==BLOCK_WDAY: continue
            if ti.hour in BLOCK_HOURS: continue
            ps=max(0,i-96); pb=float(df_w.iloc[ps]['close']); ep=float(row['close'])
            if (ep-pb)/pb*100>=0: continue
            if i+1>=n: continue
            if float(df_w.iloc[i+1]['close'])<=float(df_w.iloc[i+1]['open']): continue
            entry_px=float(d['c'][i+2]) if i+2<n else ep
            if i+2>=n: continue
            tp_px=entry_px*(1+TP/100); sl_px=entry_px*(1-SL/100)
            pl_px=entry_px+(tp_px-entry_px)*(PL/100)
            pl_trig=False; peak=entry_px; trail_px=0
            et='TIME'; exit_px=entry_px
            for j in range(i+3,min(i+3+MH*4,n)):
                cur=d['c'][j]; cur_h=d['h'][j]; cur_l=d['l'][j]
                if cur_l<=sl_px: et='SL'; exit_px=sl_px; break
                if cur_h>=tp_px and not pl_trig: et='TP'; exit_px=tp_px; break
                if not pl_trig and cur_h>=pl_px:
                    pl_trig=True; peak=cur_h; trail_px=cur_h*(1-TRAIL/100)
                if pl_trig:
                    if cur_h>peak: peak=cur_h; trail_px=cur_h*(1-TRAIL/100)
                    if cur_l<=trail_px: et='TRAIL'; exit_px=trail_px; break
            else: exit_px=d['c'][min(i+3+MH*4,n-1)]
            pnl=(exit_px/entry_px-1)*100-COMM
            trades.append({'pnl':pnl,'type':et,'coin':coin})
        done+=1
        if done%50==0: print(f"  🐋 {done}/{n_coins}",flush=True)
    return trades

# ═══════ 🦉 الحوت ═══════
def run_hoot():
    TP=2.0; SL=1.5; MH=48; trades=[]
    done=0
    for coin, d in all_data.items():
        c=d['c']; h_arr=d['h']; l_arr=d['l']; v=d['v']; n=len(c)
        avg_vol=pd.Series(v).rolling(20).mean().values
        rsi=np.zeros(n)
        for i in range(14,n):
            delta=np.diff(c[i-14:i+1])
            gain=np.mean(delta[delta>0]) if any(delta>0) else 0
            loss=-np.mean(delta[delta<0]) if any(delta<0) else 0.0001
            rsi[i]=100-100/(1+gain/loss)
        last_entry=-999
        for i in range(50,n-1):
            if i-last_entry<24: continue
            if v[i]<=avg_vol[i]*2.5: continue
            if rsi[i]>=35: continue
            if c[i]>=c[i-1]: continue
            if i+1>=n or c[i+1]<=c[i]: continue
            ep=c[i+1]; tp_px=ep*(1+TP/100); sl_px=ep*(1-SL/100)
            ex=et=None
            for j in range(i+2,min(i+MH,n)):
                if l_arr[j]<=sl_px: ex=sl_px; et='SL'; break
                elif h_arr[j]>=tp_px: ex=tp_px; et='TP'; break
            if not ex: ex=c[min(i+MH,n-1)]; et='TIME'
            pnl=(ex/ep-1)*100-COMM
            trades.append({'pnl':pnl,'type':et,'coin':coin})
            last_entry=i
        done+=1
        if done%50==0: print(f"  🦉 {done}/{n_coins}",flush=True)
    return trades

# ═══════ ⚡ خاطف ═══════
def run_khatef():
    TP=1.5; SL=1.0; MH=24; trades=[]
    done=0
    for coin, d in all_data.items():
        c=d['c']; h_arr=d['h']; l_arr=d['l']; v=d['v']; n=len(c)
        avg_vol=pd.Series(v).rolling(20).mean().values
        ema9=pd.Series(c).ewm(span=9).mean().values
        last_entry=-999
        for i in range(50,n-1):
            if i-last_entry<16: continue
            if v[i]<=avg_vol[i]*4.0: continue
            if c[i]>ema9[i]: continue
            if c[i]>=c[i-1]: continue
            ep=c[i]; tp_px=ep*(1+TP/100); sl_px=ep*(1-SL/100)
            ex=et=None
            for j in range(i+1,min(i+MH,n)):
                if l_arr[j]<=sl_px: ex=sl_px; et='SL'; break
                elif h_arr[j]>=tp_px: ex=tp_px; et='TP'; break
            if not ex: ex=c[min(i+MH,n-1)]; et='TIME'
            pnl=(ex/ep-1)*100-COMM
            trades.append({'pnl':pnl,'type':et,'coin':coin})
            last_entry=i
        done+=1
        if done%50==0: print(f"  ⚡ {done}/{n_coins}",flush=True)
    return trades

# ═══════ 🗡️ صياد القاع ═══════
def run_sayad():
    TP=2.5; SL=1.2; MH=48; trades=[]
    done=0
    for coin, d in all_data.items():
        c=d['c']; h_arr=d['h']; l_arr=d['l']; v=d['v']; n=len(c)
        avg_vol=pd.Series(v).rolling(20).mean().values
        ema50=pd.Series(c).ewm(span=50).mean().values
        rsi=np.zeros(n)
        for i in range(14,n):
            delta=np.diff(c[i-14:i+1])
            gain=np.mean(delta[delta>0]) if any(delta>0) else 0
            loss=-np.mean(delta[delta<0]) if any(delta<0) else 0.0001
            rsi[i]=100-100/(1+gain/loss)
        sw_l=np.zeros(n,dtype=bool)
        for i in range(5,n-5):
            if all(l_arr[i]<=l_arr[j] for j in range(i-5,i+6) if j!=i):
                sw_l[i]=True
        last_entry=-999
        for i in range(50,n-1):
            if i-last_entry<24: continue
            if rsi[i]>=40: continue
            if v[i]<=avg_vol[i]*2.0: continue
            if c[i]>=ema50[i]: continue
            near_sw=any(sw_l[j] and abs(l_arr[i]-l_arr[j])/l_arr[j]<0.02 for j in range(max(10,i-20),i-5))
            if not near_sw: continue
            if c[i]<=c[i-1]: continue
            ep=c[i]; tp_px=ep*(1+TP/100); sl_px=ep*(1-SL/100)
            ex=et=None
            for j in range(i+1,min(i+MH,n)):
                if l_arr[j]<=sl_px: ex=sl_px; et='SL'; break
                elif h_arr[j]>=tp_px: ex=tp_px; et='TP'; break
            if not ex: ex=c[min(i+MH,n-1)]; et='TIME'
            pnl=(ex/ep-1)*100-COMM
            trades.append({'pnl':pnl,'type':et,'coin':coin})
            last_entry=i
        done+=1
        if done%50==0: print(f"  🗡️ {done}/{n_coins}",flush=True)
    return trades

# ═══════════ Run ═══════════
def stats(name, trades):
    if not trades: return None
    w=[t for t in trades if t['pnl']>0]; lo=[t for t in trades if t['pnl']<=0]
    wr=len(w)/len(trades)*100
    aw=np.mean([t['pnl'] for t in w]) if w else 0
    al=np.mean([t['pnl'] for t in lo]) if lo else 0
    rr=abs(aw/al) if al!=0 else 0
    curve=[CAP]
    for t in trades: sz=curve[-1]*(POS_PCT/100); curve.append(curve[-1]+sz*t['pnl']/100)
    dd=np.min((curve-np.maximum.accumulate(curve))/np.maximum.accumulate(curve)*100)
    tp_n=sum(1 for t in trades if t['type']in('TP',))
    sl_n=sum(1 for t in trades if t['type']in('SL',))
    tr_n=sum(1 for t in trades if t['type']=='TRAIL')
    tm_n=sum(1 for t in trades if t['type']=='TIME')
    ann=((curve[-1]/CAP)**(365/365)-1)*100
    return (name,len(trades),wr,curve[-1],dd,rr,tp_n,sl_n,tr_n,tm_n,aw,al,ann)

print(f"\n🐋 اختبار سنة كاملة — {n_coins} عملة\n")
strategies = [
    ("🐋 حوت القاع", run_whale_bottom),
    ("🦉 الحوت", run_hoot),
    ("⚡ خاطف", run_khatef),
    ("🗡️ صياد القاع", run_sayad),
]

results = []
for name, fn in strategies:
    print(f"\n⏳ {name}...", flush=True)
    trades = fn()
    r = stats(name, trades)
    if r: results.append(r)

print(f"\n{'='*95}")
print(f"📊 النتائج — سنة كاملة — 15m")
print(f"{'='*95}")
print(f"{'الاستراتيجية':<20s} {'T':>5s} {'WR':>5s} {'💰':>8s} {'سحب':>6s} {'R:R':>5s} {'TP':>5s} {'SL':>5s} {'TR':>4s} {'TM':>4s} {'سنوي':>6s} {'W':>7s} {'L':>7s}")
print("-"*95)

best = None
for r in results:
    name,t,wr,fin,dd,rr,tp_n,sl_n,tr_n,tm_n,aw,al,ann = r
    mark = "⭐" if fin > 1500 else ("🔸" if fin > 1200 else "  ")
    print(f"{mark} {name:<18s} {t:>5d} {wr:>4.0f}% ${fin:>7.0f} {dd:>+5.1f}% {rr:>4.2f} {tp_n:>5d} {sl_n:>5d} {tr_n:>4d} {tm_n:>4d} {ann:>+5.0f}% {aw:>+6.2f}% {al:>+6.2f}%")
    if best is None or fin > best[3]: best = r

if best:
    print(f"\n🏆 الأفضل: {best[0]} → ${best[3]:.0f} | WR {best[2]:.0f}% | سنوي {best[12]:+.0f}%")

print(f"\n✅ تم")
