#!/usr/bin/env python3
"""Fast whale backtest with all filters"""
import json, os, numpy as np, pandas as pd
from datetime import datetime
from collections import defaultdict

CACHE = '/data/trading28/cache/ohlcv'
SIGNALS_FILE = '/data/trading28/signals_whalesniper_all.json'
STR=50; WHALE_MIN=0.35; MIN_VOL=200000; COMM=0.20
TP=2.5; SL=2.0; PL=40; TRAIL=0.20; MH=8

STABLES={'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}
BLOCKED={'SUPER','ORCA','VANA','W','DOGS','MET','XLM','BB','COS','LUNA','S'}

def load_cached(sym,mon):
    fpath=f'{CACHE}/{sym}_{mon}.json'
    if not os.path.exists(fpath): return None
    with open(fpath) as f: data=json.load(f)
    df=pd.DataFrame(data); df['ts']=pd.to_datetime(df['ts'],unit='ms')
    return df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'}).sort_values('ts').reset_index(drop=True)

def whale_indicator(df):
    df=df.copy(); LB=30
    df['lo']=df['low'].rolling(LB).min()
    df['lc']=abs(df['low']-df['low'].shift(1))/df['low']*100
    df['sm']=df['lc'].ewm(span=3,adjust=False).mean()
    df['hi']=df['sm'].rolling(LB).max()
    df['raw']=np.where(df['low']<=df['lo'],(df['sm']+df['hi']*2)/3,0)
    df['whale']=df['raw'].ewm(span=3,adjust=False).mean().fillna(0)
    df['spike']=(df['whale']>df['whale'].shift(1))&(df['whale'].shift(1)<=0.03)
    df['wf']=df['whale'].rolling(2).mean(); df['ws']=df['whale'].rolling(5).mean()
    df['wp']=df['whale'].rolling(50).max()
    df['str']=(df['whale']/df['wp'].replace(0,np.nan)*100).fillna(0)
    df['vma']=df['volume'].rolling(20).mean()
    df['entry']=(df['spike']&(df['wf']>df['ws'])&(df['str']>STR)&(df['volume']>df['vma']*1.0))
    return df

def sim(df,ei):
    ep=df.iloc[ei]['close']; tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
    pl_p=ep+(tp_p-ep)*(PL/100)
    pl_trig=False; peak=ep; trail_p=0
    for j in range(ei+1,len(df)):
        cur=df.iloc[j]['close']; h=(j-ei)*0.25
        if h>MH: return round((cur-ep)/ep*100,4)
        if cur>=tp_p: return round(TP,4)
        if cur<=sl_p: return round(-SL,4)
        if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
        if pl_trig:
            if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
            if cur<=trail_p: return round((trail_p-ep)/ep*100,4)
    return round((df.iloc[-1]['close']-ep)/ep*100,4)

def fast_atr_pct(df):
    """Vectorized ATR as % of close"""
    tr=pd.concat([
        df['high']-df['low'],
        abs(df['high']-df['close'].shift(1)),
        abs(df['low']-df['close'].shift(1))
    ],axis=1).max(axis=1)
    atr=tr.rolling(14).mean()
    return (atr/df['close']*100).values  # numpy array

print("Loading signals...")
with open(SIGNALS_FILE) as f: raw=json.load(f)
signals=[]
for s in raw:
    if s['symbol'] in STABLES or s['symbol'] in BLOCKED: continue
    if s.get('direction','LONG')!='LONG': continue
    if s.get('volume_usdt',0)<MIN_VOL: continue
    dt=datetime.fromisoformat(s['dt'])
    if dt.month not in (4,5,6) or dt.year!=2026: continue
    signals.append({'symbol':s['symbol'],'dt':dt,'month':dt.strftime('%Y-%m')})

by_pair=defaultdict(list)
for sig in signals: by_pair[(sig['symbol'],sig['month'])].append(sig)
print(f"Signal groups: {len(by_pair)}")

entries=[]
processed=0
for (sym,mon),sigs in by_pair.items():
    processed+=1
    if processed%50==0: print(f"  Processing {processed}/{len(by_pair)}... {len(entries)} entries so far")
    
    df=load_cached(sym,mon)
    if df is None: continue
    df_w=whale_indicator(df)
    
    # Fast vectorized ATR
    atr_arr=fast_atr_pct(df_w)
    
    for sig in sigs:
        df_w['td']=abs((df_w['ts']-sig['dt']).dt.total_seconds())
        n=df_w['td'].idxmin(); fwd=df_w.iloc[n:].reset_index(drop=True)
        for j,row in fwd.iterrows():
            if j*0.25>24: break
            if row['entry'] and float(row['whale'])>=WHALE_MIN:
                global_idx=n+j
                ep=float(row['close'])
                candle_pct=(float(row['close'])-float(row['open']))/float(row['open'])*100 if row['open']!=0 else 0
                atr_pct=float(atr_arr[global_idx]) if global_idx<len(atr_arr) and not pd.isna(atr_arr[global_idx]) else 0
                whale_now=float(fwd.iloc[j]['whale'])
                whale_prev2=float(fwd.iloc[j-2]['whale']) if j>=2 else whale_now
                whale_accel=whale_now-whale_prev2
                whale_next=float(fwd.iloc[j+1]['whale']) if j+1<len(fwd) else 0
                ps=max(0,global_idx-96)
                pb=df_w['close'].iloc[ps]
                pump24=(ep-pb)/pb*100
                entries.append({
                    'fwd':fwd,'ei':j,
                    'wait_h':round(j*0.25,1),
                    'candle_pct':candle_pct,'atr_pct':atr_pct,
                    'whale_accel':whale_accel,
                    'whale_next':whale_next,
                    'pump24':pump24,
                })
                break

print(f'\nTotal entries: {len(entries)}')

# Now run all tests
all_results=[]
def test(label, ee):
    if not ee: return
    tr=[sim(e['fwd'],e['ei']) for e in ee]
    nets=[round(p-COMM,4) for p in tr]
    net=sum(nets); wr=sum(1 for n in nets if n>0)/len(nets)*100
    avg_w=np.mean([n for n in nets if n>0]) if any(n>0 for n in nets) else 0
    avg_l=np.mean([n for n in nets if n<=0]) if any(n<=0 for n in nets) else 0
    line=f'{label:<42} {len(ee):>5} | WR {wr:>5.1f}% | Net {net:>+7.1f}% | AvgW {avg_w:>+6.2f} | AvgL {avg_l:>+6.2f}'
    all_results.append(line)
    print(line)

print()
print(f'{"Filter":<42} {"#":>5} | {"WR":>6} | {"Net":>8} | {"AvgW":>7} | {"AvgL":>7}')
print('-'*90)

test('BASELINE (كل الصفقات)', entries)

print('\n=== 1️⃣ عمر الإشارة (متى تأكد الحوت) ===')
for lo,hi in [(0,1),(1,4),(4,8),(8,12),(12,24)]:
    ee=[e for e in entries if lo<=e['wait_h']<hi]
    test(f'  تأكيد {lo}-{hi:>2}h', ee)

print('\n=== 2️⃣ سرعة ارتفاع الحوت (whale accel) ===')
for lo,hi in [(0,0.05),(0.05,0.15),(0.15,0.50),(0.50,99)]:
    ee=[e for e in entries if lo<=e['whale_accel']<hi]
    test(f'  تسارع {lo:.2f}-{hi:.2f}', ee)

print('\n=== 3️⃣ فلتر الشموع الشاذة ===')
norm=[e for e in entries if not (e['candle_pct']>6 or (e['atr_pct']>0 and e['candle_pct']>3*e['atr_pct']))]
outl=[e for e in entries if (e['candle_pct']>6 or (e['atr_pct']>0 and e['candle_pct']>3*e['atr_pct']))]
test('  شمعة طبيعية (بدون شاذ)', norm)
test('  الشاذة (مستبعدة)', outl)

print('\n=== 4️⃣ فلتر Pump 24h (طلوع السعر قبل الدخول) ===')
for lim in [20,15,10]:
    ee=[e for e in entries if e['pump24']<lim]
    test(f'  Pump24h < {lim}%', ee)

print('\n=== 5️⃣ تأكيد الحوت (شمعتين متتاليتين ≥0.35) ===')
ee2=[e for e in entries if e['whale_next']>=0.35]
ee1=[e for e in entries if e['whale_next']<0.35]
test('  شمعتين ≥0.35', ee2)
test('  شمعة وحدة فقط', ee1)

print('\n' + '='*90)
print('🔶 المجموعات المركّبة (أفضل فلترين: 4-8h + تسارع 0.15-0.50)')
print('='*90)

combo=[e for e in entries if 4<=e['wait_h']<8 and 0.15<=e['whale_accel']<0.50]
test('🔶 4-8h + تسارع 0.15-0.50', combo)

combo2=[e for e in combo if e['pump24']<15]
test('🔶 + Pump24h < 15%', combo2)

combo3=[e for e in combo2 if not (e['candle_pct']>6 or (e['atr_pct']>0 and e['candle_pct']>3*e['atr_pct']))]
test('🔶 + شمعة طبيعية', combo3)

combo4=[e for e in combo if e['whale_next']>=0.35]
test('🔶 + حوت شمعتين ≥0.35', combo4)

# Save results
with open('/data/trading28/whale_filter_results.txt','w') as f:
    f.write('\n'.join(all_results))
print('\n✅ Saved to whale_filter_results.txt')
