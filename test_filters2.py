import json, os, sys, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}
PRE_DROP_PCT = -2.0
PRE_DROP_CANDLES = 4

print('🐋 اختبار فلاتر حوت الشراء vs حوت البيع', flush=True)
print(f'فلاتر: فلتر2(نزول>{abs(PRE_DROP_PCT)}%/4شمعات) فلتر3(تأكيد خضراء)', flush=True)

def compute_indicators(df):
    LB=30
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
    delta=df['close'].diff(); gain=delta.where(delta>0,0).rolling(14).mean()
    loss=(-delta.where(delta<0,0)).rolling(14).mean()
    df['rsi']=100-(100/(1+gain/loss.replace(0,np.nan)))
    return df

def one_trade(df, i, ep):
    tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
    pl_p=ep+(tp_p-ep)*(PL/100); pl_trig=False; peak=ep; trail_p=0
    for k in range(i+1,len(df)):
        cur=float(df.iloc[k]['close']); h=(k-i)*0.25
        if h>MH: return round((cur-ep)/ep*100-COMM,4),'TIME'
        if cur>=tp_p: return round(TP-COMM,4),'TP'
        if cur<=sl_p: return round(-SL-COMM,4),'SL'
        if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
        if pl_trig:
            if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
            if cur<=trail_p: return round((trail_p-ep)/ep*100-COMM,4),'TRAIL'
    return round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4),'EOD'

def run_mode(mode):
    """mode: 'base','f2','f3','combo'"""
    trades=[]; done=0; total=0
    files=[f for f in sorted(os.listdir(CACHE_DIR)) if f.endswith('.json')]
    for fname in files:
        fpath=f'{CACHE_DIR}/{fname}'
        if not os.path.exists(fpath): continue
        total+=1; sym=fname.replace('_15m.json','')
        with open(fpath) as f: data=json.load(f)
        df=pd.DataFrame(data,columns=['ts','open','high','low','close','volume'])
        df['ts']=pd.to_datetime(df['ts'],unit='ms'); df=df.sort_values('ts').reset_index(drop=True)
        if len(df)<500: continue
        df=compute_indicators(df)
        for i in range(50,len(df)-10):
            row=df.iloc[i]
            if not row['entry']: continue
            if float(row['whale'])<WHALE_MIN: continue
            if i+1<len(df) and float(df.iloc[i+1]['whale'])>=0.35: continue
            rsi=float(row['rsi'])
            if np.isnan(rsi) or rsi>=25: continue
            if row['ts'].weekday()==3: continue
            if row['ts'].hour in BLOCK_HOURS: continue
            ps=max(0,i-96); pb=float(df.iloc[ps]['close']); ep=float(row['close'])
            if (ep-pb)/pb*100>=0: continue
            # Filter 2: pre-trend
            if mode in ('f2','combo'):
                pre_start=max(0,i-PRE_DROP_CANDLES)
                pre_price=float(df.iloc[pre_start]['close'])
                if (ep-pre_price)/pre_price*100>PRE_DROP_PCT: continue
            # Filter 3: next candle green
            if mode in ('f3','combo'):
                if i+1>=len(df): continue
                nc=df.iloc[i+1]
                if float(nc['close'])<=float(nc['open']): continue
            pnl,ex=one_trade(df,i,ep)
            trades.append({'sym':sym,'dt':str(row['ts']),'pnl':pnl,'exit':ex})
        done+=1
        if done%30==0: print(f'  [{mode}] {done}/{total} عملات | {len(trades)} صفقة', flush=True)
    return trades

def portfolio_stats(trades):
    if not trades: return None
    nets=[t['pnl'] for t in trades]
    wins=sum(1 for n in nets if n>0); exits=defaultdict(int)
    for t in trades: exits[t['exit']]+=1
    sorted_t=sorted(trades,key=lambda x:x['dt'])
    capital=1000.0; peak=1000.0; max_dd=0.0; active=[]; skipped=0; taken=0; exec_t=[]
    for t in sorted_t:
        dt=pd.to_datetime(t['dt'])
        still_active=[]
        for edt,cost,pnl_amt in active:
            if dt>=edt: capital+=cost+pnl_amt
            else: still_active.append((edt,cost,pnl_amt))
        active=still_active
        if len(active)>=2: skipped+=1; continue
        pos_size=capital*0.50
        if capital<pos_size: skipped+=1; continue
        pnl_amt=pos_size*t['pnl']/100; capital-=pos_size
        active.append((dt+timedelta(hours=MH),pos_size,pnl_amt)); taken+=1; exec_t.append(t)
        equity=capital+sum(pc+pd for _,pc,pd in active)
        if equity>peak: peak=equity
        dd=(equity-peak)/peak*100
        if dd<max_dd: max_dd=dd
    for _,cost,pnl_amt in active: capital+=cost+pnl_amt
    exec_nets=[t['pnl'] for t in exec_t]; exec_wins=sum(1 for n in exec_nets if n>0)
    avg_w=sum(n for n in exec_nets if n>0)/max(1,sum(1 for n in exec_nets if n>0))
    avg_l=sum(n for n in exec_nets if n<0)/max(1,sum(1 for n in exec_nets if n<0))
    return {'signals':len(trades),'executed':taken,'skipped':skipped,'wr':exec_wins/taken*100 if taken else 0,
            'net':sum(exec_nets),'avg_win':avg_w,'avg_loss':avg_l,'ev':sum(exec_nets)/taken if taken else 0,
            'pf':capital,'pf_ret':(capital/1000-1)*100,'max_dd':max_dd,'exits':dict(exits),
            'annual':((capital/1000)**(1/5)-1)*100}

print('='*60, flush=True)
results={}
for mode,label in [('base','بدون فلاتر'),('f2','فلتر 2: نزول'),('f3','فلتر 3: تأكيد'),('combo','فلتر 2+3')]:
    print(f'\n🐋 {label}...', flush=True)
    trades=run_mode(mode)
    stats=portfolio_stats(trades)
    results[mode]=stats
    print(f'  ✅ {stats["executed"]} صفقة | WR {stats["wr"]:.1f}% | عائد {stats["pf_ret"]:.1f}% | DD {stats["max_dd"]:.2f}%', flush=True)

print('\n'+'='*80)
print('📊 مقارنة الفلاتر — 179 عملة — 5 سنوات')
print('='*80)
print(f'{"":<22} {"بدون":>10} {"فلتر2":>10} {"فلتر3":>10} {"2+3":>10}')
print('-'*60)
for key,desc,fmt in [('signals','📋 إشارات','d'),('executed','✅ منفذة','d'),('wr','📈 Win Rate','.1f'),('avg_win','🟢 متوسط ربح','.2f'),('avg_loss','🔴 متوسط خسارة','.2f'),('pf_ret','💼 عائد المحفظة','.1f'),('max_dd','📉 أقصى سحب','.2f'),('annual','📈 عائد سنوي','.1f')]:
    vals=[]
    for m in ['base','f2','f3','combo']:
        v=results[m][key]
        if '%%' in fmt: vals.append(f'{v:{fmt.replace("%","")}}%')
        else: vals.append(f'{v:{fmt}}')
    print(f'{desc:<22} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}')

# Best
best=max(results.items(),key=lambda x:x[1]['wr'])
print(f'\n🏆 الأفضل WR: {best[1]["wr"]:.1f}% — ({best[0]})'
      f' | عائد {best[1]["pf_ret"]:.1f}% | DD {best[1]["max_dd"]:.2f}%')
print('='*80, flush=True)
