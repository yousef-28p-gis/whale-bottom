#!/usr/bin/env python3 -u
"""اختبار فلاتر التفريق بين حوت الشراء وحوت البيع — 5 سنوات — حلال"""
import json, os, sys, numpy as np, pandas as pd
from collections import defaultdict
from datetime import timedelta

# فرض unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

CACHE_DIR = '/data/trading28/data/5year_halal'
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
BLOCK_HOURS = {1,3,6,12,0,4}

# Filter params
PRE_DROP_PCT = -2.0   # انخفاض % في آخر شموع
PRE_DROP_CANDLES = 4  # عدد الشموع

print('🐋 اختبار فلاتر حوت الشراء vs حوت البيع', flush=True)
print(f'⚙️ TP={TP} SL={SL} PL={PL} TR={TRAIL} MH={MH}h WHALE≥{WHALE_MIN} RSI<25', flush=True)
print(f'🔍 فلاتر إضافية:', flush=True)
print(f'  فلتر 2 (سياق النزول): انخفاض ≥ {abs(PRE_DROP_PCT)}% في {PRE_DROP_CANDLES} شمعات', flush=True)
print(f'  فلتر 3 (تأكيد الشمعة التالية): شمعة التأكيد خضراء (close > open)', flush=True)
print()

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
    delta = df['close'].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100/(1+rs))
    return df

def simulate_exit(df, i, ep):
    """Simulate trade exit. Returns (pnl, exit_type)"""
    tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
    pl_p=ep+(tp_p-ep)*(PL/100)
    pl_trig=False; peak=ep; trail_p=0
    for k in range(i+1, len(df)):
        cur=float(df.iloc[k]['close']); h=(k-i)*0.25
        if h>MH: return round((cur-ep)/ep*100-COMM,4), 'TIME'
        if cur>=tp_p: return round(TP-COMM,4), 'TP'
        if cur<=sl_p: return round(-SL-COMM,4), 'SL'
        if not pl_trig and cur>=pl_p: pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
        if pl_trig:
            if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
            if cur<=trail_p: return round((trail_p-ep)/ep*100-COMM,4), 'TRAIL'
    return round((float(df.iloc[-1]['close'])-ep)/ep*100-COMM,4), 'EOD'

def run_backtest(filter_mode):
    """filter_mode: 'baseline', 'f2_pre_trend', 'f3_confirm', 'combo'"""
    all_trades = []
    done = 0
    
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith('.json'): continue
        fpath = f'{CACHE_DIR}/{fname}'
        if not os.path.exists(fpath): continue  # skip broken symlinks
        sym = fname.replace('_15m.json','')
        
        with open(fpath) as f:
            data = json.load(f)
        df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.sort_values('ts').reset_index(drop=True)
        if len(df) < 500: continue
        
        df = compute_indicators(df)
        
        for i in range(50, len(df)-10):
            row = df.iloc[i]
            if not row['entry']: continue
            if float(row['whale']) < WHALE_MIN: continue
            if i+1 < len(df) and float(df.iloc[i+1]['whale']) >= 0.35: continue
            rsi = float(row['rsi'])
            if np.isnan(rsi) or rsi >= 25: continue
            if row['ts'].weekday() == 3: continue
            if row['ts'].hour in BLOCK_HOURS: continue
            ps=max(0,i-96); pb=float(df.iloc[ps]['close']); ep=float(row['close'])
            if (ep-pb)/pb*100 >= 0: continue
            
            # === FILTER 2: Pre-entry trend ===
            if filter_mode in ('f2_pre_trend', 'combo'):
                # Check if price dropped sharply before entry
                pre_start = max(0, i - PRE_DROP_CANDLES)
                pre_price = float(df.iloc[pre_start]['close'])
                pre_change = (ep - pre_price) / pre_price * 100
                if pre_change > PRE_DROP_PCT:  # Price didn't drop enough
                    continue
            
            # === FILTER 3: Next candle confirmation ===
            if filter_mode in ('f3_confirm', 'combo'):
                if i+1 >= len(df): continue
                next_candle = df.iloc[i+1]
                # Next candle must be green (close > open)
                if float(next_candle['close']) <= float(next_candle['open']):
                    continue
            
            pnl, exit_ = simulate_exit(df, i, ep)
            all_trades.append({'sym':sym, 'dt':row['ts'], 'pnl':pnl, 'exit':exit_})
        
        done += 1
        if done % 20 == 0:
            print(f'  [{filter_mode}] {done} عملة | {len(all_trades)} صفقة', flush=True)
    
    return all_trades

def compute_stats(trades, label):
    if not trades: return {'label':label, 'signals':0, 'wins':0, 'losses':0, 'wr':0, 'net':0}
    nets=[t['pnl'] for t in trades]
    wins=sum(1 for n in nets if n>0)
    exits=defaultdict(int)
    for t in trades: exits[t['exit']] += 1
    
    # Portfolio: 2×50%
    trades_sorted = sorted(trades, key=lambda x: x['dt'])
    capital = 1000.0; peak = 1000.0; max_dd = 0.0
    active = []; skipped = 0; taken = 0; exec_trades = []
    
    for t in trades_sorted:
        dt = t['dt']
        still_active = []
        for exit_dt, cost, pnl_amt in active:
            if dt >= exit_dt:
                capital += cost + pnl_amt
            else:
                still_active.append((exit_dt, cost, pnl_amt))
        active = still_active
        
        if len(active) >= 2:
            skipped += 1; continue
        pos_size = capital * 0.50
        if capital < pos_size:
            skipped += 1; continue
        pnl_amt = pos_size * t['pnl'] / 100
        capital -= pos_size
        active.append((dt + timedelta(hours=MH), pos_size, pnl_amt))
        taken += 1
        exec_trades.append(t)
        
        equity = capital + sum(pc + pd for _, pc, pd in active)
        if equity > peak: peak = equity
        dd = (equity - peak) / peak * 100
        if dd < max_dd: max_dd = dd
    
    for _, cost, pnl_amt in active:
        capital += cost + pnl_amt
    
    exec_nets = [t['pnl'] for t in exec_trades]
    exec_wins = sum(1 for n in exec_nets if n > 0)
    
    ev = sum(exec_nets)/len(exec_nets) if exec_nets else 0
    avg_win = sum(n for n in exec_nets if n>0)/max(1,sum(1 for n in exec_nets if n>0))
    avg_loss = sum(n for n in exec_nets if n<0)/max(1,sum(1 for n in exec_nets if n<0))
    
    return {
        'label': label,
        'signals': len(trades),
        'executed': taken,
        'skipped_portfolio': skipped,
        'wins': exec_wins,
        'losses': taken - exec_wins,
        'wr': exec_wins/taken*100 if taken>0 else 0,
        'net': sum(exec_nets),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'ev': ev,
        'portfolio': capital,
        'portfolio_return': (capital/1000-1)*100,
        'max_dd': max_dd,
        'exits': dict(exits),
        'annual_return': ((capital/1000)**(1/5)-1)*100,
    }

# Run all 4 variants
print('='*60)
print('🔄 بدء الاختبارات...', flush=True)
print('='*60)

variants = [
    ('baseline', 'بدون فلاتر إضافية'),
    ('f2_pre_trend', 'فلتر 2: سياق النزول'),
    ('f3_confirm', 'فلتر 3: تأكيد الشمعة'),
    ('combo', 'فلتر 2+3 معاً'),
]

results = {}
for mode, desc in variants:
    print(f'\n🐋 {desc}...', flush=True)
    trades = run_backtest(mode)
    stats = compute_stats(trades, desc)
    results[mode] = stats
    print(f'  ✅ انتهى: {stats["executed"]} صفقة منفذة', flush=True)

# Print comparison
print('\n' + '='*80)
print('📊 مقارنة الفلاتر — 5 سنوات — 212 عملة حلال')
print('='*80)
print(f'{"":<20} {"بدون فلاتر":>12} {"فلتر 2":>12} {"فلتر 3":>12} {"فلتر 2+3":>12}')
print('-'*70)

metrics = [
    ('signals', '📋 إشارات', 'd'),
    ('executed', '✅ منفذة', 'd'),
    ('wr', '📈 Win Rate', '.1f%%'),
    ('avg_win', '🟢 متوسط الربح', '.2f%%'),
    ('avg_loss', '🔴 متوسط الخسارة', '.2f%%'),
    ('ev', '📊 القيمة المتوقعة', '.2f%%'),
    ('portfolio_return', '💼 عائد المحفظة', '.1f%%'),
    ('max_dd', '📉 أقصى سحب', '.2f%%'),
    ('annual_return', '📈 عائد سنوي', '.1f%%'),
]

for key, desc, fmt in metrics:
    vals = []
    for mode, _ in variants:
        v = results[mode][key]
        if '%%' in fmt:
            fmt_str = f'{v:{fmt.replace("%%","")}%}'
        else:
            fmt_str = f'{v:{fmt}}'
        vals.append(fmt_str)
    print(f'{desc:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}')

# Exit distribution
print(f'\n📤 توزيع المخارج:')
for mode, desc in variants:
    exits = results[mode]['exits']
    total = sum(exits.values()) or 1
    tp = exits.get('TP',0); sl = exits.get('SL',0)
    trail = exits.get('TRAIL',0); time_ = exits.get('TIME',0)+exits.get('EOD',0)
    print(f'  {desc}: TP={tp}({tp/total*100:.0f}%) SL={sl}({sl/total*100:.0f}%) TRAIL={trail}({trail/total*100:.0f}%) TIME={time_}({time_/total*100:.0f}%)')

print(f'\n🏆 الفلتر الأفضل: ', end='')
best = max(results.items(), key=lambda x: x[1]['wr'])
print(f'{best[1]["label"]} — WR {best[1]["wr"]:.1f}% | عائد {best[1]["portfolio_return"]:.1f}% | DD {best[1]["max_dd"]:.2f}%')
print('='*80, flush=True)
