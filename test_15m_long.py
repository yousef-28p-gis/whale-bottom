#!/usr/bin/env python3
"""🧪 باك تيست طويل — 197 عملة × 4.5 سنة 15m — استراتيجية 3m الجديدة"""
import json, os, numpy as np, pandas as pd, time
from glob import glob

COMM = 0.20; TF_MIN = 15

# ═══════════════ التجارب ═══════════════
# نأخذ أفضل 4 تجارب من اختبار 3m ونطبقها على 15m
TESTS = [
    # (name,           TP,  SL,  PL, TRAIL, MAX_H, WHALE, RSI, confirm)
    ("TP2_بتأكيد",      2.0, 1.5, 30, 0.10,  6,     0.05,  50,  True),
    ("TP2_فلاترأضيق",   2.0, 1.5, 30, 0.10,  6,     0.10,  35,  False),
    ("TP2_أساسي",       2.0, 1.5, 30, 0.10,  6,     0.05,  50,  False),
    ("TP1.5_تريل0.1",   1.5, 1.0, 20, 0.10,  6,     0.05,  50,  False),
    ("TP2.5_وسط",       2.5, 1.5, 30, 0.10,  6,     0.05,  50,  False),
]

def compute_indicators(df, whale_min, str_len=50):
    df = df.copy()
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    df['whale'] = (df['low']-df['low_raw'])/df['low_raw'].replace(0,np.nan)*100
    df['whale'] = df['whale'].clip(lower=0)
    df['spike'] = df['volume']/df['volume'].rolling(20).mean().replace(0,np.nan)
    df['hi_raw'] = df['high'].rolling(str_len).max()
    df['strength'] = (df['close']-df['low'])/(df['hi_raw']-df['low']).replace(0,np.nan)
    df['strength'] = df['strength'].clip(0,1)
    df['entry_raw'] = (df['whale']>=whale_min) & (df['spike']>=1.5)
    delta = df['close'].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain/loss.replace(0,np.nan)
    df['rsi'] = 100-(100/(1+rs))
    return df

def find_signals(df, rsi_max, confirm):
    if df is None or len(df)<100:
        return []
    signals = []
    for i in range(50, len(df)):
        row = df.iloc[i]
        if not row['entry_raw']: continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi>=rsi_max: continue
        if confirm:
            if i+1>=len(df): continue
            if df.iloc[i+1]['close']<=df.iloc[i+1]['open']: continue
        # تباعد 5 شمعات (75 دقيقة)
        skip=False
        for j in range(1,5):
            if i-j>=0 and df.iloc[i-j]['entry_raw']: skip=True; break
        if skip: continue
        signals.append({'idx':i, 'entry_price':round(float(row['close']),8)})
    return signals

def simulate(df, signals, tp, sl, pl, trail, max_h, max_pos=2):
    trades = []
    active = []
    max_bars = int(max_h*60/TF_MIN)
    
    for i in range(len(df)):
        row = df.iloc[i]
        current = float(row['close'])
        
        # دخول — حد أقصى صفقتين مفتوحتين
        for sig in signals:
            if sig['idx']==i and len(active)<max_pos:
                entry_p = sig['entry_price']
                active.append({
                    'entry':entry_p,
                    'tp':entry_p*(1+tp/100),
                    'sl':entry_p*(1-sl/100),
                    'pl_triggered':False, 'peak':entry_p,
                    'trail_price':entry_p,
                    'entry_idx':i,
                })
        
        for pos in active[:]:
            entry = pos['entry']; bars_held = i-pos['entry_idx']
            if bars_held>=max_bars:
                pnl=round((current-entry)/entry*100-COMM,4)
                pos['exit']=('TIME',pnl); active.remove(pos); trades.append(pos); continue
            if current>=pos['tp']:
                pos['exit']=('TP',round(tp-COMM,4)); active.remove(pos); trades.append(pos); continue
            if current<=pos['sl']:
                pos['exit']=('SL',round(-sl-COMM,4)); active.remove(pos); trades.append(pos); continue
            if pos.get('pl_triggered'):
                if current>pos.get('peak',entry):
                    pos['peak']=current; pos['trail_price']=current*(1-trail/100)
                if current<=pos.get('trail_price',0):
                    trail_pnl=round((pos['trail_price']-entry)/entry*100-COMM,4)
                    pos['exit']=('TRAIL',trail_pnl); active.remove(pos); trades.append(pos)
            else:
                pl_price=entry+(pos['tp']-entry)*(pl/100)
                if current>=pl_price:
                    pos['pl_triggered']=True; pos['peak']=current
                    pos['trail_price']=current*(1-trail/100)
    return trades

def calc_sharpe(returns, periods_per_year=365*24*4):  # 15m = 4 per hour = 96 per day
    if len(returns)<2: return 0
    ann_factor = np.sqrt(periods_per_year)
    return np.mean(returns)/np.std(returns)*ann_factor if np.std(returns)>0 else 0

# ═══════════════ MAIN ═══════════════
DATA_DIR = '/data/trading28/data/5year_halal'
files = sorted(glob(f'{DATA_DIR}/*_15m.json'))
print(f"📂 {len(files)} ملف بيانات 15m\n")

# تحميل كل البيانات مرة واحدة
print("⏳ تحميل البيانات التاريخية...")
all_data = {}
t0 = time.time()
for fp in files:
    coin = os.path.basename(fp).replace('_15m.json','')
    try:
        with open(fp) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['ts'] = pd.to_datetime(df['timestamp'], unit='ms')
        elif 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        # تأكيد الأعمدة
        for col in ['open','high','low','close','volume']:
            if col not in df.columns and col[0] in df.columns:
                df.rename(columns={col[0]:col}, inplace=True)
        all_data[coin] = df
    except Exception as e:
        print(f"  ❌ {coin}: {e}")

elapsed = time.time()-t0
total_candles = sum(len(df) for df in all_data.values())
print(f"  ✅ {len(all_data)} عملة | {total_candles:,} شمعة | {elapsed:.0f}s\n")

# ═══════════════ تشغيل التجارب ═══════════════
print(f"{'='*80}")
print("🧪 باك تيست طويل — 15m × 4.5 سنة")
print(f"{'='*80}")

all_results = []

for name, tp, sl, pl, trail, max_h, whale, rsi, confirm in TESTS:
    all_trades = []
    coins_with_trades = 0
    total_candles_tested = 0
    
    for coin, df in all_data.items():
        df_w = compute_indicators(df, whale)
        signals = find_signals(df_w, rsi, confirm)
        if not signals: continue
        trades = simulate(df_w, signals, tp, sl, pl, trail, max_h)
        if trades:
            coins_with_trades += 1
            all_trades.extend(trades)
            total_candles_tested += len(df)
    
    if not all_trades:
        print(f"\n📊 {name}: ❌ 0 صفقات")
        continue
    
    wins = [t for t in all_trades if t['exit'][1]>0]
    losses = [t for t in all_trades if t['exit'][1]<=0]
    wr = len(wins)/len(all_trades)*100
    net = sum(t['exit'][1] for t in all_trades)
    avg_win = np.mean([t['exit'][1] for t in wins]) if wins else 0
    avg_loss = np.mean([t['exit'][1] for t in losses]) if losses else 0
    tp_c = sum(1 for t in all_trades if t['exit'][0]=='TP')
    sl_c = sum(1 for t in all_trades if t['exit'][0]=='SL')
    tr_c = sum(1 for t in all_trades if t['exit'][0]=='TRAIL')
    tm_c = sum(1 for t in all_trades if t['exit'][0]=='TIME')
    
    returns = [t['exit'][1] for t in all_trades]
    sharpe = calc_sharpe(returns)
    
    # DD
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    dd = np.min(cumulative - peak)
    
    # محفظة
    portfolio = 1000 * np.prod([1 + r/100 for r in returns])
    
    # مدة البيانات
    first_ts = min(df['ts'].iloc[0] for coin,df in all_data.items() if coin in [t.get('symbol','') for t in all_trades[:1]] or True)
    # Use the first timestamp from the data
    first_date = all_data[list(all_data.keys())[0]]['ts'].iloc[0]
    last_date = all_data[list(all_data.keys())[0]]['ts'].iloc[-1]
    
    all_results.append({
        'name':name, 'coins':coins_with_trades, 'trades':len(all_trades),
        'wins':len(wins), 'wr':wr, 'net':net,
        'avg_win':avg_win, 'avg_loss':avg_loss,
        'tp':tp_c, 'sl':sl_c, 'trail':tr_c, 'time':tm_c,
        'sharpe':sharpe, 'dd':dd, 'portfolio':portfolio,
        'candles':total_candles_tested,
    })
    
    print(f"\n📊 {name}")
    print(f"   ⚙️ TP={tp}% SL={sl}% PL={pl}% TRAIL={trail}% MAX_H={max_h}h 🐋≥{whale} RSI<{rsi} {'✓تأكيد' if confirm else ''}")
    print(f"   🪙 {coins_with_trades} عملة | 📋 {len(all_trades)} صفقة | 🟢{len(wins)} 🔴{len(losses)}")
    print(f"   📈 WR: {wr:.1f}% | 💰 صافي: {net:+.1f}% | 📊 شارپ: {sharpe:.2f} | 📉 DD: {dd:.1f}%")
    print(f"   🟢 +{avg_win:.2f}% | 🔴 {avg_loss:.2f}%")
    print(f"   🎯TP:{tp_c} 🛑SL:{sl_c} 🐌TRAIL:{tr_c} ⏰TIME:{tm_c}")
    print(f"   🏦 محفظة: $1,000 → ${portfolio:,.0f}")

# ═══════════════ الترتيب ═══════════════
print(f"\n{'='*80}")
print("⚖️ ترتيب حسب WR — 4.5 سنة 15m")
print(f"{'='*80}")
sorted_results = sorted(all_results, key=lambda x: x['wr'], reverse=True)
print(f"  {'التجربة':<18} {'عملات':>5} {'صفقات':>6} {'WR':>7} {'صافي':>8} {'شارپ':>6} {'DD':>7} {'محفظة':>10} {'TP/SL/TR/TM'}")
print(f"  {'─'*18} {'─'*5} {'─'*6} {'─'*7} {'─'*8} {'─'*6} {'─'*7} {'─'*10} {'─'*14}")
for r in sorted_results:
    exits = f"{r['tp']}/{r['sl']}/{r['trail']}/{r['time']}"
    port = f"${r['portfolio']:,.0f}"
    print(f"  {r['name']:<18} {r['coins']:>5} {r['trades']:>6} {r['wr']:>6.1f}% {r['net']:>+7.1f}% {r['sharpe']:>6.2f} {r['dd']:>6.1f}% {port:>10} {exits}")
