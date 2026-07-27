#!/usr/bin/env python3
"""🧪 اختبار فريمات 3m و 5m — 10 عملات × 1000 شمعة"""
import ccxt, numpy as np, pandas as pd
from io import StringIO

TP = 3.5; SL = 1.5; PL = 30; TRAIL = 0.10; MAX_H = 6
STR = 50; WHALE_MIN = 0.05; RSI_MAX = 50; COMM = 0.20

COINS = ['BTC','ETH','BNB','SOL','ADA','DOGE','AVAX','DOT','LINK','MATIC']

def compute_indicators(df):
    df = df.copy()
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    df['whale'] = (df['low'] - df['low_raw']) / df['low_raw'].replace(0, np.nan) * 100
    df['whale'] = df['whale'].clip(lower=0)
    df['spike'] = df['volume'] / df['volume'].rolling(20).mean().replace(0, np.nan)
    df['hi_raw'] = df['high'].rolling(STR).max()
    df['strength'] = (df['close'] - df['low']) / (df['hi_raw'] - df['low']).replace(0, np.nan)
    df['strength'] = df['strength'].clip(0, 1)
    df['entry'] = (df['whale'] >= WHALE_MIN) & (df['spike'] >= 1.5)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def find_signals(df, symbol):
    if df is None or len(df) < 100:
        return []
    signals = []
    for i in range(50, len(df)):
        row = df.iloc[i]
        if not row['entry']:
            continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi >= RSI_MAX:
            continue
        # تباعد 3 شمعات بين الإشارات
        if i-1>=0 and df.iloc[i-1]['entry']: continue
        if i-2>=0 and df.iloc[i-2]['entry']: continue
        if i-3>=0 and df.iloc[i-3]['entry']: continue
        signals.append({
            'idx': i, 'symbol': symbol,
            'entry_price': round(float(row['close']), 8),
            'tp_price': round(float(row['close'])*(1+TP/100), 8),
            'sl_price': round(float(row['close'])*(1-SL/100), 8),
            'whale_val': round(float(row['whale']), 4),
            'rsi': round(rsi, 1),
        })
    return signals

def simulate_trades(df, signals, tf_minutes):
    """محاكاة مع خروج كل شمعة"""
    trades = []
    active = []
    max_bars = int(MAX_H * 60 / tf_minutes)
    
    for i in range(len(df)):
        row = df.iloc[i]
        current = float(row['close'])
        
        # دخول
        for sig in signals:
            if sig['idx'] == i:
                active.append({
                    'symbol': sig['symbol'],
                    'entry_price': sig['entry_price'],
                    'tp_price': sig['tp_price'],
                    'sl_price': sig['sl_price'],
                    'pl_triggered': False,
                    'peak': sig['entry_price'],
                    'trail_price': sig['entry_price'],
                    'entry_idx': i,
                    'whale_val': sig['whale_val'],
                })
        
        # خروج
        for pos in active[:]:
            entry = pos['entry_price']
            
            bars_held = i - pos['entry_idx']
            if bars_held >= max_bars:
                pnl = round((current-entry)/entry*100 - COMM, 4)
                pos['exit'] = ('TIME', pnl)
                active.remove(pos)
                trades.append(pos)
                continue
            
            if current >= pos['tp_price']:
                pos['exit'] = ('TP', round(TP-COMM, 4))
                active.remove(pos)
                trades.append(pos)
                continue
            
            if current <= pos['sl_price']:
                pos['exit'] = ('SL', round(-SL-COMM, 4))
                active.remove(pos)
                trades.append(pos)
                continue
            
            if pos.get('pl_triggered'):
                if current > pos.get('peak', entry):
                    pos['peak'] = current
                    pos['trail_price'] = current * (1 - TRAIL/100)
                if current <= pos.get('trail_price', 0):
                    trail_pnl = round((pos['trail_price']-entry)/entry*100 - COMM, 4)
                    pos['exit'] = ('TRAIL', trail_pnl)
                    active.remove(pos)
                    trades.append(pos)
            else:
                pl_price = entry + (pos['tp_price']-entry)*(PL/100)
                if current >= pl_price:
                    pos['pl_triggered'] = True
                    pos['peak'] = current
                    pos['trail_price'] = current*(1-TRAIL/100)
    
    return trades

def run_test(exchange, tf, label):
    print(f"\n{'='*60}")
    print(f"🧪 {label} (حوت≥0.05 | RSI<50 | بدون تأكيد)")
    print(f"{'='*60}")
    
    all_trades = []
    
    for coin in COINS:
        try:
            candles = exchange.fetch_ohlcv(f'{coin}/USDT', tf, limit=1000)
            df = pd.DataFrame(candles, columns=['ts','open','high','low','close','volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df_w = compute_indicators(df)
            signals = find_signals(df_w, coin)
            
            if not signals:
                print(f"  {coin}: 0 إشارات")
                continue
            
            trades = simulate_trades(df_w, signals, int(tf.replace('m','')))
            
            if trades:
                wins = [t for t in trades if t['exit'][1] > 0]
                losses = [t for t in trades if t['exit'][1] <= 0]
                wr = len(wins)/len(trades)*100
                net = sum(t['exit'][1] for t in trades)
                tp_c = sum(1 for t in trades if t['exit'][0]=='TP')
                sl_c = sum(1 for t in trades if t['exit'][0]=='SL')
                tr_c = sum(1 for t in trades if t['exit'][0]=='TRAIL')
                tm_c = sum(1 for t in trades if t['exit'][0]=='TIME')
                print(f"  🪙 {coin}: {len(trades)} صفقة | WR {wr:.0f}% | {net:+.1f}% | TP{tp_c} SL{sl_c} TR{tr_c} TM{tm_c}")
                all_trades.extend(trades)
            else:
                print(f"  {coin}: {len(signals)} إشارة → 0 صفقات")
                
        except Exception as e:
            print(f"  ❌ {coin}: {e}")
    
    return all_trades

def print_summary(trades, label):
    if not trades:
        print(f"\n📊 {label}: ❌ صفر صفقات!")
        return None
    
    wins = [t for t in trades if t['exit'][1] > 0]
    losses = [t for t in trades if t['exit'][1] <= 0]
    wr = len(wins)/len(trades)*100
    net = sum(t['exit'][1] for t in trades)
    avg_win = np.mean([t['exit'][1] for t in wins]) if wins else 0
    avg_loss = np.mean([t['exit'][1] for t in losses]) if losses else 0
    tp_c = sum(1 for t in trades if t['exit'][0]=='TP')
    sl_c = sum(1 for t in trades if t['exit'][0]=='SL')
    tr_c = sum(1 for t in trades if t['exit'][0]=='TRAIL')
    tm_c = sum(1 for t in trades if t['exit'][0]=='TIME')
    
    print(f"\n📊 {label}")
    print(f"  📋 {len(trades)} صفقة | 🟢{len(wins)} | 🔴{len(losses)}")
    print(f"  📈 WR: {wr:.1f}% | 💰 صافي: {net:+.1f}%")
    print(f"  🟢 +{avg_win:.2f}% | 🔴 {avg_loss:.2f}%")
    print(f"  🎯TP:{tp_c} 🛑SL:{sl_c} 🐌TRAIL:{tr_c} ⏰TIME:{tm_c}")
    
    return {'label':label, 'trades':len(trades), 'wins':len(wins), 'wr':wr, 'net':net,
            'avg_win':avg_win, 'avg_loss':avg_loss, 'tp':tp_c, 'sl':sl_c, 'trail':tr_c, 'time':tm_c}

# ═══════════════ MAIN ═══════════════
exchange = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})

results = []

# ت test 3m
trades_3m = run_test(exchange, '3m', 'فريم 3 دقائق')
results.append(print_summary(trades_3m, '3m'))
if trades_3m:
    print(f"\n📋 عينة 3m:")
    for t in trades_3m[:10]:
        em = '🟢' if t['exit'][1]>0 else '🔴'
        print(f"  {em} {t['symbol']} | {t['exit'][0]} | {t['exit'][1]:+.2f}% | 🐋{t['whale_val']:.3f}")

# ت test 5m
trades_5m = run_test(exchange, '5m', 'فريم 5 دقائق')
results.append(print_summary(trades_5m, '5m'))
if trades_5m:
    print(f"\n📋 عينة 5m:")
    for t in trades_5m[:10]:
        em = '🟢' if t['exit'][1]>0 else '🔴'
        print(f"  {em} {t['symbol']} | {t['exit'][0]} | {t['exit'][1]:+.2f}% | 🐋{t['whale_val']:.3f}")

# ═══════════════ مقارنة ═══════════════
print(f"\n{'='*60}")
print("⚖️ مقارنة شاملة")
print(f"{'='*60}")
print(f"  {'':>15} {'1m':>10} {'3m':>10} {'5m':>10}")
print(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*10}")

# Add 1m results from previous run
results_1m = {'label':'1m', 'trades':94, 'wins':27, 'wr':28.7, 'net':-18.99,
              'avg_win':0.88, 'avg_loss':-0.64, 'tp':0, 'sl':7, 'trail':22, 'time':65}

for r in [results_1m] + [x for x in results if x]:
    print(f"  {'صفقات':>15} {r['trades']:>10}")
for r in [results_1m] + [x for x in results if x]:
    print(f"  {'WR':>15} {r['wr']:>9.1f}%")
for r in [results_1m] + [x for x in results if x]:
    print(f"  {'صافي':>15} {r['net']:>+9.1f}%")
for r in [results_1m] + [x for x in results if x]:
    print(f"  {'TP/SL/TRAIL/TIME':>15} {r['tp']}/{r['sl']}/{r['trail']}/{r['time']}")
