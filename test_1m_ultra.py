#!/usr/bin/env python3
"""🧪 اختبار 1m بفلاتر مخففة جداً — بدون تأكيد، بدون pump24"""
import ccxt, json, os, numpy as np, pandas as pd

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

def check_entry_signal(df, symbol):
    """مخفف جداً — فقط حوت + سبايك + RSI"""
    if df is None or len(df) < 100:
        return []
    
    signals = []
    last_idx = len(df) - 1
    
    for i in range(50, last_idx + 1):
        row = df.iloc[i]
        if not row['entry']:
            continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi >= RSI_MAX:
            continue
        
        # ⚠️ ممنوع إشارات متتالية — آخر 3 شمعات
        if i - 1 >= 0 and df.iloc[i - 1]['entry']:
            continue
        if i - 2 >= 0 and df.iloc[i - 2]['entry']:
            continue
        
        signals.append({
            'idx': i,
            'entry_price': round(float(row['close']), 8),
            'tp_price': round(float(row['close']) * (1 + TP / 100), 8),
            'sl_price': round(float(row['close']) * (1 - SL / 100), 8),
            'whale_val': round(float(row['whale']), 4),
            'rsi': round(rsi, 1),
            'symbol': symbol,
            'ts': str(row['ts']),
        })
    
    return signals

def simulate_exits(df, signals, exit_every_bars=1):
    trades = []
    active = []
    
    for i in range(len(df)):
        if i % exit_every_bars != 0:
            continue
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
            tp = pos['tp_price']
            sl = pos['sl_price']
            pl_price = entry + (tp - entry) * (PL / 100)
            
            bars_held = (i - pos['entry_idx']) // exit_every_bars
            max_exit_checks = int(MAX_H * 60 / exit_every_bars)
            if bars_held >= max_exit_checks:
                pnl = round((current - entry) / entry * 100 - COMM, 4)
                pos['exit'] = ('TIME', pnl, bars_held)
                active.remove(pos)
                trades.append(pos)
                continue
            
            if current >= tp:
                pnl = round(TP - COMM, 4)
                pos['exit'] = ('TP', pnl, bars_held)
                active.remove(pos)
                trades.append(pos)
                continue
            
            if current <= sl:
                pnl = round(-SL - COMM, 4)
                pos['exit'] = ('SL', pnl, bars_held)
                active.remove(pos)
                trades.append(pos)
                continue
            
            if pos.get('pl_triggered'):
                if current > pos.get('peak', entry):
                    pos['peak'] = current
                    pos['trail_price'] = current * (1 - TRAIL / 100)
                if current <= pos.get('trail_price', 0):
                    trail_pnl = round((pos['trail_price'] - entry) / entry * 100 - COMM, 4)
                    pos['exit'] = ('TRAIL', trail_pnl, bars_held)
                    active.remove(pos)
                    trades.append(pos)
            else:
                if current >= pl_price:
                    pos['pl_triggered'] = True
                    pos['peak'] = current
                    pos['trail_price'] = current * (1 - TRAIL / 100)
    
    return trades

# ═══════════════ MAIN ═══════════════
print("=" * 55)
print("🧪 اختبار 1m بفلاتر مخففة: حوت≥0.05 | RSI<50 | بدون تأكيد")
print("=" * 55)

exchange = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})

all_results = []
all_trades = []

for coin in COINS:
    try:
        candles = exchange.fetch_ohlcv(f'{coin}/USDT', '1m', limit=1000)
        if not candles:
            continue
        
        df = pd.DataFrame(candles, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df_w = compute_indicators(df)
        signals = check_entry_signal(df_w, coin)
        
        # دخول بحد أقصى 2 صفقة مفتوحة
        trades_coin = []
        for sig in signals:
            trades = simulate_exits(df_w, [sig])
            trades_coin.extend(trades)
        
        if trades_coin:
            wins = [t for t in trades_coin if t['exit'][1] > 0]
            losses = [t for t in trades_coin if t['exit'][1] <= 0]
            wr = len(wins) / len(trades_coin) * 100
            net = sum(t['exit'][1] for t in trades_coin)
            tp_c = sum(1 for t in trades_coin if t['exit'][0]=='TP')
            sl_c = sum(1 for t in trades_coin if t['exit'][0]=='SL')
            tr_c = sum(1 for t in trades_coin if t['exit'][0]=='TRAIL')
            tm_c = sum(1 for t in trades_coin if t['exit'][0]=='TIME')
            
            print(f"\n🪙 {coin}: {len(signals)} إشارة → {len(trades_coin)} صفقة")
            print(f"   🟢{len(wins)} | 🔴{len(losses)} | WR {wr:.1f}% | صافي {net:+.2f}%")
            print(f"   🎯{tp_c} 🛑{sl_c} 🐌{tr_c} ⏰{tm_c}")
            
            all_results.append({'coin':coin, 'trades':len(trades_coin), 'wins':len(wins), 'wr':wr, 'net':net})
            all_trades.extend(trades_coin)
        else:
            print(f"\n🪙 {coin}: {len(signals)} إشارة → 0 صفقات منفذة")
            
    except Exception as e:
        print(f"\n❌ {coin}: {e}")

# ═══════════════ ملخص ═══════════════
print("\n" + "=" * 55)
print("📊 ملخص نهائي — 1m فلاتر مخففة")
print("=" * 55)

if all_trades:
    wins = [t for t in all_trades if t['exit'][1] > 0]
    losses = [t for t in all_trades if t['exit'][1] <= 0]
    wr = len(wins) / len(all_trades) * 100
    net = sum(t['exit'][1] for t in all_trades)
    avg_win = np.mean([t['exit'][1] for t in wins]) if wins else 0
    avg_loss = np.mean([t['exit'][1] for t in losses]) if losses else 0
    
    tp_c = sum(1 for t in all_trades if t['exit'][0]=='TP')
    sl_c = sum(1 for t in all_trades if t['exit'][0]=='SL')
    tr_c = sum(1 for t in all_trades if t['exit'][0]=='TRAIL')
    tm_c = sum(1 for t in all_trades if t['exit'][0]=='TIME')
    
    print(f"  🪙 {len(all_results)} عملات | 📋 {len(all_trades)} صفقة")
    print(f"  🟢 {len(wins)} رابحة | 🔴 {len(losses)} خاسرة")
    print(f"  📈 WR: {wr:.1f}% | 💰 صافي: {net:+.2f}%")
    print(f"  🟢 متوسط ربح: +{avg_win:.2f}% | 🔴 متوسط خسارة: {avg_loss:.2f}%")
    print(f"  🎯TP:{tp_c} 🛑SL:{sl_c} 🐌TRAIL:{tr_c} ⏰TIME:{tm_c}")
    
    # تفاصيل الصفقات
    print(f"\n📋 تفاصيل الصفقات:")
    for t in all_trades[:20]:
        em = '🟢' if t['exit'][1] > 0 else '🔴'
        print(f"  {em} {t['symbol']} | {t['exit'][0]} | {t['exit'][1]:+.2f}% | 🐋{t['whale_val']:.3f}")
else:
    print("  ❌ صفر صفقات!")
