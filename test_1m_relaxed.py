#!/usr/bin/env python3
"""🧪 اختبار فريم الدقيقة مع فلاتر مخففة — 10 عملات × 1000 شمعة"""
import ccxt, json, os, sys, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from io import StringIO

# ═══════════════ SETTINGS ═══════════════
TP = 3.5; SL = 1.5; PL = 30; TRAIL = 0.10; MAX_H = 6
STR = 50; WHALE_MIN = 0.30  # مخفف (كان 0.50)
RSI_MAX = 35               # مخفف (كان 25)
COMM = 0.20
MAX_POS = 2; POS_PCT = 50

# 10 عملات من القائمة الحلال
COINS = ['BTC','ETH','BNB','SOL','ADA','DOGE','AVAX','DOT','LINK','MATIC']

def compute_indicators(df):
    """نسخة طبق الأصل من whale_bottom_daemon.py"""
    df = df.copy()
    # لا نستخدم pd كمتغير loop
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
    
    # الطبقات
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    
    # حوت
    df['whale'] = (df['low'] - df['low_raw']) / df['low_raw'].replace(0, np.nan) * 100
    df['whale'] = df['whale'].clip(lower=0)
    
    # سبايك
    df['spike'] = df['volume'] / df['volume'].rolling(20).mean().replace(0, np.nan)
    
    # قوة
    df['hi_raw'] = df['high'].rolling(STR).max()
    df['strength'] = (df['close'] - df['low']) / (df['hi_raw'] - df['low']).replace(0, np.nan)
    df['strength'] = df['strength'].clip(0, 1)
    
    # إشارة الدخول
    df['entry'] = (df['whale'] >= WHALE_MIN) & (df['spike'] >= 1.5)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def check_entry_signal(df):
    """نسخة مخففة من check_entry"""
    if df is None or len(df) < 100:
        return []
    
    signals = []
    last_idx = len(df) - 1
    
    for i in range(max(50, last_idx - 10), last_idx + 1):
        row = df.iloc[i]
        if not row['entry']:
            continue
        whale_val = float(row['whale'])
        if whale_val < WHALE_MIN:
            continue
        # فلتر الحوت التالي
        if i + 1 < len(df):
            if float(df.iloc[i + 1]['whale']) >= 0.35:
                continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi >= RSI_MAX:  # مخفف
            continue
        
        # فلتر pump24
        ps = max(0, i - 96)
        pb = float(df.iloc[ps]['close'])
        ep = float(row['close'])
        pump24 = (ep - pb) / pb * 100 if pb != 0 else 0
        if pump24 >= 0:
            continue
        
        # 🔥 فلتر 3: شمعة تأكيد خضراء
        if i + 1 >= len(df):
            continue
        next_open = float(df.iloc[i + 1]['open'])
        next_close = float(df.iloc[i + 1]['close'])
        if next_close <= next_open:
            continue
        
        signals.append({
            'idx': i,
            'entry_price': round(ep, 8),
            'tp_price': round(ep * (1 + TP / 100), 8),
            'sl_price': round(ep * (1 - SL / 100), 8),
            'whale_val': round(whale_val, 4),
            'rsi': round(rsi, 1),
            'pump24': round(pump24, 2),
            'signal_ts': str(row['ts']),
        })
    
    return signals

def simulate_exits(df, signals):
    """محاكاة صفقات — خروج 1m (كل شمعة)"""
    trades = []
    active = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        current = float(row['close'])
        
        # دخول
        for sig in signals:
            if sig['idx'] == i:
                active.append({
                    'symbol': sig.get('symbol', '?'),
                    'entry_price': sig['entry_price'],
                    'tp_price': sig['tp_price'],
                    'sl_price': sig['sl_price'],
                    'pl_triggered': False,
                    'peak': sig['entry_price'],
                    'trail_price': sig['entry_price'],
                    'entry_idx': i,
                })
        
        # خروج
        for pos in active[:]:
            entry = pos['entry_price']
            tp = pos['tp_price']
            sl = pos['sl_price']
            pl_price = entry + (tp - entry) * (PL / 100)
            
            # TIME
            bars_held = i - pos['entry_idx']
            max_bars = int(MAX_H * 60)  # 6h = 360 bars @ 1m
            if bars_held >= max_bars:
                pnl = round((current - entry) / entry * 100 - COMM, 4)
                pos['exit'] = ('TIME', pnl, bars_held)
                active.remove(pos)
                trades.append(pos)
                continue
            
            # TP
            if current >= tp:
                pnl = round(TP - COMM, 4)
                pos['exit'] = ('TP', pnl, bars_held)
                active.remove(pos)
                trades.append(pos)
                continue
            
            # SL
            if current <= sl:
                pnl = round(-SL - COMM, 4)
                pos['exit'] = ('SL', pnl, bars_held)
                active.remove(pos)
                trades.append(pos)
                continue
            
            # TRAIL
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

def print_results(coin, trades):
    if not trades:
        print(f"\n  ❌ {coin}: صفر صفقات")
        return None
    
    wins = [t for t in trades if t['exit'][1] > 0]
    losses = [t for t in trades if t['exit'][1] <= 0]
    wr = len(wins) / len(trades) * 100
    
    total_ret = sum(t['exit'][1] for t in trades)
    avg_win = np.mean([t['exit'][1] for t in wins]) if wins else 0
    avg_loss = np.mean([t['exit'][1] for t in losses]) if losses else 0
    
    tp_count = sum(1 for t in trades if t['exit'][0] == 'TP')
    sl_count = sum(1 for t in trades if t['exit'][0] == 'SL')
    trail_count = sum(1 for t in trades if t['exit'][0] == 'TRAIL')
    time_count = sum(1 for t in trades if t['exit'][0] == 'TIME')
    
    print(f"\n  🪙 {coin}")
    print(f"     📋 {len(trades)} صفقة | 🟢 {len(wins)} | 🔴 {len(losses)} | WR {wr:.1f}%")
    print(f"     💰 صافي: {total_ret:+.2f}% | 🟢 +{avg_win:.2f}% | 🔴 {avg_loss:.2f}%")
    print(f"     🎯TP:{tp_count} 🛑SL:{sl_count} 🐌TRAIL:{trail_count} ⏰TIME:{time_count}")
    
    return {
        'coin': coin, 'total': len(trades), 'wins': len(wins),
        'wr': wr, 'net': total_ret,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'tp': tp_count, 'sl': sl_count, 'trail': trail_count, 'time': time_count,
    }

# ═══════════════ MAIN ═══════════════
print("⏳ جاري جلب بيانات الدقيقة...")
exchange = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})

all_results = []
total_trades = 0

for coin in COINS:
    try:
        candles = exchange.fetch_ohlcv(f'{coin}/USDT', '1m', limit=1000)
        if not candles:
            print(f"  ⚠️ {coin}: لا بيانات")
            continue
        
        df = pd.DataFrame(candles, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        
        # مؤشرات
        df_w = compute_indicators(df)
        
        # إشارات
        signals = check_entry_signal(df_w)
        for s in signals:
            s['symbol'] = coin
        
        if not signals:
            print(f"  ⚠️ {coin}: 1000 شمعة — {len(df_w.dropna(subset=['whale']))} شمعة صالحة — بلا إشارات")
            continue
        
        # محاكاة
        trades = simulate_exits(df_w, signals)
        r = print_results(coin, trades)
        if r:
            all_results.append(r)
            total_trades += len(trades)
            
    except Exception as e:
        print(f"  ❌ {coin}: خطأ — {e}")

# ═══════════════ ملخص ═══════════════
print("\n" + "=" * 50)
print("📊 ملخص الاختبار (فريم 1m — فلاتر مخففة)")
print("=" * 50)

if all_results:
    total_wins = sum(r['wins'] for r in all_results)
    wr_avg = total_wins / total_trades * 100 if total_trades > 0 else 0
    net_all = sum(r['net'] for r in all_results)
    
    print(f"  🪙 {len(all_results)} عملات | 📋 {total_trades} صفقة")
    print(f"  🟢 {total_wins} رابحة | 🔴 {total_trades - total_wins} خاسرة")
    print(f"  📈 WR: {wr_avg:.1f}% | 💰 صافي: {net_all:+.2f}%")
    print(f"  🎯 TP:{sum(r['tp'] for r in all_results)} | 🛑 SL:{sum(r['sl'] for r in all_results)} | 🐌 TRAIL:{sum(r['trail'] for r in all_results)} | ⏰ TIME:{sum(r['time'] for r in all_results)}")
else:
    print("  ❌ ولا صفقة واحدة!")
    print("  💡 السبب: 1000 شمعة (~16 ساعة) غير كافية لاستراتيجية حوت القاع")
    print("  💡 تحتاج عالأقل أسبوع (10,000 شمعة) أو تخفيف أكبر للفلاتر")
