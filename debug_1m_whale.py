#!/usr/bin/env python3
"""تصحيح: فحص قيم الحوت وRSI على 1m + اختبار بفلاتر مخففة جداً"""
import ccxt, numpy as np, pandas as pd

COINS = ['BTC','ETH','BNB','SOL','ADA','DOGE','AVAX','DOT','LINK','MATIC']
STR = 50

def compute_indicators(df):
    df = df.copy()
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
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
    df['entry'] = (df['whale'] >= 0.10) & (df['spike'] >= 1.5)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

print("🔍 فحص قيم الحوت على 1m...")
exchange = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})

for coin in COINS[:3]:  # أول 3 عملات
    candles = exchange.fetch_ohlcv(f'{coin}/USDT', '1m', limit=1000)
    df = pd.DataFrame(candles, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df_w = compute_indicators(df)
    valid = df_w.dropna(subset=['whale','rsi'])
    
    # إحصائيات الحوت
    w = valid['whale']
    e = valid['entry']
    r = valid['rsi']
    
    print(f"\n🪙 {coin} | {len(valid)} شمعة صالحة")
    print(f"   🐋 حوت: max={w.max():.4f} | mean={w.mean():.4f} | >0.10={ (w>0.10).sum()} | >0.05={ (w>0.05).sum()}")
    print(f"   📉 RSI: min={r.min():.1f} | mean={r.mean():.1f} | <35={(r<35).sum()} | <25={(r<25).sum()}")
    print(f"   🎯 entry raw (whale>=0.10 & spike>=1.5): {e.sum()}")
    
    # شموع الدخول اللي فيها حوت + سبايك + RSI<50
    relaxed = valid[(valid['whale'] >= 0.05) & (valid['spike'] >= 1.5) & (valid['rsi'] < 50)]
    print(f"   🔥 دخول مخفف جداً (whale>=0.05, spike>=1.5, RSI<50): {len(relaxed)}")
    if len(relaxed) > 0:
        for _, row in relaxed.head(5).iterrows():
            print(f"      🐋{row['whale']:.4f} spike{row['spike']:.1f} RSI{row['rsi']:.1f} close{row['close']:.4f}")
