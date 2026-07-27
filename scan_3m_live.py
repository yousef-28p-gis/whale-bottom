#!/usr/bin/env python3
"""🔍 فحص حي — كل العملات الحلال — آخر 1000 شمعة 3m"""
import ccxt, json, numpy as np, pandas as pd
from datetime import datetime, timezone

# ═══════════════ إعدادات الاستراتيجية الجديدة ═══════════════
TP = 1.5; SL = 1.0; PL = 20; TRAIL = 0.05; MAX_H = 6
STR = 50; WHALE_MIN = 0.05; RSI_MAX = 50; COMM = 0.20

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

def compute_indicators(df):
    df = df.copy()
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    df['whale'] = (df['low']-df['low_raw'])/df['low_raw'].replace(0,np.nan)*100
    df['whale'] = df['whale'].clip(lower=0)
    df['spike'] = df['volume']/df['volume'].rolling(20).mean().replace(0,np.nan)
    df['hi_raw'] = df['high'].rolling(STR).max()
    df['strength'] = (df['close']-df['low'])/(df['hi_raw']-df['low']).replace(0,np.nan)
    df['strength'] = df['strength'].clip(0,1)
    df['entry_raw'] = (df['whale']>=WHALE_MIN) & (df['spike']>=1.5)
    delta = df['close'].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain/loss.replace(0,np.nan)
    df['rsi'] = 100-(100/(1+rs))
    return df

def check_signals(df):
    """البحث عن آخر 3 شمعات فقط للإشارة الحية"""
    if df is None or len(df)<100:
        return []
    signals = []
    last_idx = len(df)-1
    for i in range(max(50, last_idx-5), last_idx+1):
        row = df.iloc[i]
        if not row['entry_raw']: continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi>=RSI_MAX: continue
        # لا إشارات متتالية
        if i-1>=0 and df.iloc[i-1]['entry_raw']: continue
        if i-2>=0 and df.iloc[i-2]['entry_raw']: continue
        signals.append({
            'idx':i, 'ts':str(row['ts']),
            'close':round(float(row['close']),6),
            'whale':round(float(row['whale']),4),
            'rsi':round(rsi,1),
            'spike':round(float(row['spike']),1),
            'strength':round(float(row['strength']),2),
        })
    return signals

# ═══════════════ MAIN ═══════════════
print("⏳ تحميل قائمة العملات الحلال...")
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
all_coins = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
print(f"📋 {len(all_coins)} عملة حلال\n")

exchange = ccxt.binance({'timeout':10000,'enableRateLimit':True})

signals_found = []
errors = []
no_data = []
scanned = 0

for i, coin in enumerate(all_coins, 1):
    try:
        candles = exchange.fetch_ohlcv(f'{coin}/USDT','3m',limit=1000)
        df = pd.DataFrame(candles,columns=['ts','open','high','low','close','volume'])
        df['ts']=pd.to_datetime(df['ts'],unit='ms')
        df_w = compute_indicators(df)
        sigs = check_signals(df_w)
        if sigs:
            for s in sigs:
                s['symbol'] = coin
                # حسب TP/SL
                price = s['close']
                s['tp_price'] = round(price*(1+TP/100),6)
                s['sl_price'] = round(price*(1-SL/100),6)
            signals_found.extend(sigs)
        scanned += 1
        if i%50==0:
            print(f"  ⏳ {i}/{len(all_coins)} ... {len(signals_found)} إشارة حتى الآن")
    except Exception as e:
        errors.append((coin, str(e)))

print(f"\n{'='*55}")
print(f"🔍 فحص حي — 3m — TP={TP}% SL={SL}%")
print(f"{'='*55}")
print(f"  ✅ تم الفحص: {scanned}")
print(f"  ❌ أخطاء: {len(errors)}")
print(f"  🚨 إشارات: {len(signals_found)}")

if signals_found:
    # ترتيب حسب قوة الحوت
    signals_found.sort(key=lambda x: x['whale'], reverse=True)
    print(f"\n{'─'*55}")
    print(f"🐋 إشارات الدخول ({len(signals_found)})")
    print(f"{'─'*55}")
    for s in signals_found:
        bars_ago = '?'
        print(f"\n  🔥 {s['symbol']}")
        print(f"     💵 سعر: {s['close']} | 🎯 هدف: {s['tp_price']} | 🛑 ستوب: {s['sl_price']}")
        print(f"     🐋 حوت: {s['whale']:.3f} | ⚡ سبايك: {s['spike']:.1f} | 📉 RSI: {s['rsi']:.1f} | 💪 قوة: {s['strength']:.2f}")
        print(f"     🕐 {s['ts']}")
else:
    print(f"\n  💤 لا توجد إشارات حالياً")

if errors:
    print(f"\n❌ أخطاء ({len(errors)}):")
    for c, e in errors[:10]:
        print(f"  {c}: {e}")
