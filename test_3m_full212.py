#!/usr/bin/env python3
"""🧪 باك تيست كامل — 212 عملة × 3m — تجارب متعددة"""
import ccxt, json, numpy as np, pandas as pd, time
from datetime import datetime, timezone
from io import StringIO

COMM = 0.20; TF_MIN = 3

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# ═══════════════ التجارب ═══════════════
# (name, TP, SL, PL, TRAIL, MAX_H, WHALE_MIN, RSI_MAX, confirm)
TESTS = [
    ("TP1.5_تريل0.1",    1.5, 1.0, 20, 0.10, 6, 0.05, 50, False),
    ("TP2.0_الأساسي",    2.0, 1.5, 30, 0.10, 6, 0.05, 50, False),
    ("TP2.5_وسط",        2.5, 1.5, 30, 0.10, 6, 0.05, 50, False),
    ("TP1.5_PL15_TR10",  1.5, 1.0, 15, 0.10, 6, 0.05, 50, False),
    ("TP2_SL1_PL25",     2.0, 1.0, 25, 0.10, 6, 0.05, 50, False),
    ("TP2_فلاترأضيق",    2.0, 1.5, 30, 0.10, 6, 0.10, 35, False),
    ("TP2_بتأكيد",       2.0, 1.5, 30, 0.10, 6, 0.05, 50, True),
    ("TP2_MAXH4",        2.0, 1.5, 30, 0.10, 4, 0.05, 50, False),
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

def find_signals(df, symbol, whale_min, rsi_max, confirm):
    """كل الإشارات عبر كامل البيانات"""
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
        # تباعد 3 شمعات
        if i-1>=0 and df.iloc[i-1]['entry_raw']: continue
        if i-2>=0 and df.iloc[i-2]['entry_raw']: continue
        if i-3>=0 and df.iloc[i-3]['entry_raw']: continue
        signals.append({
            'idx':i, 'symbol':symbol,
            'entry_price':round(float(row['close']),8),
        })
    return signals

def simulate(df, signals, tp, sl, pl, trail, max_h):
    trades = []
    active = []
    max_bars = int(max_h*60/TF_MIN)
    tp_pct = tp; sl_pct = sl
    
    for i in range(len(df)):
        row = df.iloc[i]
        current = float(row['close'])
        
        for sig in signals:
            if sig['idx']==i:
                active.append({
                    'entry':sig['entry_price'],
                    'tp':sig['entry_price']*(1+tp_pct/100),
                    'sl':sig['entry_price']*(1-sl_pct/100),
                    'pl_triggered':False, 'peak':sig['entry_price'],
                    'trail_price':sig['entry_price'],
                    'entry_idx':i, 'symbol':sig['symbol'],
                })
        
        for pos in active[:]:
            entry = pos['entry']; bars_held = i-pos['entry_idx']
            if bars_held>=max_bars:
                pnl=round((current-entry)/entry*100-COMM,4)
                pos['exit']=('TIME',pnl); active.remove(pos); trades.append(pos); continue
            if current>=pos['tp']:
                pos['exit']=('TP',round(tp_pct-COMM,4)); active.remove(pos); trades.append(pos); continue
            if current<=pos['sl']:
                pos['exit']=('SL',round(-sl_pct-COMM,4)); active.remove(pos); trades.append(pos); continue
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

# ═══════════════ MAIN ═══════════════
print("⏳ تحميل العملات...")
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
print(f"📋 {len(COINS)} عملة")

exchange = ccxt.binance({'timeout':10000,'enableRateLimit':True})

# ═══════════════ جلب البيانات ═══════════════
print("⏳ جلب بيانات 3m...")
all_data = {}
errors = 0
t0 = time.time()
for i, coin in enumerate(COINS):
    try:
        candles = exchange.fetch_ohlcv(f'{coin}/USDT','3m',limit=1000)
        df = pd.DataFrame(candles,columns=['ts','open','high','low','close','volume'])
        df['ts']=pd.to_datetime(df['ts'],unit='ms')
        all_data[coin]=df
    except:
        errors+=1
    if (i+1)%50==0:
        elapsed = time.time()-t0
        print(f"  ⏳ {i+1}/{len(COINS)} ({elapsed:.0f}s) ...")

elapsed = time.time()-t0
print(f"  ✅ {len(all_data)}/{len(COINS)} عملة ({elapsed:.0f}s) | ❌ {errors} أخطاء\n")

# ═══════════════ تشغيل التجارب ═══════════════
print(f"{'='*80}")
print("🧪 8 تجارب × 212 عملة — 3m")
print(f"{'='*80}")

all_results = []

for name, tp, sl, pl, trail, max_h, whale, rsi, confirm in TESTS:
    all_trades = []
    coins_with_trades = 0
    
    for coin in COINS:
        df = all_data.get(coin)
        if df is None: continue
        df_w = compute_indicators(df, whale)
        signals = find_signals(df_w, coin, whale, rsi, confirm)
        if not signals: continue
        trades = simulate(df_w, signals, tp, sl, pl, trail, max_h)
        if trades:
            coins_with_trades += 1
            all_trades.extend(trades)
    
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
    
    # Sharpe
    returns = [t['exit'][1] for t in all_trades]
    sharpe = np.mean(returns)/np.std(returns)*np.sqrt(len(returns)) if len(returns)>1 and np.std(returns)>0 else 0
    
    # DD
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    dd = np.min(cumulative - peak)
    
    all_results.append({
        'name':name, 'coins':coins_with_trades, 'trades':len(all_trades),
        'wins':len(wins), 'wr':wr, 'net':net,
        'avg_win':avg_win, 'avg_loss':avg_loss,
        'tp':tp_c, 'sl':sl_c, 'trail':tr_c, 'time':tm_c,
        'sharpe':sharpe, 'dd':dd,
    })
    
    print(f"\n📊 {name}")
    print(f"   ⚙️ TP={tp}% SL={sl}% PL={pl}% TRAIL={trail}% MAX_H={max_h}h 🐋≥{whale} RSI<{rsi} {'✓تأكيد' if confirm else ''}")
    print(f"   🪙 {coins_with_trades} عملة | 📋 {len(all_trades)} صفقة | 🟢{len(wins)} 🔴{len(losses)}")
    print(f"   📈 WR: {wr:.1f}% | 💰 صافي: {net:+.1f}% | 📊 شارپ: {sharpe:.2f} | 📉 DD: {dd:.1f}%")
    print(f"   🟢 +{avg_win:.2f}% | 🔴 {avg_loss:.2f}%")
    print(f"   🎯TP:{tp_c} 🛑SL:{sl_c} 🐌TRAIL:{tr_c} ⏰TIME:{tm_c}")

# ═══════════════ الترتيب ═══════════════
print(f"\n{'='*80}")
print("⚖️ ترتيب حسب WR")
print(f"{'='*80}")
sorted_results = sorted(all_results, key=lambda x: x['wr'], reverse=True)
print(f"  {'التجربة':<18} {'عملات':>5} {'صفقات':>6} {'WR':>7} {'صافي':>8} {'شارپ':>6} {'DD':>7} {'TP/SL/TR/TM'}")
print(f"  {'─'*18} {'─'*5} {'─'*6} {'─'*7} {'─'*8} {'─'*6} {'─'*7} {'─'*14}")
for r in sorted_results:
    exits = f"{r['tp']}/{r['sl']}/{r['trail']}/{r['time']}"
    print(f"  {r['name']:<18} {r['coins']:>5} {r['trades']:>6} {r['wr']:>6.1f}% {r['net']:>+7.1f}% {r['sharpe']:>6.2f} {r['dd']:>6.1f}% {exits}")
