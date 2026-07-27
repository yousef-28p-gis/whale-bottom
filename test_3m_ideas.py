#!/usr/bin/env python3
"""🧪 6 تجارب على فريم 3m — استراتيجية جديدة"""
import ccxt, numpy as np, pandas as pd

COINS = ['BTC','ETH','BNB','SOL','ADA','DOGE','AVAX','DOT','LINK','MATIC']
TF = '3m'; TF_MIN = 3; COMM = 0.20

# ═══════════════ التجارب ═══════════════
TESTS = [
    #  (name,      TP,  SL,  PL, TRAIL, MAX_H, WHALE, RSI, confirm, exit_bars)
    ("أساسي",      3.5, 1.5, 30, 0.10,  6,     0.05,  50,  False,  1),
    ("هدف 2%",     2.0, 1.5, 30, 0.10,  6,     0.05,  50,  False,  1),
    ("هدف 1.5%",   1.5, 1.0, 20, 0.05,  6,     0.05,  50,  False,  1),
    ("فلاتر أضيق", 3.5, 1.5, 30, 0.10,  6,     0.10,  35,  False,  1),
    ("خروج 6m",    3.5, 1.5, 30, 0.10,  6,     0.05,  50,  False,  2),
    ("بتأكيد",     3.5, 1.5, 30, 0.10,  6,     0.05,  50,  True,   1),
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
    df['entry'] = (df['whale']>=whale_min) & (df['spike']>=1.5)
    delta = df['close'].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain/loss.replace(0,np.nan)
    df['rsi'] = 100-(100/(1+rs))
    return df

def find_signals(df, symbol, rsi_max, confirm):
    if df is None or len(df)<100:
        return []
    signals = []
    for i in range(50, len(df)):
        row = df.iloc[i]
        if not row['entry']: continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi>=rsi_max: continue
        
        # تأكيد شمعة خضراء
        if confirm:
            if i+1>=len(df): continue
            if df.iloc[i+1]['close']<=df.iloc[i+1]['open']: continue
        
        # تباعد
        if i-1>=0 and df.iloc[i-1]['entry']: continue
        if i-2>=0 and df.iloc[i-2]['entry']: continue
        if i-3>=0 and df.iloc[i-3]['entry']: continue
        
        signals.append({
            'idx':i, 'symbol':symbol,
            'entry_price':round(float(row['close']),8),
        })
    return signals

def simulate(df, signals, tp, sl, pl, trail, max_h, exit_bars):
    trades = []
    active = []
    max_bars = int(max_h*60/TF_MIN)
    tp_pct = tp; sl_pct = sl
    
    for i in range(len(df)):
        if i%exit_bars!=0: continue
        row = df.iloc[i]
        current = float(row['close'])
        
        for sig in signals:
            if sig['idx']==i:
                active.append({
                    'symbol':sig['symbol'],
                    'entry':sig['entry_price'],
                    'tp':sig['entry_price']*(1+tp_pct/100),
                    'sl':sig['entry_price']*(1-sl_pct/100),
                    'pl_triggered':False, 'peak':sig['entry_price'],
                    'trail_price':sig['entry_price'],
                    'entry_idx':i,
                })
        
        for pos in active[:]:
            entry = pos['entry']
            bars_held = i-pos['entry_idx']
            if bars_held>=max_bars:
                pnl = round((current-entry)/entry*100-COMM,4)
                pos['exit']=('TIME',pnl); active.remove(pos); trades.append(pos); continue
            if current>=pos['tp']:
                pos['exit']=('TP',round(tp_pct-COMM,4)); active.remove(pos); trades.append(pos); continue
            if current<=pos['sl']:
                pos['exit']=('SL',round(-sl_pct-COMM,4)); active.remove(pos); trades.append(pos); continue
            if pos.get('pl_triggered'):
                if current>pos.get('peak',entry):
                    pos['peak']=current
                    pos['trail_price']=current*(1-trail/100)
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
exchange = ccxt.binance({'timeout':15000,'enableRateLimit':True})

print("⏳ جلب بيانات 3m...")
all_dfs = {}
for coin in COINS:
    try:
        candles = exchange.fetch_ohlcv(f'{coin}/USDT','3m',limit=1000)
        df = pd.DataFrame(candles,columns=['ts','open','high','low','close','volume'])
        df['ts']=pd.to_datetime(df['ts'],unit='ms')
        all_dfs[coin]=df
        print(f"  ✅ {coin}: {len(df)} شمعة")
    except Exception as e:
        print(f"  ❌ {coin}: {e}")

# ═══════════════ تشغيل التجارب ═══════════════
print(f"\n{'='*75}")
print("🧪 6 تجارب على 3m")
print(f"{'='*75}")

all_results = []

for name, tp, sl, pl, trail, max_h, whale, rsi, confirm, exit_bars in TESTS:
    all_trades = []
    for coin in COINS:
        df = all_dfs.get(coin)
        if df is None: continue
        df_w = compute_indicators(df, whale)
        signals = find_signals(df_w, coin, rsi, confirm)
        if not signals: continue
        trades = simulate(df_w, signals, tp, sl, pl, trail, max_h, exit_bars)
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
    
    all_results.append({
        'name':name, 'trades':len(all_trades), 'wr':wr, 'net':net,
        'avg_win':avg_win, 'avg_loss':avg_loss,
        'tp':tp_c, 'sl':sl_c, 'trail':tr_c, 'time':tm_c
    })
    
    print(f"\n📊 {name}")
    print(f"   ⚙️ TP={tp}% SL={sl}% PL={pl}% TRAIL={trail}% MAX_H={max_h}h WHALE≥{whale} RSI<{rsi} {'تأكيد' if confirm else 'بدون تأكيد'} exit={exit_bars*TF_MIN}m")
    print(f"   📋 {len(all_trades)} صفقة | 🟢{len(wins)} 🔴{len(losses)}")
    print(f"   📈 WR: {wr:.1f}% | 💰 صافي: {net:+.1f}%")
    print(f"   🟢 +{avg_win:.2f}% | 🔴 {avg_loss:.2f}%")
    print(f"   🎯TP:{tp_c} 🛑SL:{sl_c} 🐌TRAIL:{tr_c} ⏰TIME:{tm_c}")

# ═══════════════ مقارنة ═══════════════
print(f"\n{'='*75}")
print("⚖️ ترتيب حسب WR")
print(f"{'='*75}")
sorted_results = sorted(all_results, key=lambda x: x['wr'], reverse=True)
print(f"  {'التجربة':<15} {'صفقات':>6} {'WR':>7} {'صافي':>8} {'TP':>4} {'SL':>4} {'TRAIL':>6} {'TIME':>5}")
print(f"  {'─'*15} {'─'*6} {'─'*7} {'─'*8} {'─'*4} {'─'*4} {'─'*6} {'─'*5}")
for r in sorted_results:
    print(f"  {r['name']:<15} {r['trades']:>6} {r['wr']:>6.1f}% {r['net']:>+7.1f}% {r['tp']:>4} {r['sl']:>4} {r['trail']:>6} {r['time']:>5}")
