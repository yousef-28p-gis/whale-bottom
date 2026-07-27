#!/usr/bin/env python3
"""🧪 باك تيست 7 أيام — 3m — كل العملات الحلال"""
import ccxt, json, numpy as np, pandas as pd, time
from datetime import datetime, timezone, timedelta

COMM = 0.20; TF = '3m'; TF_MIN = 3

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

# ═══════════════ التجارب ═══════════════
TESTS = [
    # (name,           TP,  SL,  PL, TRAIL, MAX_H, WHALE, RSI, confirm)
    ("TP2_بتأكيد",      2.0, 1.5, 30, 0.10,  6,     0.05,  50,  True),
    ("TP2_أساسي",       2.0, 1.5, 30, 0.10,  6,     0.05,  50,  False),
    ("TP1.5_تريل0.1",   1.5, 1.0, 20, 0.10,  6,     0.05,  50,  False),
    ("TP2_فلاترأضيق",   2.0, 1.5, 30, 0.10,  6,     0.10,  35,  False),
    ("TP2.5_وسط",       2.5, 1.5, 30, 0.10,  6,     0.05,  50,  False),
]

def compute_indicators(df, whale_min):
    df = df.copy()
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    df['whale'] = (df['low']-df['low_raw'])/df['low_raw'].replace(0,np.nan)*100
    df['whale'] = df['whale'].clip(lower=0)
    df['spike'] = df['volume']/df['volume'].rolling(20).mean().replace(0,np.nan)
    df['hi_raw'] = df['high'].rolling(50).max()
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
    if df is None or len(df)<100: return []
    signals = []
    for i in range(50, len(df)):
        row = df.iloc[i]
        if not row['entry_raw']: continue
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi>=rsi_max: continue
        if confirm:
            if i+1>=len(df): continue
            if df.iloc[i+1]['close']<=df.iloc[i+1]['open']: continue
        skip=False
        for j in range(1,5):
            if i-j>=0 and df.iloc[i-j]['entry_raw']: skip=True; break
        if skip: continue
        signals.append({'idx':i, 'entry_price':round(float(row['close']),8)})
    return signals

def simulate(df, signals, tp, sl, pl, trail, max_h):
    trades = []; active = []; max_bars = int(max_h*60/TF_MIN)
    for i in range(len(df)):
        row = df.iloc[i]; current = float(row['close'])
        for sig in signals:
            if sig['idx']==i and len(active)<2:
                ep = sig['entry_price']
                active.append({'entry':ep, 'tp':ep*(1+tp/100), 'sl':ep*(1-sl/100),
                    'pl_triggered':False, 'peak':ep, 'trail_price':ep, 'entry_idx':i})
        for pos in active[:]:
            e=pos['entry']; bh=i-pos['entry_idx']
            if bh>=max_bars:
                pnl=round((current-e)/e*100-COMM,4); pos['exit']=('TIME',pnl); active.remove(pos); trades.append(pos); continue
            if current>=pos['tp']:
                pos['exit']=('TP',round(tp-COMM,4)); active.remove(pos); trades.append(pos); continue
            if current<=pos['sl']:
                pos['exit']=('SL',round(-sl-COMM,4)); active.remove(pos); trades.append(pos); continue
            if pos.get('pl_triggered'):
                if current>pos.get('peak',e): pos['peak']=current; pos['trail_price']=current*(1-trail/100)
                if current<=pos.get('trail_price',0):
                    plr=round((pos['trail_price']-e)/e*100-COMM,4); pos['exit']=('TRAIL',plr); active.remove(pos); trades.append(pos)
            else:
                pl_price=e+(pos['tp']-e)*(pl/100)
                if current>=pl_price: pos['pl_triggered']=True; pos['peak']=current; pos['trail_price']=current*(1-trail/100)
    return trades

# ═══════════════ MAIN ═══════════════
print("⏳ تحميل قائمة العملات...")
with open('/data/trading28/config/shariah_coins.json') as f:
    shariah = json.load(f)
COINS = [c for c in shariah['halal']+shariah['halal2'] if c not in STABLES]
print(f"📋 {len(COINS)} عملة حلال\n")

exchange = ccxt.binance({'timeout':10000,'enableRateLimit':True})

# جلب 7 أيام من 3m (3360 شمعة — 4 دفعات)
print("⏳ جلب 7 أيام 3m...")
all_data = {}
errors = 0
t0 = time.time()
since_7d = int((datetime.now(timezone.utc)-timedelta(days=7)).timestamp()*1000)

for i, coin in enumerate(COINS):
    try:
        all_candles = []
        since = since_7d
        for batch in range(4):
            candles = exchange.fetch_ohlcv(f'{coin}/USDT', TF, since=since, limit=1000)
            if not candles: break
            all_candles.extend(candles)
            since = candles[-1][0] + 1  # next candle
            if len(candles) < 1000: break
        if all_candles:
            df = pd.DataFrame(all_candles, columns=['ts','open','high','low','close','volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df = df.drop_duplicates(subset=['ts']).sort_values('ts').reset_index(drop=True)
            all_data[coin] = df
    except:
        errors += 1
    if (i+1)%50==0:
        print(f"  ⏳ {i+1}/{len(COINS)} ({time.time()-t0:.0f}s) ...")

print(f"  ✅ {len(all_data)}/{len(COINS)} عملة ({time.time()-t0:.0f}s) | ❌ {errors} أخطاء")
total_candles = sum(len(df) for df in all_data.values())
total_hours = total_candles * 3 / 60
print(f"  📊 {total_candles:,} شمعة = {total_hours:,.0f} ساعة\n")

# ═══════════════ تشغيل التجارب ═══════════════
print(f"{'='*80}")
print("🧪 5 تجارب × 7 أيام — 3m")
print(f"{'='*80}")

all_results = []

for name, tp, sl, pl, trail, max_h, whale, rsi, confirm in TESTS:
    all_trades = []
    coins_with_trades = 0
    
    for coin in COINS:
        df = all_data.get(coin)
        if df is None: continue
        df_w = compute_indicators(df, whale)
        signals = find_signals(df_w, rsi, confirm)
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
    
    returns = [t['exit'][1] for t in all_trades]
    sharpe = np.mean(returns)/np.std(returns)*np.sqrt(len(returns)) if len(returns)>1 and np.std(returns)>0 else 0
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    dd = np.min(cumulative - peak)
    portfolio = 1000 * np.prod([1+r/100 for r in returns])
    
    all_results.append({
        'name':name, 'coins':coins_with_trades, 'trades':len(all_trades),
        'wins':len(wins), 'wr':wr, 'net':net,
        'avg_win':avg_win, 'avg_loss':avg_loss,
        'tp':tp_c, 'sl':sl_c, 'trail':tr_c, 'time':tm_c,
        'sharpe':sharpe, 'dd':dd, 'portfolio':portfolio,
    })
    
    print(f"\n📊 {name}")
    print(f"   ⚙️ TP={tp}% SL={sl}% PL={pl}% TRAIL={trail}% 🐋≥{whale} RSI<{rsi} {'✓تأكيد' if confirm else ''}")
    print(f"   🪙 {coins_with_trades} عملة | 📋 {len(all_trades)} صفقة | 🟢{len(wins)} 🔴{len(losses)}")
    print(f"   📈 WR: {wr:.1f}% | 💰 صافي: {net:+.1f}% | 📊 شارپ: {sharpe:.2f} | 📉 DD: {dd:.1f}%")
    print(f"   🟢 +{avg_win:.2f}% | 🔴 {avg_loss:.2f}%")
    print(f"   🎯TP:{tp_c} 🛑SL:{sl_c} 🐌TRAIL:{tr_c} ⏰TIME:{tm_c}")
    print(f"   🏦 $1,000 → ${portfolio:,.0f}")

# ═══════════════ الترتيب ═══════════════
print(f"\n{'='*80}")
print("⚖️ ترتيب حسب WR — 7 أيام 3m")
print(f"{'='*80}")
sorted_results = sorted(all_results, key=lambda x: x['wr'], reverse=True)
print(f"  {'التجربة':<18} {'عملات':>5} {'صفقات':>6} {'WR':>7} {'صافي':>8} {'شارپ':>6} {'DD':>7} {'محفظة':>10} {'TP/SL/TR/TM'}")
print(f"  {'─'*18} {'─'*5} {'─'*6} {'─'*7} {'─'*8} {'─'*6} {'─'*7} {'─'*10} {'─'*14}")
for r in sorted_results:
    exits = f"{r['tp']}/{r['sl']}/{r['trail']}/{r['time']}"
    port = f"${r['portfolio']:,.0f}"
    print(f"  {r['name']:<18} {r['coins']:>5} {r['trades']:>6} {r['wr']:>6.1f}% {r['net']:>+7.1f}% {r['sharpe']:>6.2f} {r['dd']:>6.1f}% {port:>10} {exits}")
