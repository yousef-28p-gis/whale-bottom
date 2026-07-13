#!/usr/bin/env python3
"""May 1 - Jul 8 2026 period analysis"""
import pandas as pd, numpy as np

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv', parse_dates=['ts'])
mask = (df['ts'] >= '2026-05-01') & (df['ts'] <= '2026-07-08')
tdf = df[mask].copy().reset_index(drop=True)
print(f"Period: {tdf['ts'].iloc[0]} → {tdf['ts'].iloc[-1]} | Candles: {len(tdf)}")

# Whale
lowest = tdf['low'].rolling(200, min_periods=1).min()
at_low = (tdf['low'] <= lowest).astype(float)
lc = abs(tdf['low'] - tdf['low'].shift(1)) / tdf['low'] * 100
sm = lc.ewm(span=3, adjust=False).mean()
hi = sm.rolling(200, min_periods=1).max()
st = np.where(at_low > 0, (sm + hi * 2) / 3, 0)
tdf['whale'] = pd.Series(st).ewm(span=3, adjust=False).mean().fillna(0)
tdf['spike'] = (tdf['whale'] > tdf['whale'].shift(1)) & (tdf['whale'].shift(1) <= 0.02)
tdf['wma50'] = tdf['whale'].rolling(50, min_periods=1).mean()
tdf['wma200'] = tdf['whale'].rolling(200, min_periods=1).mean()
tdf['wstr'] = tdf['whale'] / tdf['whale'].rolling(50, min_periods=1).max().replace(0, np.nan) * 100
tdf['atr'] = (tdf['high'] - tdf['low']).rolling(14).mean()
tdf['vma'] = tdf['volume'].rolling(20, min_periods=1).mean()

delta = tdf['close'].diff(); g = delta.clip(lower=0); l = -delta.clip(upper=0)
ag = g.ewm(alpha=1/14, adjust=False).mean(); al = l.ewm(alpha=1/14, adjust=False).mean()
tdf['rsi'] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
vs = tdf['volume'].rolling(20, min_periods=1).mean()
hh20 = tdf['high'].rolling(20, min_periods=1).max().shift(1)
ll10 = tdf['low'].rolling(10, min_periods=1).min().shift(1)
c = np.zeros(len(tdf))
c += ((tdf['volume'] > vs * 1.5) & (tdf['close'] < tdf['open'])).astype(int)
c += ((tdf['high'] > hh20) & (tdf['close'] < hh20)).astype(int)
c += ((tdf['high'] > hh20) & (tdf['close'] < tdf['open'])).astype(int)
c += ((tdf['close'].shift(1) > tdf['open'].shift(1)) & (tdf['volume'] > vs * 1.5) & (tdf['close'] < tdf['open'])).astype(int)
c += (tdf['low'] < ll10).astype(int)
c += ((tdf['high'] > tdf['high'].shift(1)) & (tdf['rsi'] < tdf['rsi'].shift(1))).astype(int)
tdf['sell_str'] = c / 6 * 100

lb = 5; swl = np.zeros(len(tdf), dtype=bool)
for i in range(lb*2, len(tdf)):
    w = tdf['low'].iloc[i-lb*2:i+1]; m = i - lb
    if tdf['low'].iloc[m] == w.min() and w.values.argmax() == lb: swl[i] = True

def nsl(idx):
    for j in range(idx-1, max(0, idx-100), -1):
        if swl[j]: return tdf['low'].iloc[j]
    return tdf['low'].iloc[idx] * 0.95

long_ok = tdf['wma50'] > tdf['wma200']
entry_sig = (tdf['spike'] & (tdf['wstr'] > 50) & long_ok &
             (tdf['volume'] > tdf['vma']) & (tdf['atr'] > tdf['atr'].rolling(20).mean()))
entry_idxs = np.where(entry_sig)[0]
print(f"Entry signals: {len(entry_idxs)}")

# Show each entry
for ei in entry_idxs:
    if ei < 10: continue
    print(f"  {tdf['ts'].iloc[ei]} | Px: ${tdf['close'].iloc[ei]:.4f} | Str: {tdf['wstr'].iloc[ei]:.0f}% | Wh: {tdf['whale'].iloc[ei]:.3f} | VolOK:{tdf['volume'].iloc[ei]>tdf['vma'].iloc[ei]} | AtrOK:{tdf['atr'].iloc[ei]>tdf['atr'].rolling(20).mean().iloc[ei]}")

# Simulate
trades=[]; it=False; ed=0; eq=1000
for ei in entry_idxs:
    if ei<10: continue
    if it and ei<ed: continue
    e=tdf['close'].iloc[ei]; sl=nsl(ei)*0.998
    end=min(ei+192,len(tdf)); r=None; ep=e; exi=ei
    for j in range(ei+1,end):
        if tdf['low'].iloc[j]<=sl: r='SL'; ep=sl; exi=j; break
        if tdf['sell_str'].iloc[j]>=60: r='SELL'; ep=tdf['close'].iloc[j]; exi=j; break
    if not r: r='TIME'; ep=tdf['close'].iloc[end-1]; exi=end-1
    pnl=(ep-e)/e*100-0.2
    trades.append({'ets':tdf['ts'].iloc[ei],'ep':e,'xts':tdf['ts'].iloc[exi],'xp':ep,'r':r,'pnl':pnl,'sl':sl})
    it=True; ed=exi; eq+=1000*(pnl/100)

n=len(trades); wins=[t for t in trades if t['pnl']>0]
print(f"\n--- RESULTS ---")
print(f"Trades: {n} | Wins: {len(wins)} | Losses: {n-len(wins)} | WR: {len(wins)/n*100:.0f}% | Portfolio: ${eq:,.0f}")
for t in trades:
    e = "🟢" if t['pnl']>0 else "🔴"
    print(f"  {e} {str(t['ets'])[:16]} → {str(t['xts'])[:16]} | {t['ep']:.4f}→{t['xp']:.4f} | {t['r']:>4} | {t['pnl']:+.2f}% | SL:{t['sl']:.4f}")
