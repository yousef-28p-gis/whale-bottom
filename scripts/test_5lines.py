import ccxt, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

exchange = ccxt.binance({'timeout': 15000})
symbol = 'BTC/USDT'
df = pd.DataFrame(exchange.fetch_ohlcv(symbol, '15m', limit=500),
                  columns=['ts','open','high','low','close','volume'])
df['ts'] = pd.to_datetime(df['ts'], unit='ms')
df.set_index('ts', inplace=True)

# 1. Supertrend
def supertrend(df, period=10, factor=3.0):
    tr = pd.concat([df['high']-df['low'],
                    abs(df['high']-df['close'].shift()),
                    abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr
    trend = pd.Series(1, index=df.index)
    st_line = pd.Series(np.nan, index=df.index)
    for i in range(period, len(df)):
        if df['close'].iloc[i-1] > upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif df['close'].iloc[i-1] < lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
            if trend.iloc[i] == 1 and lower.iloc[i] < lower.iloc[i-1]:
                lower.iloc[i] = lower.iloc[i-1]
            if trend.iloc[i] == -1 and upper.iloc[i] > upper.iloc[i-1]:
                upper.iloc[i] = upper.iloc[i-1]
        st_line.iloc[i] = lower.iloc[i] if trend.iloc[i] == 1 else upper.iloc[i]
    return st_line, trend

df['st_line'], df['st_trend'] = supertrend(df)

# EMAs
df['ema9'] = df['close'].ewm(span=9).mean()
df['ema21'] = df['close'].ewm(span=21).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['ema200'] = df['close'].ewm(span=200).mean()

# Slope
df['slope'] = (df['ema50'] - df['ema50'].shift(5)) / df['ema50'].shift(5) * 100
df['trend_up'] = df['slope'] > 0.08
df['trend_down'] = df['slope'] < -0.08

# Cross
df['buy_x'] = (df['close'] > df['ema9']) & (df['close'].shift(1) <= df['ema9'].shift(1))
df['sell_x'] = (df['close'] < df['ema21']) & (df['close'].shift(1) >= df['ema21'].shift(1))

# Plot
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

width = 0.0004
for i, (idx, row) in enumerate(df.iterrows()):
    c = '#00e676' if row['close'] >= row['open'] else '#ff1744'
    ax.plot([idx, idx], [row['low'], row['high']], color=c, linewidth=0.8)
    body_h = abs(row['close'] - row['open'])
    body_b = min(row['open'], row['close'])
    ax.add_patch(plt.Rectangle((mdates.date2num(idx)-width/2, body_b), width, body_h,
                                facecolor=c, edgecolor='none'))

ax.plot(df.index, df['st_line'], color='#00e676', linewidth=2, label='1. Stop (Supertrend)')
ax.plot(df.index, df['ema9'], color='#2979ff', linewidth=2, label='2. Buy Cross (EMA 9)')
ax.plot(df.index, df['ema21'], color='#ff9100', linewidth=2, label='3. Sell Cross (EMA 21)')

for i in range(5, len(df)-1):
    c = '#00e676' if df['trend_up'].iloc[i] else '#ff1744' if df['trend_down'].iloc[i] else '#9e9e9e'
    ax.plot(df.index[i:i+2], df['ema50'].iloc[i:i+2], color=c, linewidth=2.5)

ax.plot([], [], color='#9e9e9e', linewidth=2.5, label='4. Trend/Range (EMA 50)')
ax.plot(df.index, df['ema200'], color='#ffffff', linewidth=1, alpha=0.4, label='5. Ref (EMA 200)')

buy_i = df[df['buy_x']].index
sell_i = df[df['sell_x']].index
ax.scatter(buy_i, df.loc[buy_i, 'low']*0.998, marker='^', s=80, color='#00e676',
           edgecolors='white', linewidths=0.5, zorder=5, label='Buy')
ax.scatter(sell_i, df.loc[sell_i, 'high']*1.002, marker='v', s=80, color='#ff1744',
           edgecolors='white', linewidths=0.5, zorder=5, label='Sell')

ax.set_title(f'{symbol} 15m — 5 Lines Indicator + Multi-TF Concept Test', color='white', fontsize=14)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
ax.tick_params(colors='white', labelsize=8)
ax.grid(alpha=0.1, color='white')
ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=9, loc='upper left')

n_buy = int(df['buy_x'].sum())
n_sell = int(df['sell_x'].sum())
tp = (df['trend_up'].sum()/len(df)*100)
rp = ((~df['trend_up'] & ~df['trend_down']).sum()/len(df)*100)
stats = f"Price: ${df['close'].iloc[-1]:,.2f} | Buys: {n_buy} | Sells: {n_sell} | Trend: {tp:.0f}% | Range: {rp:.0f}%"
ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=8, color='#8b949e', family='monospace')

plt.tight_layout()
out = '/data/trading28/charts/5lines_test.png'
os.makedirs('/data/trading28/charts', exist_ok=True)
plt.savefig(out, dpi=120, facecolor='#0d1117', bbox_inches='tight')
plt.close()
print(f'OK: {out}')
print(f'{df.index[0]} -> {df.index[-1]} ({len(df)} candles)')
print(f'Buys={n_buy} Sells={n_sell} Trend%={tp:.1f} Range%={rp:.1f}')
