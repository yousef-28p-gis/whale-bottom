#!/usr/bin/env python3
import json, numpy as np, pandas as pd, os
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

CACHE = '/data/trading28/cache/ohlcv'
OUT = '/data/trading28/charts'
os.makedirs(OUT, exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('dark_background')
TP, SL, PL, TRAIL, MAX_HOURS = 2.5, 2.0, 40, 0.10, 2

WINNERS = [('DEXE','2026-07-09T21:36:00'),('NFP','2026-07-05T00:36:00'),('BEL','2026-07-05T15:27:00'),('TLM','2026-07-02T01:50:00'),('XPL','2026-07-02T08:21:00')]
LOSERS = [('DYDX','2026-07-01T06:47:00'),('ZAMA','2026-07-02T11:21:00'),('KAITO','2026-07-02T04:31:00'),('PENDLE','2026-07-05T07:39:00'),('OPG','2026-07-07T07:40:00')]

def load_cached(sym, mon):
    fpath = f'{CACHE}/{sym}_{mon}.json'
    if not os.path.exists(fpath): return None
    with open(fpath) as f: data = json.load(f)
    df = pd.DataFrame(data)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    return df.sort_values('ts').reset_index(drop=True)

def whale_indicator(df):
    df = df.copy()
    LB, WF, WS, VM = 30, 2, 5, 1.0
    df['lo'] = df['low'].rolling(LB).min()
    df['lc'] = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
    df['sm'] = df['lc'].ewm(span=3, adjust=False).mean()
    df['hi'] = df['sm'].rolling(LB).max()
    df['raw'] = np.where(df['low'] <= df['lo'], (df['sm'] + df['hi'] * 2) / 3, 0)
    df['whale'] = df['raw'].ewm(span=3, adjust=False).mean().fillna(0)
    df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.03)
    df['wf'] = df['whale'].rolling(WF).mean()
    df['ws'] = df['whale'].rolling(WS).mean()
    df['wp'] = df['whale'].rolling(50).max()
    df['str'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
    df['vma'] = df['volume'].rolling(20).mean()
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) & (df['str'] > 50) & (df['volume'] > df['vma'] * VM))
    return df

def find_entry(df_w, signal_dt):
    df_w = df_w.copy()
    df_w['td'] = abs((df_w['ts'] - signal_dt).dt.total_seconds())
    nearest = df_w['td'].idxmin()
    for j in range(min(len(df_w) - nearest, 96)):
        idx = nearest + j
        if idx < len(df_w) and df_w.iloc[idx]['entry']:
            wv = float(df_w.iloc[idx]['whale'])
            if wv >= 0.35:
                return idx, float(df_w.iloc[idx]['close']), df_w.iloc[idx]['ts'], wv
    return None

def sim_exit(df_w, entry_idx, entry_price):
    tp_p = entry_price * (1 + TP/100)
    sl_p = entry_price * (1 - SL/100)
    pl_p = entry_price + (tp_p - entry_price) * (PL/100)
    pl_trig = False; peak = entry_price; trail_p = 0; first = None
    for j in range(entry_idx+1, min(len(df_w), entry_idx+96)):
        c = df_w.iloc[j]; h = (j-entry_idx)*0.25
        if first is None:
            if c['high'] >= tp_p: first = 'tp'
            elif c['low'] <= sl_p: first = 'sl'
        if h > MAX_HOURS:
            pnl = round((c['close']-entry_price)/entry_price*100, 4)
            return j, 'timeout', pnl, first
        if not pl_trig and c['high'] >= pl_p:
            pl_trig = True; peak = c['high']; trail_p = c['high'] * (1-TRAIL/100)
        if pl_trig:
            if c['high'] > peak: peak = c['high']; trail_p = c['high'] * (1-TRAIL/100)
            if c['low'] <= trail_p:
                pnl = round((trail_p-entry_price)/entry_price*100, 4)
                return j, 'trail', pnl, first
        if c['high'] >= tp_p: return j, 'tp', round(TP,4), 'tp'
        if c['low'] <= sl_p: return j, 'sl', round(-SL,4), 'sl'
    j = min(len(df_w)-1, entry_idx+96)
    pnl = round((df_w.iloc[j]['close']-entry_price)/entry_price*100, 4)
    return j, 'eod', pnl, first

def make_chart(sym, sdt_str, label):
    sdt = datetime.fromisoformat(sdt_str)
    mon = sdt.strftime('%Y-%m')
    df = load_cached(sym, mon)
    if df is None: return print(f'  SKIP {sym}: no cache')
    df_w = whale_indicator(df)
    r = find_entry(df_w, sdt)
    if r is None: return print(f'  SKIP {sym}: no entry')
    eidx, eprice, cts, wv = r
    xidx, xreason, xpnl, first = sim_exit(df_w, eidx, eprice)
    xts = df_w.iloc[xidx]['ts']
    tp_p = eprice*(1+TP/100); sl_p = eprice*(1-SL/100)
    si = max(0, eidx-16); ei = min(len(df_w), xidx+16)
    cdf = df_w.iloc[si:ei]; dts = cdf['ts'].values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,8), gridspec_kw={'height_ratios':[3,1]}, sharex=True, facecolor='#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    for i in range(len(cdf)):
        row = cdf.iloc[i]; clr = '#00ff88' if row['close']>=row['open'] else '#ff4444'
        ax1.plot([dts[i],dts[i]], [row['low'],row['high']], color=clr, linewidth=0.8)
        bh = abs(row['close']-row['open']); bb = min(row['open'],row['close'])
        ax1.add_patch(plt.Rectangle((mdates.date2num(dts[i])-0.003, bb), 0.006, max(bh,0.000001), facecolor=clr, edgecolor='none'))
    ax1.axhline(eprice, color='white', linestyle='--', linewidth=1)
    ax1.axhline(tp_p, color='#00ff88', linestyle=':', linewidth=1, alpha=0.6)
    ax1.axhline(sl_p, color='#ff4444', linestyle=':', linewidth=1, alpha=0.6)
    ax1.axvline(cts, color='cyan', linewidth=2, alpha=0.8)
    ax1.axvline(sdt, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axvline(xts, color='#ff8800' if xpnl>0 else '#ff4444', linewidth=2, alpha=0.9)
    ax1.fill_between(dts, eprice, tp_p, alpha=0.05, color='green')
    ax1.fill_between(dts, sl_p, eprice, alpha=0.05, color='red')
    dm = (cts-sdt).total_seconds()/60
    em = 'WIN' if xpnl>0 else 'LOSS'; tc = '#00ff88' if xpnl>0 else '#ff4444'
    ax1.set_title(f'{em} {sym} | Signal:{sdt.strftime("%m/%d %H:%M")} Confirm:+{dm:.0f}m Whale:{wv:.3f} PnL:{xpnl:+.2f}%({xreason})', color=tc, fontsize=11, fontweight='bold')
    ax1.text(0.01,0.98,f'Entry:{eprice:.6f}',transform=ax1.transAxes,color='white',fontsize=9,va='top')
    ax1.text(0.01,0.93,f'TP:{tp_p:.6f}',transform=ax1.transAxes,color='#00ff88',fontsize=8,va='top')
    ax1.text(0.01,0.88,f'SL:{sl_p:.6f}',transform=ax1.transAxes,color='#ff4444',fontsize=8,va='top')
    if first:
        fc = 'yellow' if first=='tp' else '#ff4444'
        ax1.text(0.99,0.98,f'First:{first.upper()}',transform=ax1.transAxes,color=fc,fontsize=9,va='top',ha='right')
    ax1.set_ylabel('Price',color='white'); ax1.tick_params(colors='white')
    ax1.grid(True,alpha=0.15,color='white'); ax1.set_facecolor('#1a1a2e')
    cv = ['#00ff88' if cdf.iloc[i]['close']>=cdf.iloc[i]['open'] else '#ff4444' for i in range(len(cdf))]
    ax2.bar(dts,cdf['volume'],color=cv,width=0.005,alpha=0.7)
    ax2.set_ylabel('Vol',color='white',fontsize=9); ax2.tick_params(colors='white',labelsize=8)
    ax2.grid(True,alpha=0.15,color='white'); ax2.set_facecolor('#1a1a2e')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(),rotation=30,ha='right')
    plt.tight_layout()
    fn = f'{OUT}/{label}_{sym}.png'
    fig.savefig(fn,dpi=150,facecolor='#1a1a2e',bbox_inches='tight')
    plt.close(fig)
    print(f'  OK {sym}: {xpnl:+.2f}% ({xreason}) first={first}')

print('WINNERS:')
for s,d in WINNERS: make_chart(s,d,'winner')
print('LOSERS:')
for s,d in LOSERS: make_chart(s,d,'loser')
print(f'DONE -> {OUT}/')
