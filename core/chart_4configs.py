"""4 Configs x 10 days — 4 charts"""
import sys; sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('backtests/cache/FET_USDT_15m_10d.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
CAP=1000.0; n=len(df)

price_start = df['close'].iloc[0]
price_end = df['close'].iloc[-1]
price_chg = (price_end/price_start-1)*100

configs = [
    ('🛡️ v10 آمن', 200, 20, 50, 50, True, True),
    ('🛡️ آمن بلا حجم', 200, 20, 50, 50, False, True),
    ('⚡ وسط', 50, 5, 10, 30, False, True),
    ('🔥 شرس', 50, 3, 10, 10, False, False),
]

def run_backtest(lb, wf, ws, smin, use_vol, use_sma50):
    whale = whale_indicator(df, lb)
    entry = (whale_spike(whale) & (whale_ma(whale, wf) > whale_ma(whale, ws)) &
             (whale_strength(whale, 50) > smin))
    if use_vol: entry &= volume_filter(df)
    if use_sma50: entry &= (df['close'] > sma50_daily(df))
    
    ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)
    capital=CAP; trades=[]; in_trade=False; trade=None; monthly_pnl={}
    
    for i in range(200,n):
        row=df.iloc[i]; ts=row['timestamp']
        mk=f'{ts.year}-{ts.month:02d}'
        if monthly_pnl.get(mk,0.0)<=-7 and not in_trade: continue
        if not in_trade:
            if entry.iloc[i] and not pd.isna(ema.iloc[i-1]) and ema.iloc[i-1]>row['close']:
                sl=df.iloc[max(0,i-60):i][sm[max(0,i-60):i]]['low'].min()*0.998 if sm[max(0,i-60):i].sum()>0 else row['close']*0.95
                trade={'ei':i,'et':ts,'e1':row['close'],'ae':row['close'],'al':0.25,'sl':sl,'tp':ema.iloc[i-1],'pl_act':False,'hi':row['close'],'dca':False,'di':None}
                trade['pl']=row['close']+(trade['tp']-row['close'])*60/100
                in_trade=True
        else:
            if row['high']>trade['hi']: trade['hi']=row['high']
            if not trade['pl_act'] and row['high']>=trade['pl']: trade['pl_act']=True
            if not trade['dca']:
                s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sm[s2:i+1]]
                if len(ns)>0 and ns['low'].min()<trade['e1']:
                    trade['ae']=(trade['e1']*25+row['close']*75)/100; trade['al']=1.0; trade['dca']=True; trade['di']=i
                    trade['sl']=ns['low'].min()*0.998
                    trade['pl']=trade['ae']+(trade['tp']-trade['ae'])*60/100
                    if row['high']>=trade['pl']: trade['pl_act']=True
            st2=max(0,i-100); swt=df.iloc[st2:i+1][sm[st2:i+1]]
            if len(swt)>0:
                nsl=swt['low'].min()*0.998
                if nsl>trade['sl']: trade['sl']=nsl
            if trade['pl_act']:
                trail_sl=trade['hi']*(1-0.3/100)
                if trail_sl>trade['sl']: trade['sl']=trail_sl
            er=None; epx=None; hrs=(ts-trade['et']).total_seconds()/3600
            tp_h=row['high']>=trade['tp']
            sl_h=(row['high']>=trade['sl']) if trade['sl']>trade['ae'] else (row['low']<=trade['sl'])
            if tp_h: er,epx='TP',trade['tp']
            elif sl_h:
                if trade['pl_act']: er,epx='PL',max(trade['sl'],row['low'])
                elif trade['sl']<=trade['ae']: er,epx='SL',max(trade['sl'],row['low'])
                else: er,epx='SL_UP',min(trade['sl'],row['high'])
            elif hrs>=4: er,epx='TIME',row['close']
            if er:
                pnl=(epx-trade['ae'])/trade['ae']-0.002; eff=pnl*trade['al']
                monthly_pnl[mk]=monthly_pnl.get(mk,0.0)+eff*100
                capital*=(1+eff)
                trades.append({'ei':trade['ei'],'di':trade.get('di'),'xi':i,'pnl':pnl*100,'er':er,'dca':trade['dca'],'ae':trade['ae'],'epx':epx,'et':trade['et'],'xt':ts})
                in_trade=False; trade=None
    
    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return None, entry, ema
    return tdf, entry, ema

# ═══════════════ رسم 4 مخططات ═══════════════
fig, axes = plt.subplots(2, 2, figsize=(28, 18))
fig.patch.set_facecolor('#0a0a14')
axes = axes.flatten()

for idx, (name, lb, wf, ws, smin, use_vol, use_sma50) in enumerate(configs):
    ax = axes[idx]
    ax.set_facecolor('#0d1117')
    
    tdf, entry_sig, ema_vals = run_backtest(lb, wf, ws, smin, use_vol, use_sma50)
    
    # شموع
    for i in range(len(df)):
        row = df.iloc[i]
        c = '#00e676' if row['close'] >= row['open'] else '#ff1744'
        x = mdates.date2num(row['timestamp']); bw = 0.0002
        ax.plot([x, x], [row['low'], row['high']], color=c, linewidth=0.5)
        ax.add_patch(plt.Rectangle((x-bw, min(row['open'],row['close'])), bw*2,
                    abs(row['close']-row['open']), facecolor=c, edgecolor=c, linewidth=0.3))
    
    # EMA
    ax.plot(mdates.date2num(df['timestamp']), ema_vals, color='#ffa726', linewidth=1, alpha=0.8, label='EMA21')
    
    if tdf is not None and len(tdf) > 0:
        w=tdf[tdf['pnl']>0]; l=tdf[tdf['pnl']<=0]
        dca_trades = tdf[tdf['di'].notna()]
        
        # دخول
        e_idx = df[entry_sig].index
        ax.scatter(mdates.date2num(df.iloc[e_idx]['timestamp']), df.iloc[e_idx]['low']*0.997,
                   color='#00e5ff', s=60, marker='^', zorder=10, edgecolors='white', linewidths=0.5, label='Entry')
        
        # تعزيز
        if len(dca_trades) > 0:
            dca_dates = mdates.date2num(df.iloc[dca_trades['di'].astype(int).values]['timestamp'])
            dca_prices = [df.iloc[int(di)]['low']*0.996 for di in dca_trades['di']]
            ax.scatter(dca_dates, dca_prices, color='#ff9100', s=50, marker='D', zorder=10,
                       edgecolors='white', linewidths=0.5, label='DCA')
        
        # صفقات
        for _, t in tdf.iterrows():
            c = '#00e676' if t['pnl'] > 0 else '#ff1744'
            x1 = mdates.date2num(t['et']); x2 = mdates.date2num(t['xt'])
            ax.plot([x1, x2], [t['ae'], t['epx']], color=c, linewidth=2, alpha=0.8, zorder=8)
            ax.scatter([x1], [t['ae']], color='#00e5ff', s=40, zorder=11, edgecolors='white', linewidths=0.5)
            ax.scatter([x2], [t['epx']], color=c, s=40, zorder=11, marker='s', edgecolors='white', linewidths=0.5)
            # Label
            mid_x = (x1+x2)/2
            ax.annotate(f'{t["er"]} {t["pnl"]:+.1f}%', (mid_x, ax.get_ylim()[0]+0.0005),
                       fontsize=6, color=c, ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', edgecolor=c, alpha=0.8, linewidth=0.8))
        
        tp_c = len(tdf[tdf['er']=='TP']); pl_c = len(tdf[tdf['er']=='PL'])
        sl_c = len(tdf[tdf['er'].isin(['SL','SL_UP'])]); tm_c = len(tdf[tdf['er']=='TIME'])
        avg_w = w['pnl'].mean(); avg_l = l['pnl'].mean() if len(l)>0 else 0
        net = tdf['pnl'].sum()
        
        title = (f'{name} | {len(tdf)}T | 🟢{len(w)} 🔴{len(l)} | WR:{len(w)/len(tdf)*100:.0f}% | Net:{net:+.1f}%\n'
                f'AvgWin:+{avg_w:.2f}% AvgLoss:{avg_l:.2f}% | TP:{tp_c} PL:{pl_c} SL:{sl_c} ⏱:{tm_c} | DCA:{len(dca_trades)}')
    else:
        title = f'{name} | 0 صفقات — السوق تحت SMA50 اليومي'
    
    ax.set_title(title, color='white', fontsize=8.5, pad=8, fontfamily='monospace', fontweight='bold')
    ax.tick_params(colors='white', labelsize=7)
    ax.grid(alpha=0.08)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('USDT', color='white', fontsize=7)

# عنوان رئيسي
fig.suptitle(f'Whale Strategy — Last 10 Days | FET/USDT 15m | Price: ${price_start:.3f} -> ${price_end:.3f} ({price_chg:+.1f}%)',
             color='white', fontsize=14, fontweight='bold', y=0.98)

# مفتاح موحد
legend_elements = [
    mpatches.Patch(color='#00e676', alpha=0.7, label='🟢 ربح'),
    mpatches.Patch(color='#ff1744', alpha=0.7, label='🔴 خسارة'),
    plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='#00e5ff', markersize=10, label='Entry'),
    plt.Line2D([0],[0], marker='D', color='w', markerfacecolor='#ff9100', markersize=9, label='DCA'),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#888', markersize=9, label='خروج'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10,
           framealpha=0.9, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig('backtests/charts/10d_4configs.png', dpi=150, facecolor='#0a0a14', bbox_inches='tight')
print('✅ 4-config chart saved')
