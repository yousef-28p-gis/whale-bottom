"""رسم محسن مع مفتاح خريطة وتفاصيل أوضح"""
import sys; sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)

cut = df['timestamp'].max() - pd.Timedelta(days=7)
df = df[df['timestamp'] >= cut].reset_index(drop=True)
CAP=1000.0; n=len(df)

lb=50; wf=3; ws=10; smin=10
whale = whale_indicator(df, lb)
entry = (whale_spike(whale) & (whale_ma(whale, wf) > whale_ma(whale, ws)) &
         (whale_strength(whale, 50) > smin))
ema=ema21(df); sm=swing_lows(df,5)
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

tdf=pd.DataFrame(trades); w=tdf[tdf['pnl']>0]; l=tdf[tdf['pnl']<=0]
print(f'صفقات: {len(tdf)} | 🟢 {len(w)} | 🔴 {len(l)} | WR: {len(w)/len(tdf)*100:.0f}%')

# ═══════════════ رسم ═══════════════
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(22, 10))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# شموع
for i in range(len(df)):
    row = df.iloc[i]
    c = '#00e676' if row['close'] >= row['open'] else '#ff1744'
    x = mdates.date2num(row['timestamp']); bw = 0.00025
    ax.plot([x, x], [row['low'], row['high']], color=c, linewidth=0.7)
    ax.add_patch(plt.Rectangle((x-bw, min(row['open'],row['close'])), bw*2,
                abs(row['close']-row['open']), facecolor=c, edgecolor=c, linewidth=0.4))

# EMA21
ax.plot(mdates.date2num(df['timestamp']), ema, color='#ffa726', linewidth=1.3, alpha=0.85, label='EMA21')

# إشارات الدخول 🟢
e_idx = df[entry].index
entry_dates = mdates.date2num(df.iloc[e_idx]['timestamp'])
entry_prices = df.iloc[e_idx]['low'] * 0.998
ax.scatter(entry_dates, entry_prices, color='#00e5ff', s=150, marker='^', zorder=10,
           edgecolors='white', linewidths=1, label='🐋 دخول (Entry)')

# إشارات التعزيز 🟠
dca_trades = tdf[tdf['di'].notna()]
if len(dca_trades) > 0:
    dca_dates = mdates.date2num(df.iloc[dca_trades['di'].astype(int).values]['timestamp'])
    dca_prices = [df.iloc[int(di)]['low'] * 0.997 for di in dca_trades['di']]
    ax.scatter(dca_dates, dca_prices, color='#ff9100', s=130, marker='D', zorder=10,
               edgecolors='white', linewidths=1, label='⬇ تعزيز (DCA)')

# صفقات وخطوط
for _, t in tdf.iterrows():
    is_win = t['pnl'] > 0
    c = '#00e676' if is_win else '#ff1744'
    x1 = mdates.date2num(t['et']); x2 = mdates.date2num(t['xt'])
    # خط
    ax.plot([x1, x2], [t['ae'], t['epx']], color=c, linewidth=2.8, alpha=0.8, zorder=8)
    # نقطة دخول
    ax.scatter([x1], [t['ae']], color='#00e5ff', s=80, zorder=11, edgecolors='white', linewidths=0.8)
    # نقطة خروج
    ax.scatter([x2], [t['epx']], color=c, s=80, zorder=11, marker='s', edgecolors='white', linewidths=0.8)
    
    # نسبة أسفل الشمعة
    mid_x = (x1+x2)/2
    label = f'{t["er"]}  {t["pnl"]:+.1f}%'
    y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * (0.04 + 0.025 * (t.name % 5))
    ax.annotate(label, (mid_x, ax.get_ylim()[0] + 0.002),
                fontsize=7.5, color=c, ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', edgecolor=c, alpha=0.85, linewidth=1))

# ═══════════════ مفتاح الخريطة ═══════════════
legend_elements = [
    mpatches.Patch(color='#00e676', alpha=0.7, label='🟢 ربح (Win)'),
    mpatches.Patch(color='#ff1744', alpha=0.7, label='🔴 خسارة (Loss)'),
    plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='#00e5ff', markersize=12, label='🐋 دخول'),
    plt.Line2D([0],[0], marker='D', color='w', markerfacecolor='#ff9100', markersize=10, label='⬇ تعزيز'),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#aaaaaa', markersize=10, label='خروج'),
]
leg = ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
                title='🔑 مفتاح الخريطة', title_fontsize=11, framealpha=0.9,
                facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
leg.get_title().set_color('white')

# ═══════════════ عنوان مفصل ═══════════════
tp_count = len(tdf[tdf['er']=='TP'])
pl_count = len(tdf[tdf['er']=='PL'])
sl_count = len(tdf[tdf['er'].isin(['SL','SL_UP'])])
time_count = len(tdf[tdf['er']=='TIME'])
dca_count = len(dca_trades)
avg_win = w['pnl'].mean() if len(w)>0 else 0
avg_loss = l['pnl'].mean() if len(l)>0 else 0
net_pnl = tdf['pnl'].sum()

title = (
    f'🐋 استراتيجية الحوت — آخر ٧ أيام | FET/USDT 15m\n'
    f'📊 {len(tdf)} صفقة | 🟢 {len(w)} رابحة | 🔴 {len(l)} خاسرة | 📈 WR {len(w)/len(tdf)*100:.0f}%\n'
    f'💰 صافي: {net_pnl:+.1f}% | 🟢 متوسط ربح: +{avg_win:.2f}% | 🔴 متوسط خسارة: {avg_loss:.2f}% | R:R {abs(avg_win/avg_loss):.1f}x\n'
    f'🏷️ خروج: TP={tp_count} | PL={pl_count} | SL={sl_count} | ⏱️={time_count} | تعزيزات={dca_count}'
)
ax.set_title(title, color='white', fontsize=11, pad=12, fontfamily='monospace',
             loc='center', fontweight='bold')

ax.set_ylabel('السعر (USDT)', color='white', fontsize=11)
ax.tick_params(colors='white', labelsize=9)
ax.grid(alpha=0.1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
fig.autofmt_xdate()

fig.tight_layout()
fig.savefig('backtests/charts/last_week.png', dpi=150, facecolor='#0d1117', bbox_inches='tight')
print('✅ تم')
