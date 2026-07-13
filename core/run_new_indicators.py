"""
مؤشرات جديدة على الحوت + فلاتر لم تستخدم من قبل
ADX, SuperTrend, RSI على الحوت, MACD على الحوت, BB على الحوت, Choppiness
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd
import numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts'])
df = df.sort_values('timestamp').reset_index(drop=True)

CAP=1000.0; n=len(df)
whale = whale_indicator(df, 200)
spike = whale_spike(whale)
wma20 = whale_ma(whale,20); wma50 = whale_ma(whale,50)
strength = whale_strength(whale,50)
vol_ok = volume_filter(df)
sma50 = sma50_daily(df)

# ═══════════════════════════════════════════════
# مؤشرات جديدة على الحوت نفسه
# ═══════════════════════════════════════════════

# RSI على الحوت
w_delta = whale.diff()
w_gain = w_delta.clip(lower=0); w_loss = -w_delta.clip(upper=0)
w_rsi = 100 - (100/(1+(w_gain.ewm(alpha=1/14,adjust=False).mean()/
    w_loss.ewm(alpha=1/14,adjust=False).mean().replace(0,1e-10))))

# MACD على الحوت
w_ema12 = whale.ewm(span=12,adjust=False).mean()
w_ema26 = whale.ewm(span=26,adjust=False).mean()
w_macd = w_ema12 - w_ema26
w_macd_sig = w_macd.ewm(span=9,adjust=False).mean()
w_macd_hist = w_macd - w_macd_sig

# BB على الحوت
w_bb_mid = whale.rolling(20).mean()
w_bb_std = whale.rolling(20).std()
w_bb_low = w_bb_mid - 2*w_bb_std

# ═══════════════════════════════════════════════
# مؤشرات جديدة على السعر
# ═══════════════════════════════════════════════

# ADX
def calc_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up = high - high.shift(); dn = low.shift() - low
    plus_dm = np.where((up>dn) & (up>0), up, 0)
    minus_dm = np.where((dn>up) & (dn>0), dn, 0)
    plus_di = pd.Series(plus_dm).ewm(span=period,adjust=False).mean()/atr*100
    minus_di = pd.Series(minus_dm).ewm(span=period,adjust=False).mean()/atr*100
    dx = abs(plus_di-minus_di)/(plus_di+minus_di).replace(0,1)*100
    adx = dx.ewm(span=period,adjust=False).mean()
    return adx.fillna(0)

adx = calc_adx(df)

# SuperTrend
def calc_supertrend(df, period=10, mult=3):
    atr_st = atr(df, period)
    hl2 = (df['high']+df['low'])/2
    upper = hl2 + mult*atr_st; lower = hl2 - mult*atr_st
    st = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]: trend.iloc[i]=1
        elif df['close'].iloc[i] < lower.iloc[i-1]: trend.iloc[i]=-1
        else: trend.iloc[i]=trend.iloc[i-1]
        if trend.iloc[i]==1 and lower.iloc[i]<lower.iloc[i-1]: lower.iloc[i]=lower.iloc[i-1]
        if trend.iloc[i]==-1 and upper.iloc[i]>upper.iloc[i-1]: upper.iloc[i]=upper.iloc[i-1]
        st.iloc[i] = lower.iloc[i] if trend.iloc[i]==1 else upper.iloc[i]
    return trend  # 1=uptrend, -1=downtrend

st_trend = calc_supertrend(df)

# Choppiness Index
def calc_choppiness(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    hh = high.rolling(period).max(); ll = low.rolling(period).min()
    chop = 100 * np.log10(atr_sum/(hh-ll).replace(0,1)) / np.log10(period)
    return chop.fillna(50)

chop = calc_choppiness(df)

# ═══════════════════════════════════════════════
print(f"📦 {len(df)} candles | 🚦 signals: {spike.sum()}\n")

# إشارة أساسية
base = spike & (wma20>wma50) & (strength>50) & vol_ok & (df['close']>sma50)

# ═══════════════════════════════════════════════
# Run function
# ═══════════════════════════════════════════════
def run_dca(entry_signal, label=""):
    ema = ema21(df); sell = sell_signal(df); sm = swing_lows(df,5)
    capital=CAP; peak=CAP; max_dd=0.0; monthly_pnl={}; trades=[]
    in_trade=False; trade=None
    for i in range(500,n):
        row=df.iloc[i]; ts=row['timestamp']
        mk=f"{ts.year}-{ts.month:02d}"
        if monthly_pnl.get(mk,0.0)<=-7 and not in_trade: continue
        if not in_trade:
            if entry_signal.iloc[i]:
                ep=row['close']
                if i<1 or pd.isna(ema.iloc[i-1]): continue
                tp=ema.iloc[i-1]
                if tp<=ep: continue
                sw_s=max(0,i-60); sw_r=df.iloc[sw_s:i][sm[sw_s:i]]
                sl=sw_r['low'].min()*0.998 if len(sw_r)>0 else ep*0.95
                trade={'ei':i,'et':ts,'e1':ep,'e2':None,'ae':ep,'al':0.5,
                       'sl':sl,'tp':tp,'dca':False}
                in_trade=True
        else:
            if not trade['dca']:
                s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sm[s2:i+1]]
                if len(ns)>0 and ns['low'].min()<trade['e1']:
                    trade['e2']=row['close']; trade['ae']=(trade['e1']+trade['e2'])/2
                    trade['al']=1.0; trade['dca']=True
                    trade['sl']=ns['low'].min()*0.998
            st2=max(0,i-100); swt=df.iloc[st2:i+1][sm[st2:i+1]]
            if len(swt)>0:
                nsl=swt['low'].min()*0.998
                if nsl>trade['sl']: trade['sl']=nsl
            er=None; epx=None; hrs=(ts-trade['et']).total_seconds()/3600
            tp_h=row['high']>=trade['tp']
            sl_h=(row['high']>=trade['sl']) if trade['sl']>trade['ae'] else (row['low']<=trade['sl'])
            if tp_h: er,epx='TP',trade['tp']
            elif i>=2 and sell.iloc[i-1]>=60: er,epx='SELL',row['close']
            elif sl_h:
                er='SL_UP' if trade['sl']>trade['ae'] else 'SL'
                epx=(min(trade['sl'],row['high']) if trade['sl']>trade['ae'] else max(trade['sl'],row['low']))
            elif hrs>=4: er,epx='TIME',row['close']
            if er:
                pnl=(epx-trade['ae'])/trade['ae']-0.002; eff=pnl*trade['al']
                monthly_pnl[mk]=monthly_pnl.get(mk,0.0)+eff*100
                capital*=(1+eff)
                if capital>peak: peak=capital
                dd=(capital-peak)/peak
                if dd<max_dd: max_dd=dd
                trades.append({'pnl':pnl*100,'er':er,'dca':trade['dca']})
                in_trade=False; trade=None
    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return 0,0,0,0
    wins=tdf[tdf['pnl']>0]
    return len(tdf), len(wins)/len(tdf)*100, capital, max_dd*100

# ═══════════════════════════════════════════════
configs = [
    ('1️⃣ الأساس (بدون فلتر جديد)', base),
    ('2️⃣ + ADX > 20', base & (adx > 20)),
    ('3️⃣ + ADX > 25', base & (adx > 25)),
    ('4️⃣ + SuperTrend صاعد', base & (st_trend == 1)),
    ('5️⃣ + Choppiness < 38.2 (مترند)', base & (chop < 38.2)),
    ('6️⃣ + RSI حوت < 30 (متطرف)', base & (w_rsi < 30)),
    ('7️⃣ + RSI حوت < 20', base & (w_rsi < 20)),
    ('8️⃣ + MACD حوت > 0 (تسارع)', base & (w_macd > 0)),
    ('9️⃣ + BB حوت: عند الحد السفلي', base & (whale < w_bb_low)),
    ('🔟 + ADX>20 + RSI حوت<30', base & (adx>25) & (w_rsi<30)),
]

print(f"{'='*90}")
print(f"🏆 مؤشرات جديدة × حوت | DCA + 4hr | FET/USDT 15m")
print(f"{'='*90}")
print(f"{'الفلتر':<35} {'إشارات':>7} {'صفقات':>6} {'WR%':>6} {'المحفظة':>10} {'DD%':>7}")
print(f"{'-'*90}")

for name, entry in configs:
    sigs = entry.sum()
    t, wr, cap, dd = run_dca(entry, name)
    if t==0:
        print(f"{name:<35} {sigs:>7} {'—':>6} {'—':>6} {'—':>10} {'—':>7}")
        continue
    star = ' ⬅' if 'الأساس' in name else ''
    print(f"{name:<35} {sigs:>7} {t:>6} {wr:>5.1f}% ${cap:>9.0f} {dd:>6.1f}%{star}")

print(f"\n✅ تم")
