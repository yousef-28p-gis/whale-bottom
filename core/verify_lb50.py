"""
تدقيق شامل لـ LB50 WMA3/10 STR>10 — فحص look-ahead والأخطاء
"""
import sys; sys.path.insert(0,'/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
CAP=1000.0; n=len(df)

# LB50 Config
lookback = 50; strength_min = 10; wma_fast = 3; wma_slow = 10
whale = whale_indicator(df, lookback)
spike = whale_spike(whale)
wma_f = whale_ma(whale, wma_fast)
wma_s = whale_ma(whale, wma_slow)
strn = whale_strength(whale, 50)

entry_signal = spike & (wma_f > wma_s) & (strn > strength_min)
# بدون حجم وبدون SMA50

ema = ema21(df); sell = sell_signal(df); sm = swing_lows(df,5)

print(f"📦 {len(df)} candles | 🚦 {entry_signal.sum()} signals\n")
print(f"⏳ تشغيل LB50 مع تدقيق كامل...")

# ── جمع كل الصفقات مع بيانات التدقيق ──
trades = []
in_trade = False; trade = None
errors = []

for i in range(500, n):
    row = df.iloc[i]; ts = row['timestamp']
    
    if not in_trade:
        if entry_signal.iloc[i]:
            ep = row['close']
            # ⚡ فحص look-ahead: TP من البار السابق فقط
            tp = ema.iloc[i-1] if i >= 1 and not pd.isna(ema.iloc[i-1]) else None
            if tp is None or tp <= ep:
                continue
            
            sw_s = max(0, i-60)
            sw_r = df.iloc[sw_s:i][sm[sw_s:i]]
            sl = sw_r['low'].min() * 0.998 if len(sw_r) > 0 else ep * 0.95
            
            # ⚡ فحص: SL محسوب من بيانات حتى i-1 فقط (لا look-ahead)
            if len(sw_r) > 0:
                latest_swing_idx = sw_r.index[-1]
                if latest_swing_idx > i:
                    errors.append(f"LOOK-AHEAD: SL used swing at idx {latest_swing_idx} >= entry idx {i}")
            
            pl_price = ep + (tp - ep) * 60 / 100
            trade = {
                'entry_idx': i, 'entry_time': ts,
                'entry1': ep, 'entry2': None, 'avg_entry': ep,
                'allocation': 0.25,
                'sl_price': sl, 'tp_price': tp,
                'pl_price': pl_price, 'pl_active': False,
                'highest_high': row['high'],
                'dca_done': False, 'dca_idx': None,
                'bars_in': 0,
            }
            in_trade = True
    
    else:
        trade['bars_in'] += 1
        
        if row['high'] > trade['highest_high']:
            trade['highest_high'] = row['high']
        
        # PL activation
        if not trade['pl_active'] and row['high'] >= trade['pl_price']:
            trade['pl_active'] = True
        
        # DCA
        if not trade['dca_done']:
            s2 = max(0, trade['entry_idx'] + 1)
            ns = df.iloc[s2:i+1][sm[s2:i+1]]
            if len(ns) > 0 and ns['low'].min() < trade['entry1']:
                trade['entry2'] = row['close']
                trade['avg_entry'] = (trade['entry1'] * 25 + trade['entry2'] * 75) / 100
                trade['allocation'] = 1.0
                trade['dca_done'] = True
                trade['dca_idx'] = i
                trade['sl_price'] = ns['low'].min() * 0.998
                trade['pl_price'] = trade['avg_entry'] + (trade['tp_price'] - trade['avg_entry']) * 60 / 100
                if row['high'] >= trade['pl_price']:
                    trade['pl_active'] = True
                
                # ⚡ فحص: SL بعد DCA من بيانات سابقة فقط
                latest_dca_swing = ns.index[-1]
                if latest_dca_swing > i:
                    errors.append(f"LOOK-AHEAD DCA: SL used swing at idx {latest_dca_swing} >= current idx {i}")
        
        # Trail SL
        st = max(0, i - 100)
        swt = df.iloc[st:i+1][sm[st:i+1]]
        if len(swt) > 0:
            nsl = swt['low'].min() * 0.998
            if nsl > trade['sl_price']:
                trade['sl_price'] = nsl
        
        # PL trail
        if trade['pl_active']:
            trail_sl = trade['highest_high'] * (1 - 0.3/100)
            if trail_sl > trade['sl_price']:
                trade['sl_price'] = trail_sl
        
        # ══════ Exit ══════
        er = None; epx = None
        hours_elapsed = (ts - trade['entry_time']).total_seconds() / 3600
        
        tp_hit = row['high'] >= trade['tp_price']
        sl_hit = (row['high'] >= trade['sl_price']) if trade['sl_price'] > trade['avg_entry'] else (row['low'] <= trade['sl_price'])
        
        if tp_hit:
            er = 'TP'; epx = trade['tp_price']
        elif i >= 2 and sell.iloc[i-1] >= 60:
            er = 'SELL'; epx = row['close']
        elif sl_hit:
            if trade['pl_active']:
                er = 'PL'; epx = trade['sl_price']
            else:
                er = 'SL_UP' if trade['sl_price'] > trade['avg_entry'] else 'SL'
                epx = (min(trade['sl_price'], row['high']) if trade['sl_price'] > trade['avg_entry'] else max(trade['sl_price'], row['low']))
        elif hours_elapsed >= 4:
            er = 'TIME'; epx = row['close']
        
        if er:
            # ⚡ تدقيق سعر الخروج
            if epx > row['high'] + 1e-6:
                errors.append(f"TRADE {len(trades)}: exit_price {epx:.6f} > candle high {row['high']:.6f}")
            if epx < row['low'] - 1e-6:
                errors.append(f"TRADE {len(trades)}: exit_price {epx:.6f} < candle low {row['low']:.6f}")
            
            # ⚡ تدقيق: TP وصل قبل SL بنفس الشمعة؟
            if er in ('SL', 'SL_UP') and tp_hit:
                # TP و SL لمسوا نفس الشمعة — اللي لمس أول هو الفائز
                # إذا open أقرب للـ SL → SL فازت
                dist_tp = abs(row['open'] - trade['tp_price'])
                dist_sl = abs(row['open'] - trade['sl_price'])
                if dist_tp < dist_sl:
                    errors.append(f"TRADE {len(trades)}: {er} but TP was closer at open! (tp_dist={dist_tp:.6f}, sl_dist={dist_sl:.6f})")
            
            # ⚡ تدقيق P&L
            pnl_pct = (epx - trade['avg_entry']) / trade['avg_entry'] - 0.002
            eff_pnl = pnl_pct * trade['allocation']
            
            # ⚡ تدقيق: SL فوق الدخول = هدف وليس وقف
            if er in ('SL',) and trade['sl_price'] > trade['avg_entry']:
                errors.append(f"TRADE {len(trades)}: er=SL but sl_price ({trade['sl_price']:.6f}) > avg_entry ({trade['avg_entry']:.6f}) — should be SL_UP!")
            
            if er in ('SL_UP',) and trade['sl_price'] <= trade['avg_entry']:
                errors.append(f"TRADE {len(trades)}: er=SL_UP but sl_price ({trade['sl_price']:.6f}) <= avg_entry ({trade['avg_entry']:.6f})")
            
            trades.append({
                'entry_idx': trade['entry_idx'],
                'exit_idx': i,
                'entry_time': trade['entry_time'],
                'exit_time': ts,
                'entry1': trade['entry1'],
                'entry2': trade['entry2'],
                'avg_entry': trade['avg_entry'],
                'exit_price': epx,
                'pnl_pct': pnl_pct * 100,
                'eff_pnl_pct': eff_pnl * 100,
                'exit_reason': er,
                'tp_price': trade['tp_price'],
                'sl_price': trade['sl_price'],
                'sl_above_entry': trade['sl_price'] > trade['avg_entry'],
                'dca_done': trade['dca_done'],
                'pl_active': trade['pl_active'],
                'allocation': trade['allocation'] * 100,
                'bars_in': trade['bars_in'],
                'highest_high': trade['highest_high'],
            })
            
            in_trade = False
            trade = None

tdf = pd.DataFrame(trades)

# ═══════════════════════════════════════════════
# تدقيق إضافي
# ═══════════════════════════════════════════════

# تدقيق: Compounding يدوي
cap_manual = CAP
peak_manual = CAP
max_dd_manual = 0.0
for _, t in tdf.iterrows():
    cap_manual *= (1 + t['eff_pnl_pct'] / 100)
    if cap_manual > peak_manual:
        peak_manual = cap_manual
    dd = (cap_manual - peak_manual) / peak_manual * 100
    if dd < max_dd_manual:
        max_dd_manual = dd

# تدقيق: Linear P&L
linear_eff = tdf['eff_pnl_pct'].sum()
linear_cap_manual = CAP * (1 + linear_eff / 100)

# ═══════════════════════════════════════════════
# عرض النتائج
# ═══════════════════════════════════════════════
wins = tdf[tdf['pnl_pct'] > 0]
losses = tdf[tdf['pnl_pct'] <= 0]
wr = len(wins) / len(tdf) * 100
years = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days / 365
yearly = linear_eff / years

print(f"\n{'='*70}")
print(f"🔍 تدقيق شامل — LB50 WMA3/10 STR>10")
print(f"{'='*70}")
print(f"صفقات: {len(tdf)} | WR: {wr:.1f}%")
print(f"AvgWin: {wins['pnl_pct'].mean():.2f}% | AvgLoss: {losses['pnl_pct'].mean():.2f}%")
print(f"رأس مال (مركب): ${cap_manual:.0f}")
print(f"رأس مال (خطي): ${linear_cap_manual:.0f} | إجمالي: {linear_eff:.1f}%")
print(f"DD (مركب): {max_dd_manual:.1f}% | سنوي: {yearly:.1f}%")
print(f"DCA: {tdf['dca_done'].sum()} | PL: {tdf['pl_active'].sum()}")
print(f"مخارج: {tdf['exit_reason'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════
# فحوصات إضافية
# ═══════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"🔬 فحوصات متقدمة")
print(f"{'='*70}")

# 1. فحص: أي صفقة TP ضربت لكن SL كان أقرب؟
tp_sl_same = 0
for _, t in tdf.iterrows():
    if t['exit_reason'] in ('SL', 'SL_UP'):
        ei, xi = int(t['entry_idx']), int(t['exit_idx'])
        exit_row = df.iloc[xi]
        if exit_row['high'] >= t['tp_price']:
            tp_sl_same += 1

print(f"⚠️ صفقات SL مع TP بنفس الشمعة: {tp_sl_same}")
if tp_sl_same > 0:
    print(f"   (هذا طبيعي إذا الـ SL كان أقرب لسعر الافتتاح)")

# 2. فحص: أي صفقة خرجت قبل 4 ساعات مع إنها فرضاً تنتظر؟
early_time = tdf[(tdf['exit_reason'] == 'TIME') & (tdf['bars_in'] < 16)]
print(f"⏱ خرج مبكر (<4hr): {len(early_time)}")

# 3. فحص: هل في صفقات SL_UP = هدف وليس وقف فعلاً؟
sl_up_wins = tdf[(tdf['exit_reason'] == 'SL_UP') & (tdf['pnl_pct'] > 0)]
print(f"📈 SL_UP ربحانة: {len(sl_up_wins)}/{len(tdf[tdf['exit_reason']=='SL_UP'])}")

# 4. check NaN
nan_trades = tdf[tdf['avg_entry'].isna() | tdf['exit_price'].isna()]
print(f"🔢 صفقات NaN: {len(nan_trades)}")

# 5. فحص توزيع سنوي
print(f"\n📅 سنوي:")
for yr, grp in tdf.groupby(tdf['entry_time'].dt.year):
    y_eff = grp['eff_pnl_pct'].sum()
    y_cap = CAP * (1 + y_eff/100)
    y_wr = (grp['pnl_pct'] > 0).sum() / len(grp) * 100
    print(f"   {yr}: {len(grp)}T | WR={y_wr:.0f}% | +{y_eff:.0f}% | ${y_cap:.0f}")

# ═══════════════════════════════════════════════
# فحص ٣ صفقات عشوائية شمعة بشمعة
# ═══════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"🔍 فحص ٣ صفقات عشوائية — شمعة بشمعة")
print(f"{'='*70}")

np.random.seed(42)
sample = tdf.sample(min(3, len(tdf)))

for idx, t in sample.iterrows():
    ei, xi = int(t['entry_idx']), int(t['exit_idx'])
    print(f"\n📌 صفقة #{idx}: {t['entry_time']} → {t['exit_time']}")
    print(f"   دخول: {t['entry1']:.6f} | DCA: {t['entry2'] if not pd.isna(t['entry2']) else 'لا'}")
    print(f"   متوسط: {t['avg_entry']:.6f} | تخصيص: {t['allocation']:.0f}%")
    print(f"   TP: {t['tp_price']:.6f} | SL: {t['sl_price']:.6f} | PL: {t['pl_active']}")
    print(f"   خروج: {t['exit_reason']} @ {t['exit_price']:.6f} | PnL: {t['pnl_pct']:.2f}%")
    
    # فحص ٣ شمعات بعد الدخول
    check_range = df.iloc[ei:min(ei+5, xi+1)]
    print(f"   شموع بعد الدخول:")
    for j, cr in check_range.iterrows():
        mark = ''
        if cr['high'] >= t['tp_price']: mark += ' [TP!]'
        if t['sl_above_entry']:
            if cr['high'] >= t['sl_price']: mark += ' [SL_UP!]'
        else:
            if cr['low'] <= t['sl_price']: mark += ' [SL!]'
        print(f"     {cr['timestamp']}: O={cr['open']:.6f} H={cr['high']:.6f} L={cr['low']:.6f} C={cr['close']:.6f}{mark}")

# ═══════════════════════════════════════════════
# النتيجة النهائية
# ═══════════════════════════════════════════════
print(f"\n{'='*70}")
if len(errors) == 0:
    print(f"✅ لا توجد أخطاء! كل {len(tdf)} صفقة صحيحة.")
else:
    print(f"❌ {len(errors)} أخطاء:")
    for e in errors[:10]:
        print(f"   {e}")

# فحص ختامي: هل الـ compounding صحيح؟
recap = CAP
for _, t in tdf.iterrows():
    recap *= (1 + t['eff_pnl_pct'] / 100)
diff = abs(recap - cap_manual)
print(f"✅ تدقيق التراكم: الفرق = {diff:.6f} (يجب أن يكون 0)")

print(f"\n✅ تم التدقيق الشامل")
