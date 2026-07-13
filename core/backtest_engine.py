"""
محرك الباك تست — بسيط، شفاف، بدون مفاجآت.
"""
import pandas as pd
import numpy as np


def run_backtest(df: pd.DataFrame,
                 entry_signal: pd.Series,
                 tp_series: pd.Series,    # EMA21
                 atr_series: pd.Series,
                 sell_series: pd.Series,
                 swing_mask: np.ndarray,
                 sma50_series: pd.Series,
                 tp_mode: str = 'ema21',   # 'ema21' or '3atr'
                 max_hours: int = 48,
                 monthly_limit: float = 0.07,
                 fee: float = 0.001) -> dict:
    """
    باك تست بسيط مع منع look-ahead في الأهداف.
    
    القواعد:
    - الدخول: close[i] إذا entry_signal[i] = True
    - الهدف TP: من البار السابق فقط (shift(1))
    - الوقف SL: من قيعان سوينج حتى البار i-1
    - SL فوق الدخول = هدف (high >= SL)، SL تحت الدخول = وقف (low <= SL)
    """
    
    n = len(df)
    capital = 1000.0
    equity_peak = 1000.0
    max_dd = 0.0
    monthly_pnl = {}
    trades = []
    
    in_trade = False
    trade = None
    
    for i in range(500, n):
        row = df.iloc[i]
        ts = row['timestamp']
        mk = f"{ts.year}-{ts.month:02d}"
        
        # حد خسارة شهري
        ml = monthly_pnl.get(mk, 0.0)
        if ml <= -monthly_limit and not in_trade:
            continue
        
        if not in_trade:
            if entry_signal.iloc[i]:
                ep = row['close']
                
                # === TP: من البار السابق فقط (لا look-ahead) ===
                if tp_mode == 'ema21':
                    if i < 1 or pd.isna(tp_series.iloc[i-1]):
                        continue
                    tp = tp_series.iloc[i-1]  # EMA21 البار السابق
                    if tp <= ep:
                        continue
                else:
                    if i < 1 or pd.isna(atr_series.iloc[i-1]):
                        continue
                    tp = ep + 3 * atr_series.iloc[i-1]  # ATR البار السابق
                
                # === SL: من قيعان سوينج حتى البار i-1 ===
                sw_start = max(0, i - 60)
                sw_recent = df.iloc[sw_start:i][swing_mask[sw_start:i]]
                if len(sw_recent) > 0:
                    sl = sw_recent['low'].min() * 0.998
                else:
                    sl = ep * 0.95
                
                trade = {
                    'entry_idx': i,
                    'entry_time': ts,
                    'entry_price': ep,
                    'sl_price': sl,
                    'tp_price': tp,
                    'highest_close': ep,
                }
                in_trade = True
        
        else:
            # تحديث أعلى سعر
            if row['close'] > trade['highest_close']:
                trade['highest_close'] = row['close']
            
            # === تحديث SL المتحرك ===
            # نبحث عن قيعان سوينج جديدة (حتى البار الحالي)
            sw_start = max(0, i - 100)
            sw_recent = df.iloc[sw_start:i+1][swing_mask[sw_start:i+1]]
            if len(sw_recent) > 0:
                new_sl = sw_recent['low'].min() * 0.998
                if new_sl > trade['sl_price']:
                    trade['sl_price'] = new_sl
            
            # === فحص الخروج ===
            exit_reason = None
            exit_price = None
            exit_row = row  # شمعة الخروج
            
            # 1. TP: السعر لمس الهدف
            tp_hit = row['high'] >= trade['tp_price']
            
            # 2. SL: حسب الاتجاه
            if trade['sl_price'] > trade['entry_price']:
                sl_hit = row['high'] >= trade['sl_price']
            else:
                sl_hit = row['low'] <= trade['sl_price']
            
            # لو TP و SL اتلمسوا في نفس الشمعة — اللي لمس أول هو الفائز
            if tp_hit and sl_hit:
                # السعر فتح، تحرك للـ SL أول ولا للـ TP أول؟
                # إذا open أقرب لـ SL — معناها لمس SL أول
                if trade['sl_price'] > trade['entry_price']:
                    # SL فوق = هدف، TP فوق
                    dist_tp = abs(row['open'] - trade['tp_price'])
                    dist_sl = abs(row['open'] - trade['sl_price'])
                else:
                    # SL تحت، TP فوق
                    dist_tp = trade['tp_price'] - row['low']  # مسافة تقريبية
                    dist_sl = row['high'] - trade['sl_price']
                
                if dist_sl < dist_tp:
                    sl_hit = True
                    tp_hit = False
                else:
                    tp_hit = True
                    sl_hit = False
            
            if tp_hit:
                exit_reason = 'TP'
                exit_price = trade['tp_price']
            
            # SELL: إشارة بيع قوية (فقط إذا TP ما لمس)
            elif i >= 2 and sell_series.iloc[i-1] >= 60:
                exit_reason = 'SELL'
                exit_price = row['close']
            
            # SL
            elif sl_hit:
                if trade['sl_price'] > trade['entry_price']:
                    exit_reason = 'SL_UP'
                    exit_price = min(trade['sl_price'], row['high'])
                else:
                    exit_reason = 'SL'
                    exit_price = max(trade['sl_price'], row['low'])
            
            # 4. TIME: حد زمني
            if not exit_reason:
                hours_elapsed = (ts - trade['entry_time']).total_seconds() / 3600
                if hours_elapsed >= max_hours:
                    exit_reason = 'TIME'
                    exit_price = row['close']
            
            if exit_reason:
                # حساب الربح
                pnl_pct = (exit_price - trade['entry_price']) / trade['entry_price'] - 2 * fee
                
                # تحديث شهري
                monthly_pnl[mk] = monthly_pnl.get(mk, 0.0) + pnl_pct
                
                # تراكم
                capital *= (1 + pnl_pct)
                
                # DD
                if capital > equity_peak:
                    equity_peak = capital
                dd = (capital - equity_peak) / equity_peak
                if dd < max_dd:
                    max_dd = dd
                
                trades.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': ts,
                    'entry_idx': trade['entry_idx'],
                    'exit_idx': i,
                    'entry_price': trade['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct * 100,
                    'exit_reason': exit_reason,
                    'tp_price': trade['tp_price'],
                    'sl_price': trade['sl_price'],
                    'sl_above_entry': trade['sl_price'] > trade['entry_price'],
                    'highest_close': trade['highest_close'],
                    'duration_m': (ts - trade['entry_time']).total_seconds() / 60,
                    'year': ts.year,
                })
                
                in_trade = False
                trade = None
    
    # === تجميع النتائج ===
    tdf = pd.DataFrame(trades)
    
    if len(tdf) == 0:
        return {'trades': 0, 'error': 'No trades'}
    
    wins = tdf[tdf['pnl_pct'] > 0]
    losses = tdf[tdf['pnl_pct'] <= 0]
    wr = len(wins) / len(tdf) * 100
    
    rets = tdf['pnl_pct'].values / 100
    sharpe = rets.mean() / rets.std() * np.sqrt(len(rets)) if rets.std() > 0 else 0
    
    return {
        'trades': len(tdf),
        'wins': len(wins),
        'losses': len(losses),
        'wr': wr,
        'capital': capital,
        'return_pct': (capital / 1000 - 1) * 100,
        'dd': max_dd * 100,
        'sharpe': sharpe,
        'avg_win': wins['pnl_pct'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl_pct'].mean() if len(losses) > 0 else 0,
        'max_win': tdf['pnl_pct'].max(),
        'max_loss': tdf['pnl_pct'].min(),
        'avg_dur': tdf['duration_m'].mean(),
        'tdf': tdf,
    }
