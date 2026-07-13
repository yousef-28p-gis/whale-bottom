"""
نظام التدقيق — يفحص كل صفقة ويتأكد من صحتها.
"""
import pandas as pd
import numpy as np


def verify_trades(df: pd.DataFrame, result: dict) -> bool:
    """
    تدقيق جميع الصفقات.
    Returns: True إذا كل الصفقات صحيحة، False إذا فيه خطأ.
    """
    if 'tdf' not in result:
        print("❌ لا توجد صفقات للتدقيق")
        return False
    
    tdf = result['tdf']
    errors = []
    warnings = []
    
    for idx, t in tdf.iterrows():
        ei = int(t['entry_idx'])
        xi = int(t['exit_idx'])
        er = t['exit_reason']
        
        # === 1. تحقق من شمعة الدخول ===
        entry_row = df.iloc[ei]
        if pd.isna(entry_row['close']):
            errors.append(f"صفقة #{idx}: شمعة دخول #{ei} فيها NaN")
            continue
        
        # === 2. تحقق من شمعة الخروج ===
        exit_row = df.iloc[xi]
        
        # === 3. تحقق من سعر الخروج ===
        if t['exit_price'] > exit_row['high'] + 0.0001:
            errors.append(f"صفقة #{idx}: سعر خروج {t['exit_price']:.6f} > High الشمعة {exit_row['high']:.6f}")
        if t['exit_price'] < exit_row['low'] - 0.0001:
            errors.append(f"صفقة #{idx}: سعر خروج {t['exit_price']:.6f} < Low الشمعة {exit_row['low']:.6f}")
        
        # === 4. تحقق من سبب الخروج ===
        if er == 'TP':
            if exit_row['high'] < t['tp_price'] - 0.0001:
                errors.append(f"صفقة #{idx}: TP — High={exit_row['high']:.6f} < TP={t['tp_price']:.6f}")
        elif er == 'SL_UP':
            if exit_row['high'] < t['sl_price'] - 0.0001:
                errors.append(f"صفقة #{idx}: SL_UP — High={exit_row['high']:.6f} < SL={t['sl_price']:.6f}")
        elif er == 'SL':
            if exit_row['low'] > t['sl_price'] + 0.0001:
                errors.append(f"صفقة #{idx}: SL — Low={exit_row['low']:.6f} > SL={t['sl_price']:.6f}")
        elif er == 'TIME':
            duration_h = (t['exit_time'] - t['entry_time']).total_seconds() / 3600
            if duration_h < 47.9:
                errors.append(f"صفقة #{idx}: TIME — المدة {duration_h:.1f}h < 48h")
        
        # === 5. تحقق: ما لمس TP قبل SL؟ ===
        if er in ('SL', 'SL_UP'):
            between = df.iloc[ei+1:xi]
            if len(between) > 0 and (between['high'].max() >= t['tp_price']):
                errors.append(f"صفقة #{idx}: {er} لكن السعر لمس TP={t['tp_price']:.6f} قبلها! (High={between['high'].max():.6f})")
        
        # === 6. تحقق: ما لمس SL قبل TP؟ ===
        if er == 'TP':
            between = df.iloc[ei+1:xi]
            if len(between) > 0:
                if t['sl_above_entry']:
                    if between['high'].max() >= t['sl_price']:
                        errors.append(f"صفقة #{idx}: TP لكن لمس SL_UP={t['sl_price']:.6f} قبلها (High={between['high'].max():.6f})")
                else:
                    if between['low'].min() <= t['sl_price']:
                        errors.append(f"صفقة #{idx}: TP لكن لمس SL={t['sl_price']:.6f} قبلها! (Low={between['low'].min():.6f})")
        
        # === 7. تحقق الحساب ===
        expected_pnl = (t['exit_price'] - t['entry_price']) / t['entry_price'] - 0.002
        if abs(expected_pnl - t['pnl_pct'] / 100) > 0.001:
            errors.append(f"صفقة #{idx}: PnL محسوب={t['pnl_pct']:.2f}% ≠ متوقع={expected_pnl*100:.2f}%")
    
    # === عرض النتائج ===
    print(f"\n{'='*60}")
    print(f"🔍 تدقيق {len(tdf)} صفقة")
    print(f"{'='*60}")
    
    if errors:
        print(f"\n❌ أخطاء: {len(errors)}")
        for e in errors[:10]:
            print(f"   {e}")
        if len(errors) > 10:
            print(f"   ... و {len(errors)-10} خطأ إضافي")
    
    if warnings:
        print(f"\n⚠️ تحذيرات: {len(warnings)}")
        for w in warnings[:5]:
            print(f"   {w}")
    
    if not errors:
        print(f"\n✅ كل الصفقات صحيحة — {len(tdf)} صفقة بدون أخطاء!")
        return True
    else:
        print(f"\n❌ فشل التدقيق — {len(errors)} خطأ")
        return False
