"""
المؤشرات — v2 مبسطة.
المؤشرات تحسب عادي (تشمل البار الحالي).
الباك تست هو المسؤول عن منع look-ahead في أهداف الخروج.
"""
import pandas as pd
import numpy as np


def whale_indicator(df: pd.DataFrame, lookback: int = 200) -> pd.Series:
    """مؤشر الحيتان عند القيعان."""
    lowest_n = df['low'].rolling(lookback).min()
    at_low = df['low'] <= lowest_n
    low_change = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
    smooth_change = low_change.ewm(span=3, adjust=False).mean()
    highest_change = smooth_change.rolling(lookback).max()
    strength = np.where(at_low, (smooth_change + highest_change * 2) / 3, 0)
    return pd.Series(strength, index=df.index).ewm(span=3, adjust=False).mean().fillna(0)


def whale_ma(whale: pd.Series, period: int) -> pd.Series:
    """WMA على مؤشر الحوت."""
    return whale.rolling(period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
    )


def whale_strength(whale: pd.Series, peak_period: int = 50) -> pd.Series:
    """قوة الحوت كنسبة من الذروة."""
    peak = whale.rolling(peak_period).max()
    return pd.Series(np.where(peak > 0, whale / peak * 100, 0), index=whale.index)


def whale_spike(whale: pd.Series) -> pd.Series:
    """ارتداد الحوت من الصفر."""
    return (whale > whale.shift(1)) & (whale.shift(1) <= 0.02)


def volume_filter(df: pd.DataFrame, period: int = 20, mult: float = 1.5) -> pd.Series:
    """حجم > المتوسط × مضاعف."""
    return df['volume'] > df['volume'].rolling(period).mean() * mult


def sma50_daily(df: pd.DataFrame) -> pd.Series:
    """
    SMA50 اليومي.
    نستخدم shift(1) عشان ما نستخدم إغلاق اليوم الحالي.
    """
    daily = df.set_index('timestamp').resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    daily['sma50'] = daily['close'].rolling(50).mean()
    daily_map = daily['sma50'].shift(1)  # أمس فقط
    df_date = df['timestamp'].dt.floor('D')
    return df_date.map(daily_map)


def ema21(df: pd.DataFrame) -> pd.Series:
    """EMA21 عادي — الباك تست بيستخدم .shift(1) للهدف."""
    return df['close'].ewm(span=21, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low = df['high'], df['low']
    close_prev = df['close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def sell_signal(df: pd.DataFrame) -> pd.Series:
    """إشارة بيع — 6 شروط تصريف."""
    vs = df['volume'].rolling(20).mean()
    h20 = df['high'].rolling(20).max().shift(1)
    l10 = df['low'].rolling(10).min().shift(1)
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rsi = 100 - (100 / (1 + (gain.ewm(alpha=1/14, adjust=False).mean() / 
                              loss.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-10))))
    
    c1 = ((df['volume'] > vs * 1.5) & (df['close'] < df['open'])).astype(int)
    c2 = ((df['high'] > h20) & (df['close'] < h20)).astype(int)
    c3 = ((df['high'] > h20) & (df['close'] < df['open'])).astype(int)
    c4 = ((df['close'].shift(1) > df['open'].shift(1)) & (df['volume'] > vs * 1.5) & (df['close'] < df['open'])).astype(int)
    c5 = (df['low'] < l10).astype(int)
    c6 = ((df['high'] > df['high'].shift(1)) & (rsi < rsi.shift(1))).astype(int)
    
    return (c1 + c2 + c3 + c4 + c5 + c6) / 6.0 * 100


def swing_lows(df: pd.DataFrame, window: int = 5) -> np.ndarray:
    """قيعان سوينج — 5-bar fractals."""
    lows = df['low'].values
    n = len(lows)
    mask = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        if lows[i] == np.nanmin(lows[i-window:i+window+1]):
            mask[i] = True
    return mask
