"""
تعزيز ذكي (متدرج + مرتبط بالحوت) + خروج جزئي (50% EMA21 + 50% trail)
"""
import sys
sys.path.insert(0, '/data/trading28')
import pandas as pd, numpy as np
from core.indicators import *

df = pd.read_csv('/data/trading28/backtests/cache/FET_USDT_15m_FULL.csv')
df['timestamp'] = pd.to_datetime(df['ts']); df = df.sort_values('timestamp').reset_index(drop=True)
CAP=1000.0; n=len(df)
whale_raw = whale_indicator(df,200)

# إشارة أساسية
entry = (
    whale_spike(whale_raw) & (whale_ma(whale_raw,20) > whale_ma(whale_raw,50)) &
    (whale_strength(whale_raw,50) > 50) & volume_filter(df) &
    (df['close'] > sma50_daily(df))
)
ema=ema21(df); sell=sell_signal(df); sm=swing_lows(df,5)

def run_smart(entry_signal, use_smart_dca=True, use_partial_exit=True):
    """
    Smart DCA: 25% → 35% (if whale rising) → 40% (if whale rising again)
    Partial exit: 50% at EMA21, 50% trails at 0.5%
    """
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
                pl_price = ep + (tp-ep)*60/100
                trade={'ei':i,'et':ts,
                       'e1':ep,'e2':None,'e3':None,
                       'ae':ep,'al':0.25,
                       'sl':sl,'tp':tp,'pl':pl_price,'pl_act':False,
                       'hi':ep,'hi2':ep,
                       'dca1':False,'dca2':False,
                       'partial':False,  # تم الخروج الجزئي؟
                       'trail_sl':None}   # وقف متحرك للجزء المتبقي
                in_trade=True
        else:
            if row['high']>trade['hi']: trade['hi']=row['high']
            
            # Profit Lock
            if not trade['pl_act'] and row['high'] >= trade['pl']:
                trade['pl_act']=True
            
            # Smart DCA
            if use_smart_dca:
                s2=max(0,trade['ei']+1); ns=df.iloc[s2:i+1][sm[s2:i+1]]
                if len(ns)>0:
                    lowest_new = ns['low'].min()
                    
                    # DCA 1: 35% if new swing low + whale rising
                    if not trade['dca1'] and lowest_new < trade['e1'] and whale_raw.iloc[i] > whale_raw.iloc[i-1]:
                        trade['e2']=row['close']
                        trade['ae']=(trade['e1']*25 + trade['e2']*35)/60
                        trade['al']=0.60; trade['dca1']=True
                        trade['sl']=ns['low'].min()*0.998
                        trade['pl']=trade['ae']+(trade['tp']-trade['ae'])*60/100
                        if row['high']>=trade['pl']: trade['pl_act']=True
                    
                    # DCA 2: 40% if second new swing low + whale rising
                    elif trade['dca1'] and not trade['dca2'] and lowest_new < trade['e2'] and whale_raw.iloc[i] > whale_raw.iloc[i-1]:
                        trade['e3']=row['close']
                        trade['ae']=(trade['e1']*25 + trade['e2']*35 + trade['e3']*40)/100
                        trade['al']=1.0; trade['dca2']=True
                        trade['sl']=ns['low'].min()*0.998
                        trade['pl']=trade['ae']+(trade['tp']-trade['ae'])*60/100
                        if row['high']>=trade['pl']: trade['pl_act']=True
            
            # Trail SL
            st2=max(0,i-100); swt=df.iloc[st2:i+1][sm[st2:i+1]]
            if len(swt)>0:
                nsl=swt['low'].min()*0.998
                if nsl>trade['sl']: trade['sl']=nsl
            
            if trade['pl_act']:
                trail_sl = trade['hi'] * (1 - 0.3/100)
                if trail_sl > trade['sl']: trade['sl'] = trail_sl
            
            # ══════ الخروج الجزئي ══════
            if use_partial_exit and not trade.get('partial'):
                if row['high'] >= trade['tp']:
                    # بيع 50% عند TP، الباقي مع trailing stop
                    partial_pnl = (trade['tp'] - trade['ae'])/trade['ae'] - 0.002
                    half_eff = partial_pnl * trade['al'] * 0.5
                    
                    monthly_pnl[mk]=monthly_pnl.get(mk,0.0)+half_eff*100
                    capital*=(1+half_eff)
                    if capital>peak: peak=capital
                    
                    trade['partial']=True
                    trade['al']*=0.5  # نقص التعرض للنصف
                    trade['hi2']=row['high']  # أعلى سعر بعد الخروج الجزئي
                    trade['trail_sl']=trade['tp']*0.995  # trail 0.5%
                    trade['sl']=trade['trail_sl']
                    trade['tp']=trade['tp']*10  # TP بعيد (لن يلمس)
                    trade['pl_act']=False  # PL تم استهلاكه مع partial exit
                    continue
            
            # ══════ بعد الخروج الجزئي: trailing stop ══════
            if trade.get('partial') and trade.get('trail_sl'):
                if row['high']>trade['hi2']: trade['hi2']=row['high']
                trade['trail_sl']=trade['hi2']*0.995
                trade['sl']=trade['trail_sl']
            
            # ══════ خروج نهائي ══════
            er=None; epx=None; hrs=(ts-trade['et']).total_seconds()/3600
            tp_h=row['high']>=trade['tp']
            sl_h=(row['high']>=trade['sl']) if trade['sl']>trade['ae'] else (row['low']<=trade['sl'])
            
            if tp_h: er,epx='TP',trade['tp']
            elif i>=2 and sell.iloc[i-1]>=60: er,epx='SELL',row['close']
            elif sl_h:
                if trade.get('partial'): er,epx='TRAIL',trade['sl']
                elif trade['pl_act']: er,epx='PL',trade['sl']
                else:
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
                trades.append({'pnl':pnl*100,'eff':eff*100,'er':er,
                               'dca':(trade.get('dca1',False) or trade.get('dca2',False)),
                               'partial':trade.get('partial',False)})
                in_trade=False; trade=None
    
    tdf=pd.DataFrame(trades)
    if len(tdf)==0: return {}
    w=tdf[tdf['pnl']>0]; wr=len(w)/len(tdf)*100
    rets=tdf['eff'].values/100
    sharpe=rets.mean()/rets.std()*np.sqrt(len(rets)) if rets.std()>0 else 0
    partial_n = tdf['partial'].sum() if 'partial' in tdf.columns else 0
    trail_n = len(tdf[tdf['er']=='TRAIL']) if 'er' in tdf.columns else 0
    return {'t':len(tdf),'wr':wr,'c':capital,'ret':(capital/CAP-1)*100,
            'dd':max_dd*100,'sharpe':sharpe,'partial':partial_n,'trail':trail_n,
            'tdf':tdf}

# ═══════════════════════════════════════════════
print(f"📦 {len(df)} شمعة | 🚦 {entry.sum()} إشارة\n")
print(f"{'='*95}")
print(f"🏆 تعزيز ذكي + خروج جزئي")
print(f"{'='*95}")

configs = [
    ('1️⃣ الأساس (25/75 DCA + PL 60% + 4hr)', lambda: run_smart(entry, False, False)),
    ('2️⃣ تعزيز ذكي فقط (25→35→40 + شرط الحوت)', lambda: run_smart(entry, True, False)),
    ('3️⃣ خروج جزئي فقط (50% EMA21 + 50% trail)', lambda: run_smart(entry, False, True)),
    ('4️⃣ الاثنين معاً', lambda: run_smart(entry, True, True)),
]

for name, fn in configs:
    r = fn()
    if not r: continue
    trail_info = f" | جزئي={r['partial']}" if r['partial']>0 else ""
    trail_info += f" | TRAIL={r['trail']}" if r['trail']>0 else ""
    print(f"\n{name}: {r['t']}T | WR={r['wr']:.1f}% | ${r['c']:.0f} (+{r['ret']:.0f}%) | DD={r['dd']:.1f}% | Sharpe={r['sharpe']:.2f}{trail_info}")
    
    tdf = r['tdf']
    exits = tdf['er'].value_counts().to_dict()
    print(f"   مخارج: {exits}")

print(f"\n✅ تم")
