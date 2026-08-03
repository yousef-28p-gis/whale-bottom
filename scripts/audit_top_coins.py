#!/usr/bin/env python3
"""
تدقيق AUDIO + RAD — فحص دقيق لكل صفقة
"""
import json, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

COMM = 0.002; CAP = 1000
DATA = '/data/trading28/data/whale_15m_1y'

def load(sym):
    with open(os.path.join(DATA, f'{sym}.json')) as f:
        d = json.load(f)
    return {
        'ts': pd.to_datetime(d['ts'], unit='ms'),
        'c': np.array(d['c'],float), 'h': np.array(d['h'],float),
        'l': np.array(d['l'],float), 'o': np.array(d['o'],float),
    }

def ema(s,p): return pd.Series(s).ewm(span=p,adjust=False).mean().values

def audit_coin(sym, LB, tp, sl, use_ssl=False, ssl_p=10):
    print(f'\n{"="*70}')
    print(f'🔍 تدقيق {sym} | {"Whale" if not use_ssl else f"W+SSL({ssl_p})"} | LB={LB} | TP{tp}%/SL{sl}%')
    print(f'{"="*70}')
    
    d = load(sym)
    c=d['c']; h=d['h']; l_=d['l']; o=d['o']; idx=d['ts']; n=len(c)
    
    # Price stats
    print(f'السعر: {c[0]:.4f} → {c[-1]:.4f} | تغير: {(c[-1]/c[0]-1)*100:+.1f}%')
    print(f'المدى: {c.min():.4f} - {c.max():.4f} | شموع: {n}')
    
    # Whale — double check no look-ahead
    ln = pd.Series(l_).shift(1).rolling(LB).min().values  # shift(1)!
    lc = np.zeros(n)
    for i in range(1,n): lc[i] = abs(l_[i]-l_[i-1])/l_[i]*100
    sc = pd.Series(lc).ewm(span=3,adjust=False).mean().values
    hc = pd.Series(sc).rolling(LB).max().values
    sr = np.where(l_<=ln, (sc+hc*2)/3, 0)
    wp = pd.Series(sr).ewm(span=3,adjust=False).mean().values
    wp_up = wp > np.roll(wp, 1)
    
    # Entry
    if use_ssl:
        sma_h = pd.Series(h).rolling(ssl_p).mean().values
        sma_l = pd.Series(l_).rolling(ssl_p).mean().values
        ssl_c = np.zeros(n, int)
        for i in range(ssl_p, n):
            if h[i-1]>sma_h[i-1]: ssl_c[i]=1
            else: ssl_c[i]=-1
    
    le = np.zeros(n, bool)
    for i in range(200, n):
        if use_ssl:
            if ssl_c[i]==1 and wp_up[i] and wp[i]>wp[i-2]*2 and wp[i]>0:
                le[i] = True
        else:
            if wp_up[i] and wp[i]>wp[i-2]*1.5 and wp[i]>0:
                le[i] = True
    
    print(f'إشارات الدخول: {le.sum()}')
    
    # Simulate with full trade audit
    trades = []
    pos = 0; ep = 0; ei = 0
    for i in range(200, n):
        if pos:
            # TP hit first (high)
            if h[i] >= ep*(1+tp/100):
                xp = ep*(1+tp/100)
                pnl = tp - COMM*100
                trades.append({'ei':ei,'xi':i,'ep':ep,'xp':xp,'pnl':pnl,'type':'TP','c_entry':c[ei],'c_exit':c[i]})
                pos = 0
            # SL hit (low)
            elif l_[i] <= ep*(1-sl/100):
                pnl = (c[i]/ep - 1)*100 - COMM*100
                xp = c[i]
                trades.append({'ei':ei,'xi':i,'ep':ep,'xp':xp,'pnl':pnl,'type':'SL','c_entry':c[ei],'c_exit':c[i]})
                pos = 0
        if not pos and le[i]:
            pos = 1; ep = c[i]; ei = i
    
    if pos:
        pnl = (c[-1]/ep - 1)*100 - COMM*100
        trades.append({'ei':ei,'xi':n-1,'ep':ep,'xp':c[-1],'pnl':pnl,'type':'OPEN','c_entry':c[ei],'c_exit':c[-1]})
    
    # ── Audit each trade ──
    print(f'\n📋 {len(trades)} صفقة:')
    print(f'{"#":>3} {"نوع":>5} {"دخول":>20} {"خروج":>20} {"سعر دخول":>10} {"سعر خروج":>10} {"PnL%":>8} {"تراكمي":>10}')
    print('-'*90)
    
    eq = CAP
    consecutive_losses = 0
    max_consecutive = 0
    big_wins = 0
    big_losses = 0
    
    for i, t in enumerate(trades):
        eq_before = eq
        eq *= (1 + t['pnl']/100)
        
        # Check for suspicious patterns
        bars = t['xi'] - t['ei']
        entry_ok = t['ep'] == t['c_entry']  # entry at close (correct)
        exit_ok = t['type']=='TP' or abs(t['xp'] - t['c_exit']) < 0.0001  # SL exit at close
        
        flags = []
        if t['pnl'] > tp*0.9: big_wins += 1
        if t['pnl'] < -sl*0.9: big_losses += 1
        if bars < 3: flags.append('⏱️سريع')
        if t['pnl'] < -10: flags.append('🚩خسارة كبيرة')
        if t['pnl'] > tp*1.5: flags.append('🚩ربح غير طبيعي')
        
        if t['pnl'] <= 0:
            consecutive_losses += 1
            max_consecutive = max(max_consecutive, consecutive_losses)
        else:
            consecutive_losses = 0
        
        flag_str = ' '.join(flags) if flags else ''
        print(f'{i+1:>3} {t["type"]:>5} {str(idx[t["ei"]])[:19]:>20} {str(idx[t["xi"]])[:19]:>20} '
              f'${t["ep"]:>8.4f} ${t["xp"]:>8.4f} {t["pnl"]:>+7.2f}% ${eq:>9.1f} {flag_str}')
    
    # ── Statistics ──
    w = [t for t in trades if t['pnl']>0]
    lo = [t for t in trades if t['pnl']<=0]
    wr = len(w)/len(trades)*100 if trades else 0
    aw = np.mean([t['pnl'] for t in w]) if w else 0
    al = np.mean([t['pnl'] for t in lo]) if lo else 0
    
    # Profit curve for DD
    eq_curve = [CAP]; eq_tmp = CAP
    for t in trades:
        eq_tmp *= (1+t['pnl']/100)
        eq_curve.append(eq_tmp)
    dd = ((pd.Series(eq_curve) - pd.Series(eq_curve).expanding().max()) / pd.Series(eq_curve).expanding().max() * 100).min()
    
    # Verify: manual compound vs formula
    manual_eq = CAP
    for t in trades:
        manual_eq *= (1+t['pnl']/100)
    
    print(f'\n📊 إحصائيات:')
    print(f'   صفقات: {len(trades)} | {len(w)}W/{len(lo)}L | WR: {wr:.1f}%')
    print(f'   متوسط ربح: +{aw:.2f}% | متوسط خسارة: {al:.2f}% | R:R: {aw/abs(al) if al else 0:.2f}')
    print(f'   أقصى سحب: {dd:.1f}% | أقصى خسائر متتالية: {max_consecutive}')
    print(f'   محفظة نهائية: ${eq:.1f} | ربح: ${eq-CAP:+.1f} ({((eq/CAP)**(365/365)-1)*100:+.1f}% سنوي)')
    print(f'   تدقيق حسابي: يدوي=${manual_eq:.1f} | تلقائي=${eq:.1f} | متطابق: {"✅" if abs(manual_eq-eq)<0.01 else "❌"}')
    
    # 🚩 RED FLAGS
    print(f'\n🚩 تدقيق الأخطاء:')
    issues = []
    if wr > 70: issues.append(f'WR={wr:.1f}% > 70% — مشبوه!')
    if dd > -1: issues.append(f'DD={dd:.1f}% قريب من الصفر — مشبوه!')
    if eq/CAP > 6: issues.append(f'عائد {(eq/CAP-1)*100:.0f}% > 500% — خطأ حسابي محتمل!')
    if max_consecutive > 15: issues.append(f'{max_consecutive} خسارة متتالية — طبيعي ولا؟')
    if len(trades) > 500: issues.append(f'{len(trades)} صفقة كثيرة — فلترة ضعيفة')
    
    if issues:
        for iss in issues: print(f'   ❌ {iss}')
    else:
        print(f'   ✅ لا توجد أخطاء واضحة')
    
    return {'sym':sym,'eq':eq,'wr':wr,'dd':dd,'t':len(trades),
            'w':len(w),'l':len(lo),'max_cons':max_consecutive,'issues':issues}

# ── Run audits ──
audit_coin('AUDIO', LB=70, tp=5.0, sl=2.5, use_ssl=False)
audit_coin('RAD', LB=30, tp=5.0, sl=2.5, use_ssl=False)
audit_coin('NMR', LB=70, tp=5.0, sl=2.5, use_ssl=False)

# Also audit a normal coin for comparison
audit_coin('SUSHI', LB=50, tp=5.0, sl=2.5, use_ssl=True, ssl_p=20)

print('\n\n✅ تم التدقيق')
