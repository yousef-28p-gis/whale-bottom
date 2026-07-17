#!/usr/bin/env python3 -u
"""رسم شارتات - صفقتين ناجحة وصفتين خاسرة"""
import json, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CACHE_DIR = "/data/trading28/cache/ohlcv"
MONTHS = ["2026-04", "2026-05", "2026-06"]
TP=3.5; SL=1.5; PL=30; TRAIL=0.10; MH=6; STR=50; WHALE_MIN=0.50; COMM=0.20
plt.rcParams["font.family"] = "DejaVu Sans"

all_trades = []
print("جاري تحميل البيانات...", flush=True)
done = 0

for fname in sorted(os.listdir(CACHE_DIR)):
    if not fname.endswith(".json"): continue
    parts = fname.replace(".json","").rsplit("_",1)
    if len(parts) != 2: continue
    sym, mon = parts
    if mon not in MONTHS: continue
    
    with open(f"{CACHE_DIR}/{fname}") as f:
        try: data = json.load(f)
        except: continue
    
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)
    if len(df) < 500: continue
    
    LB=30
    df["lo"]=df["low"].rolling(LB).min()
    df["lc"]=abs(df["low"]-df["low"].shift(1))/df["low"]*100
    df["sm"]=df["lc"].ewm(span=3,adjust=False).mean()
    df["hi"]=df["sm"].rolling(LB).max()
    df["raw"]=np.where(df["low"]<=df["lo"],(df["sm"]+df["hi"]*2)/3,0)
    df["whale"]=df["raw"].ewm(span=3,adjust=False).mean().fillna(0)
    df["spike"]=(df["whale"]>df["whale"].shift(1))&(df["whale"].shift(1)<=0.03)
    df["wf"]=df["whale"].rolling(2).mean(); df["ws"]=df["whale"].rolling(5).mean()
    df["wp"]=df["whale"].rolling(50).max()
    df["str"]=(df["whale"]/df["wp"].replace(0,np.nan)*100).fillna(0)
    df["vma"]=df["volume"].rolling(20).mean()
    df["entry"]=(df["spike"]&(df["wf"]>df["ws"])&(df["str"]>STR)&(df["volume"]>df["vma"]*1.0))
    delta = df["close"].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100/(1+rs))
    
    for i in range(50, len(df)-10):
        row = df.iloc[i]
        if row["ts"].month < 4: continue
        if not row["entry"]: continue
        wv = float(row["whale"])
        if wv < WHALE_MIN: continue
        if i+1 < len(df) and float(df.iloc[i+1]["whale"]) >= 0.35: continue
        rsi = float(row["rsi"])
        if np.isnan(rsi) or rsi >= 25: continue
        if row["ts"].weekday() == 3: continue
        if row["ts"].hour in (1,3,6,12,0,4): continue
        ps=max(0,i-96); pb=float(df.iloc[ps]["close"]); ep=float(row["close"])
        if (ep-pb)/pb*100 >= 0: continue
        
        tp_p=ep*(1+TP/100); sl_p=ep*(1-SL/100)
        pl_p=ep+(tp_p-ep)*(PL/100)
        pl_trig=False; peak=ep; trail_p=0; exit_idx=None; exit_price=None; exit_reason=""
        for k in range(i+1, len(df)):
            cur=float(df.iloc[k]["close"]); h=(k-i)*0.25
            if h>MH:
                exit_idx=k; exit_price=cur; exit_reason="TIME"
                pnl=round((cur-ep)/ep*100-COMM,4); break
            if cur>=tp_p:
                exit_idx=k; exit_price=cur; exit_reason="TP"
                pnl=round(TP-COMM,4); break
            if cur<=sl_p:
                exit_idx=k; exit_price=cur; exit_reason="SL"
                pnl=round(-SL-COMM,4); break
            if not pl_trig and cur>=pl_p:
                pl_trig=True; peak=cur; trail_p=cur*(1-TRAIL/100)
            if pl_trig:
                if cur>peak: peak=cur; trail_p=cur*(1-TRAIL/100)
                if cur<=trail_p:
                    exit_idx=k; exit_price=trail_p; exit_reason="TRAIL"
                    pnl=round((trail_p-ep)/ep*100-COMM,4); break
        else:
            exit_idx=len(df)-1; exit_price=float(df.iloc[-1]["close"]); exit_reason="EOD"
            pnl=round((exit_price-ep)/ep*100-COMM,4)
        
        all_trades.append({
            "sym":sym, "mon":mon, "file":fname,
            "entry_idx":i, "exit_idx":exit_idx, "ep":ep,
            "pnl":pnl, "exit":exit_reason, "exit_price":exit_price,
            "whale_val":wv, "rsi":rsi, "dt":row["ts"]
        })
    done += 1
    if done % 400 == 0: print(f"  {done} ملف...", flush=True)

print(f"\nإجمالي الصفقات: {len(all_trades)}", flush=True)

wins = sorted([t for t in all_trades if t["pnl"]>0], key=lambda x: x["pnl"], reverse=True)
losses = sorted([t for t in all_trades if t["pnl"]<=0], key=lambda x: x["pnl"])
selected = wins[:2] + losses[:2]

for idx, tr in enumerate(selected):
    label = "WIN" if tr["pnl"]>0 else "LOSS"
    sym = tr["sym"]
    print(f"\n📊 {label}: {sym} | PnL: {tr['pnl']:+.2f}% | {tr['exit']} | Whale={tr['whale_val']:.3f} | RSI={tr['rsi']:.1f}", flush=True)
    
    with open(f"{CACHE_DIR}/{tr['file']}") as f:
        data = json.load(f)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)
    
    start_i = max(0, tr["entry_idx"]-48)
    end_i = min(len(df), tr["exit_idx"]+12)
    window = df.iloc[start_i:end_i].reset_index(drop=True)
    times = window["ts"]
    e_local = tr["entry_idx"] - start_i
    x_local = tr["exit_idx"] - start_i
    
    # RSI for window
    d2 = window["close"].diff()
    g2 = d2.where(d2>0,0).rolling(14).mean()
    l2 = (-d2.where(d2<0,0)).rolling(14).mean()
    rs2 = g2 / l2.replace(0, np.nan)
    rsi_w = 100 - (100/(1+rs2))
    
    # Whale for window
    LB=30
    window["lo"]=window["low"].rolling(LB).min()
    window["lc"]=abs(window["low"]-window["low"].shift(1))/window["low"]*100
    window["sm"]=window["lc"].ewm(span=3,adjust=False).mean()
    window["hi"]=window["sm"].rolling(LB).max()
    window["raw"]=np.where(window["low"]<=window["lo"],(window["sm"]+window["hi"]*2)/3,0)
    whale_w = pd.Series(window["raw"]).ewm(span=3,adjust=False).mean().fillna(0)
    
    # Plot
    color = "#00aa00" if tr["pnl"]>0 else "#dd0000"
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [5, 2, 2]}, sharex=True)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["top"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["right"].set_color("#444")
    
    # Candlestick
    for j in range(len(window)):
        o = window.iloc[j]["open"]; h = window.iloc[j]["high"]
        l = window.iloc[j]["low"]; c = window.iloc[j]["close"]
        clr = "#00aa00" if c >= o else "#dd0000"
        ax1.plot([times.iloc[j], times.iloc[j]], [l, h], color=clr, linewidth=1)
        ax1.plot([times.iloc[j], times.iloc[j]], [o, c], color=clr, linewidth=4)
    
    # Entry/Exit markers
    ax1.axvline(times.iloc[e_local], color="cyan", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.scatter(times.iloc[e_local], tr["ep"], color="cyan", s=120, marker="^", zorder=5, edgecolors="white")
    ax1.annotate(f"Entry {tr['ep']:.4f}", (times.iloc[e_local], tr["ep"]),
                xytext=(10, -20), textcoords="offset points", color="cyan", fontsize=10)
    
    ax1.scatter(times.iloc[x_local], tr["exit_price"], color=color, s=120, marker="X", zorder=5, edgecolors="white")
    ax1.annotate(f"Exit {tr['exit']} {tr['exit_price']:.4f}", (times.iloc[x_local], tr["exit_price"]),
                xytext=(10, 10), textcoords="offset points", color=color, fontsize=10)
    
    # Whale
    ax2.plot(times, whale_w, color="#ff9800", linewidth=1.5)
    ax2.axhline(0.50, color="orange", linestyle="--", linewidth=1, alpha=0.6)
    ax2.fill_between(times, 0, whale_w, alpha=0.3, color="#ff9800")
    ax2.set_ylabel("Whale", color="white")
    
    # RSI
    ax3.plot(times, rsi_w, color="#7c4dff", linewidth=1.5)
    ax3.axhline(25, color="green", linestyle="--", linewidth=1, alpha=0.6)
    ax3.axhline(70, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax3.fill_between(times, 0, 25, alpha=0.15, color="green")
    ax3.set_ylabel("RSI", color="white")
    ax3.set_ylim(0, 100)
    
    title = f"{'WIN' if tr['pnl']>0 else 'LOSS'} | {sym}/USDT | PnL: {tr['pnl']:+.2f}% | Whale={tr['whale_val']:.3f} | RSI={tr['rsi']:.1f} | Exit: {tr['exit']}"
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold")
    
    ax1.set_ylabel("Price", color="white")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()
    
    fname_out = f"/data/trading28/charts/trade_{idx+1}_{sym}_{label}.png"
    os.makedirs("/data/trading28/charts", exist_ok=True)
    fig.savefig(fname_out, dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {fname_out}", flush=True)

print("\n✨ تم!")
