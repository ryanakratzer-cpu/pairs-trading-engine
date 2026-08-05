#!/usr/bin/env bash
# 2026-08-03 close watcher: record the gated session, then grade the forecast.
cd "C:/Users/ryana/OneDrive/Desktop/Ryan's Obsidian/01_RAW_CLIPS/quant_research/pairs-trading-engine" || exit 1
until [ "$(date +%H%M)" -ge 1635 ]; do sleep 120; done

echo "=== POST-CLOSE 2026-08-03 $(date '+%H:%M %Z') ==="
py run_daily_tracker.py 2>&1 | tail -20
echo
echo "=== FORECAST SCORECARD (pre-registered 09:35) ==="
py - <<'PY'
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import coint
from data.loader import fetch_price_history, align_and_clean
from screening.cointegration import test_pair_cointegration
from signals.spread import SignalConfig, build_spread, rolling_zscore, generate_signals
from screening.events import event_exclusion_mask
from screening.focus_book import focus_pairs, focus_tickers
cfg=SignalConfig()
h,_=align_and_clean(fetch_price_history(focus_tickers()+["SPY"],start="2024-02-01",end="2026-08-04",use_cache=False))
prev,today=h.index[-2],h.index[-1]
print(f"anchor {prev.date()} -> close {today.date()}\n")
entries=[]; tot=0.0; zs={}; coints={}
for a,b in focus_pairs():
    r=test_pair_cointegration(h[a].iloc[-252:],h[b].iloc[-252:])
    z=rolling_zscore(build_spread(h[a],h[b],r.hedge_ratio),cfg.zscore_window).dropna()
    sig=generate_signals(z,cfg,tradeable=event_exclusion_mask(z.index))
    ev=str(sig["event"].iloc[-1]); held=int(sig["position"].iloc[-2]) if len(sig)>=2 else 0
    zs[f"{a}/{b}"]=float(z.iloc[-1]); coints[f"{a}/{b}"]=coint(np.log(h[a].iloc[-252:]),np.log(h[b].iloc[-252:]))[1]
    if ev in ("ENTER_LONG_SPREAD","ENTER_SHORT_SPREAD"): entries.append(f"{a}/{b}")
    sp=lambda pa,pb: np.log(pa)-r.hedge_ratio*np.log(pb)
    pnl=held*(sp(h[a][today],h[b][today])-sp(h[a][prev],h[b][prev]))*10_000
    tot+=pnl
    print(f"{a+'/'+b:11s} z={zs[f'{a}/{b}']:+6.2f} held={held:+d} {ev:18s} EGp={coints[f'{a}/{b}']:.4f} P&L {pnl:+8.2f}")
spy=100*(h["SPY"][today]/h["SPY"][prev]-1)
nc=sum(1 for v in coints.values() if v<0.05)
print(f"\nBOOK P&L {tot:+.2f}   SPY {spy:+.2f}%   pairs still cointegrated: {nc}/6")
print("\n--- SCORECARD ---")
print(f"#1 no new entry fired?        {'HIT ' if not entries else 'MISS'}  ({entries or 'none'})")
print(f"#2 book P&L exactly 0.00?     {'HIT ' if abs(tot)<1e-9 else 'MISS'}  ({tot:+.2f})")
print(f"#3 JNJ/MRK |z|<2.0?           {'HIT ' if abs(zs['JNJ/MRK'])<2.0 else 'MISS'}  (z={zs['JNJ/MRK']:+.2f})")
print(f"#4 all 6 still cointegrated?  {'HIT ' if nc==6 else 'MISS'}  ({nc}/6)")
PY
