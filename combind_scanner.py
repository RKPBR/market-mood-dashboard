"""
Combined "Market Mood" Scanner — Step 4 of the Master Plan (wiring all 4 layers together)
============================================================================================
IMPORTANT: This is a TRIAL / PARALLEL scanner, per Master Plan Section 3 Step 5.
It does NOT replace "1990". Run both side by side for a while before trusting this one.

WHAT THIS SCRIPT DOES, LAYER BY LAYER
--------------------------------------
Layer 1 (market-wide filter): Is Nifty 50 above or below its 200-day EMA?
    Above -> normal trading (layers 2-4 run as usual)
    Below -> flagged as RISK-OFF; this run will still show you what WOULD have
             triggered, but prints a clear warning to lean toward cash / reduce size.

Layer 2 (sector rotation): Ranks ALL sectors by relative strength (% return over the
    last RS_LOOKBACK_DAYS, on the same custom equal-weight sector index used in
    regime_detector.py). The Top-3 strongest sectors are used to gate the 1990 leg
    (see Layer 4) — Bollinger does NOT use this Top-3 gate (see FIX note below).

Layer 3 (regime detection): Computed for EVERY sector (not just the Top-3), using the
    SAME ADX(14) + 200EMA-slope logic as regime_detector.py (imported directly from
    that file so both scripts always agree — no duplicated, possibly-drifting logic).

Layer 4 (strategy routing + entry scan):
    Strong Uptrend + in today's Top-3 -> "1990" logic (RSI(2)<10 AND Close>200EMA),
                         scanned on ALL stocks in that sector.
    Sideways/Choppy (Top-3 NOT required) -> Bollinger Band mean-reversion
                         (Close < 20-day band, 1.5 std), but ONLY on stocks in
                         bollinger_tradeable.csv — the 20 individually-proven stocks.
    Strong Downtrend  -> no entries (cash), as agreed — CNC shorting isn't practical.
    Transitional      -> no entries (regime unclear).

FIX (validated via combined_system_backtest.py historical replay): gating Bollinger by
Top-3 membership too starved it to just 8 trades in 3 years — a sector strong enough to
rank in the daily Top-3 almost never coincides with being Sideways/Choppy at the same
time. Removing that gate for Bollinger only (keeping it for 1990, which measurably
improved with it) raised Bollinger to 75 trades/11 stocks/66.7% win/+1.45%-per-trade
over the same 3 years — this script now matches that validated behavior.

REQUIRED FILES IN THE SAME COLAB SESSION (upload/keep all of these):
    - nifty500_stocklist.csv     (symbol, yfinance_symbol, company_name, industry)
    - regime_detector.py         (this script imports functions from it — must be
                                   present as a .py file in the same folder, not just
                                   run once and discarded)
    - bollinger_tradeable.csv    (output of bollinger_backtest.py — the 20 validated
                                   Sideways-regime stocks)

OUTPUT: combined_scanner_today.csv — the final "TRADEABLE TODAY" list, each row tagged
with which sector/regime/strategy triggered it, plus entry price and ATR-based stop-loss.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from regime_detector import build_sector_index, compute_adx, classify_regime, fetch_stock_ohlc

# ============================== TUNABLE PARAMETERS ==============================
STOCK_LIST_CSV        = "nifty500_stocklist.csv"
TRADEABLE_BOLLINGER   = "bollinger_tradeable.csv"
NIFTY50_TICKER        = "^NSEI"
PERIOD                = "3y"

RS_LOOKBACK_DAYS      = 21        # ~1 month, for Layer 2 sector relative-strength ranking
TOP_N_SECTORS         = 3         # Layer 2: how many top sectors to trade today
MAX_STOCKS_PER_SECTOR = 25        # same cap as regime_detector.py, for speed/consistency

RSI_PERIOD            = 2
RSI_ENTRY_MAX         = 10        # "1990" entry: RSI(2) < 10
EMA_TREND_WINDOW      = 200
ATR_WINDOW            = 14
ATR_SL_MULT           = 2.0

BB_WINDOW             = 20
BB_NDEV               = 1.5
# ==================================================================================


def compute_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, window=ATR_WINDOW):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def layer1_market_filter():
    print("Layer 1: checking Nifty 50 vs its 200-day EMA...")
    try:
        nifty = yf.download(NIFTY50_TICKER, period=PERIOD, interval="1d", progress=False, auto_adjust=True)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        ema200 = nifty["Close"].ewm(span=EMA_TREND_WINDOW, adjust=False).mean()
        last_close = nifty["Close"].iloc[-1]
        last_ema = ema200.iloc[-1]
        risk_on = last_close > last_ema
        print(f"  Nifty50 close={last_close:.1f}  200EMA={last_ema:.1f}  "
              f"-> {'RISK-ON (above 200EMA)' if risk_on else 'RISK-OFF (below 200EMA)'}")
        return risk_on
    except Exception as e:
        print(f"  Could not fetch Nifty50 ({e}) — defaulting to RISK-ON, verify manually.")
        return True


def layer2_rank_sectors(sectors):
    print(f"\nLayer 2: ranking {len(sectors)} sectors by {RS_LOOKBACK_DAYS}-day relative strength...")
    scores = []
    for sector, symbols in sectors.items():
        syms = symbols[:MAX_STOCKS_PER_SECTOR] if MAX_STOCKS_PER_SECTOR else symbols
        sector_df, used = build_sector_index(syms, PERIOD)
        if sector_df is None or used < 3 or len(sector_df) < RS_LOOKBACK_DAYS + 5:
            continue
        recent_return = (sector_df["Close"].iloc[-1] / sector_df["Close"].iloc[-RS_LOOKBACK_DAYS] - 1) * 100
        scores.append((sector, round(recent_return, 2), sector_df))

    scores.sort(key=lambda x: x[1], reverse=True)
    print("  Sector relative-strength ranking (top to bottom):")
    for sector, ret, _ in scores:
        print(f"    {sector:30s} {ret:+.2f}%")

    top = scores[:TOP_N_SECTORS]
    print(f"\n  TOP {TOP_N_SECTORS} sectors today (for the 1990/Uptrend leg only): {[s[0] for s in top]}")
    return scores  # ALL ranked sectors, not just the top N


def layer3_regime_for_sector(sector_df):
    sector_df = sector_df.copy()
    sector_df["EMA200"] = sector_df["Close"].ewm(span=EMA_TREND_WINDOW, adjust=False).mean()
    sector_df["ADX14"] = compute_adx(sector_df)
    sector_df["EMA_SLOPE"] = sector_df["EMA200"] - sector_df["EMA200"].shift(10)
    latest = sector_df.iloc[-1]
    regime = classify_regime(latest["ADX14"], latest["EMA_SLOPE"])
    return regime, round(latest["ADX14"], 2), round(latest["EMA_SLOPE"], 3)


def scan_uptrend_stock(symbol):
    df = fetch_stock_ohlc(symbol, PERIOD)
    if df is None:
        return None
    close = df["Close"]
    ema200 = close.ewm(span=EMA_TREND_WINDOW, adjust=False).mean()
    rsi2 = compute_rsi(close)
    atr = compute_atr(df)
    last = -1
    if close.iloc[last] > ema200.iloc[last] and rsi2.iloc[last] < RSI_ENTRY_MAX:
        entry = close.iloc[last]
        sl = entry - ATR_SL_MULT * atr.iloc[last]
        return round(entry, 2), round(sl, 2)
    return None


def scan_sideways_stock(symbol):
    df = fetch_stock_ohlc(symbol, PERIOD)
    if df is None:
        return None
    close = df["Close"]
    mid = close.rolling(BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW).std()
    lower = mid - BB_NDEV * std
    atr = compute_atr(df)
    last = -1
    if close.iloc[last] < lower.iloc[last]:
        entry = close.iloc[last]
        sl = entry - ATR_SL_MULT * atr.iloc[last]
        return round(entry, 2), round(sl, 2)
    return None


def main():
    universe = pd.read_csv(STOCK_LIST_CSV)
    sectors = universe.groupby("industry")["yfinance_symbol"].apply(list).to_dict()
    sector_stocks = universe.groupby("industry")["yfinance_symbol"].apply(list).to_dict()

    try:
        bollinger_ok = set(pd.read_csv(TRADEABLE_BOLLINGER)["symbol"].tolist())
    except Exception:
        print(f"WARNING: could not read {TRADEABLE_BOLLINGER} — Sideways regime will have NO stocks to trade.")
        bollinger_ok = set()

    risk_on = layer1_market_filter()
    if not risk_on:
        print("\n*** RISK-OFF: Nifty50 below 200EMA. Master Plan says lean toward cash / reduce size. ***")
        print("*** Entries below are shown for information only — consider skipping new entries today. ***\n")

    all_sectors_ranked = layer2_rank_sectors(sectors)
    top_sector_names = {s[0] for s in all_sectors_ranked[:TOP_N_SECTORS]}

    results = []
    print("\nLayer 3 + 4: regime for every sector, then routed entry scan "
          "(1990 -> Top-3 Uptrend sectors only; Bollinger -> ANY Sideways sector)...")
    for sector, ret, sector_df in all_sectors_ranked:
        regime, adx, slope = layer3_regime_for_sector(sector_df)
        in_top3 = sector in top_sector_names
        print(f"\n  Sector: {sector}  (RS={ret:+.2f}%, ADX={adx}, slope={slope}) -> {regime}"
              f"{'  [TOP-3]' if in_top3 else ''}")

        if regime == "Strong Uptrend" and in_top3:
            for sym in sector_stocks[sector]:
                hit = scan_uptrend_stock(sym)
                if hit:
                    results.append({"symbol": sym, "sector": sector, "regime": regime,
                                     "strategy": "1990 (RSI2+200EMA)",
                                     "entry": hit[0], "stop_loss": hit[1]})

        elif regime == "Strong Uptrend" and not in_top3:
            print("    Strong Uptrend but not in today's Top-3 -> skipped (1990 leg requires Top-3).")

        elif regime == "Sideways/Choppy":
            # No Top-3 requirement here — a sector strong enough to rank in the
            # daily Top-3 almost never coincides with Sideways/Choppy, so gating
            # Bollinger this way was starving it of trades (see project notes).
            candidates = [s for s in sector_stocks[sector] if s in bollinger_ok]
            if not candidates:
                print(f"    (no stocks from this sector are in the validated Bollinger list — skipping)")
            for sym in candidates:
                hit = scan_sideways_stock(sym)
                if hit:
                    results.append({"symbol": sym, "sector": sector, "regime": regime,
                                     "strategy": "Bollinger mean-reversion",
                                     "entry": hit[0], "stop_loss": hit[1]})

        else:
            print(f"    Regime is '{regime}' -> no entries taken (per Master Plan rule).")

    out = pd.DataFrame(results)
    out.to_csv("combined_scanner_today.csv", index=False)

    print("\n================ TRADEABLE TODAY (combined system) ================")
    if out.empty:
        print("No entries today across the top sectors.")
    else:
        print(out.to_string(index=False))
    print("======================================================================")
    print("\nSaved -> combined_scanner_today.csv")
    print("Reminder: this is running IN PARALLEL with '1990' — compare, don't replace, for now.")


if __name__ == "__main__":
    main()
