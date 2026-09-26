"""
Market Mood — "1990" leg: FROZEN engine v1 (26 Sep 2026)

Exactly the same rules and code as step5_locked_test.py (the script that passed the
one-time LOCKED test). Do NOT change the numbers below — only bug fixes are allowed.
The dashboard imports build() from here so live signals == backtest signals.
"""
import numpy as np
import pandas as pd

# ------------------------- FROZEN RULES (બદલવા નહીં) -------------------------
EMA_WIN, RSI_PERIOD, RSI_ENTRY, RSI_EXIT, SMA_EXIT = 200, 2, 10, 70, 5
ATR_WIN, ATR_MULT = 14, 2.0
ADX_WIN, ADX_STRONG, SLOPE_LB = 14, 25, 10
RS_LB, TOP_N, MIN_SECTOR_STOCKS = 21, 2, 3
NIFTY_DIP = -0.005            # signal ના દિવસે Nifty 0.5% કે વધુ પડ્યો હોવો જોઈએ
MIN_BARS, MAX_HOLD, JUMP = 220, 10, 0.40
CAPITAL, SLOT, MAX_SLOTS = 150000, 25000, 6
SIDE_COST, DP = 0.0015, 16.0  # 0.15% દરેક બાજુ (0.30% round trip) + ₹16 DP દરેક વેચાણ પર


# ============================== ENGINE ==============================
def splice_corporate_actions(d):
    """>40% એક-દિવસ ફેરફાર = corporate action: અગાઉના બધા ભાવ event-day open પર rescale."""
    d = d.copy()
    c = d["C"].to_numpy()
    o = d["O"].to_numpy()
    for i in range(1, len(d)):
        if c[i - 1] > 0 and abs(c[i] / c[i - 1] - 1) > JUMP:
            f = o[i] / c[i - 1] if (o[i] > 0 and np.isfinite(o[i])) else c[i] / c[i - 1]
            d.iloc[:i, d.columns.get_indexer(["O", "H", "L", "C"])] *= f
            c = d["C"].to_numpy()
            o = d["O"].to_numpy()
    return d


def build(w, lst, trade_start, trade_end=None):
    """w: price rows (Date, Ticker, Open, High, Low, Close, AdjClose, Volume); lst: universe list."""
    lst = lst[~lst["symbol"].astype(str).str.upper().str.startswith("DUMMY")]
    sec_of = dict(zip(lst["yfinance_symbol"], lst["industry"]))
    nif = w[w["Ticker"] == "^NSEI"].set_index("Date").sort_index()
    cal = pd.DatetimeIndex(nif.index)
    have = set(w["Ticker"].unique())
    tickers = sorted(t for t in sec_of if t in have)
    n, m = len(cal), len(tickers)
    A = {k: np.full((n, m), np.nan) for k in
         ["O", "H", "L", "C", "rawC", "ema", "rsi", "sma5", "atr", "bars"]}
    g = dict(tuple(w[w["Ticker"].isin(tickers)].groupby("Ticker")))
    for j, t in enumerate(tickers):
        s = g[t].set_index("Date").sort_index()
        s = s[s.index.isin(cal)]
        s = s[~s.index.duplicated(keep="last")]
        f = s["AdjClose"] / s["Close"]
        d = pd.DataFrame({"O": s["Open"] * f, "H": s["High"] * f, "L": s["Low"] * f,
                          "C": s["AdjClose"]}, index=s.index).dropna()
        if len(d) < 2:
            continue
        d = splice_corporate_actions(d)
        c = d["C"]
        delta = c.diff()
        ag = delta.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
        al = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
        pc = c.shift(1)
        tr = pd.concat([d["H"] - d["L"], (d["H"] - pc).abs(), (d["L"] - pc).abs()], axis=1).max(axis=1)
        cols = {"O": d["O"], "H": d["H"], "L": d["L"], "C": c, "rawC": s.loc[d.index, "Close"],
                "ema": c.ewm(span=EMA_WIN, adjust=False).mean(), "rsi": 100 - 100 / (1 + ag / al),
                "sma5": c.rolling(SMA_EXIT).mean(), "atr": tr.ewm(alpha=1 / ATR_WIN, adjust=False).mean(),
                "bars": pd.Series(np.arange(1, len(d) + 1), index=d.index, dtype=float)}
        pos = cal.get_indexer(d.index)
        for k, v in cols.items():
            A[k][pos, j] = v.to_numpy()

    # Layer 2 + 3: sector index (equal-weight chained daily returns), 21-day strength, ADX + EMA slope
    sectors = sorted(set(sec_of[t] for t in tickers))
    sec_idx = np.array([sectors.index(sec_of[t]) for t in tickers])
    C, H, L = A["C"], A["H"], A["L"]
    prevC = np.vstack([np.full((1, m), np.nan), C[:-1]])
    rc, rh, rl = C / prevC - 1, H / prevC - 1, L / prevC - 1
    k = len(sectors)
    rs_arr = np.full((n, k), np.nan)
    up_arr = np.zeros((n, k), dtype=bool)
    adx_arr = np.full((n, k), np.nan)      # display only
    slope_arr = np.full((n, k), np.nan)    # display only
    for si in range(k):
        cols = sec_idx == si
        valid = np.isfinite(rc[:, cols])
        cnt = valid.sum(axis=1)
        with np.errstate(all="ignore"):
            mc = np.where(cnt >= MIN_SECTOR_STOCKS, np.nanmean(np.where(valid, rc[:, cols], np.nan), axis=1), np.nan)
            mh = np.where(cnt >= MIN_SECTOR_STOCKS, np.nanmean(np.where(valid, rh[:, cols], np.nan), axis=1), np.nan)
            ml = np.where(cnt >= MIN_SECTOR_STOCKS, np.nanmean(np.where(valid, rl[:, cols], np.nan), axis=1), np.nan)
        idx = 100 * np.cumprod(1 + np.nan_to_num(mc))
        prev_idx = np.concatenate([[100.0], idx[:-1]])
        ok = np.isfinite(mc)
        sdf = pd.DataFrame({"H": np.where(ok, prev_idx * (1 + mh), np.nan),
                            "L": np.where(ok, prev_idx * (1 + ml), np.nan),
                            "C": np.where(ok, idx, np.nan)}, index=cal).dropna()
        if len(sdf) < 2:
            continue
        ema = sdf["C"].ewm(span=EMA_WIN, adjust=False).mean()
        slope = ema - ema.shift(SLOPE_LB)
        pc, ph, pl = sdf["C"].shift(1), sdf["H"].shift(1), sdf["L"].shift(1)
        tr = pd.concat([sdf["H"] - sdf["L"], (sdf["H"] - pc).abs(), (sdf["L"] - pc).abs()], axis=1).max(axis=1)
        upm, dnm = sdf["H"] - ph, pl - sdf["L"]
        pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=sdf.index)
        mdm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=sdf.index)
        atr = tr.ewm(alpha=1 / ADX_WIN, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=1 / ADX_WIN, adjust=False).mean() / atr
        mdi = 100 * mdm.ewm(alpha=1 / ADX_WIN, adjust=False).mean() / atr
        adx = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=1 / ADX_WIN, adjust=False).mean()
        p = cal.get_indexer(sdf.index)
        rs_arr[p, si] = (sdf["C"] / sdf["C"].shift(RS_LB) - 1).to_numpy()
        up_arr[p, si] = ((adx > ADX_STRONG) & (slope > 0)).to_numpy()
        adx_arr[p, si] = adx.to_numpy()
        slope_arr[p, si] = slope.to_numpy()
    rsm = np.where(np.isfinite(rs_arr), rs_arr, -np.inf)
    order = np.argsort(-rsm, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(k)[None, :] + 1
    rank = np.where(np.isfinite(rs_arr), rank, 99)

    # Layer 1 + Nifty dip
    nc = nif["Close"]
    risk_on = (nc > nc.ewm(span=EMA_WIN, adjust=False).mean()).to_numpy()
    nifty_ret1 = (nc / nc.shift(1) - 1).to_numpy()

    # Layer 4: Step-2 base (Top-3) signal + frozen filters
    rank_stock = rank[:, sec_idx]
    with np.errstate(invalid="ignore"):
        base = (risk_on[:, None] & (rank_stock <= 3) & up_arr[:, sec_idx] & (A["bars"] >= MIN_BARS)
                & (A["C"] > A["ema"]) & (A["rsi"] < RSI_ENTRY))
    filt = (rank_stock <= TOP_N) & (nifty_ret1 <= NIFTY_DIP)[:, None]
    start = cal.searchsorted(pd.Timestamp(trade_start))
    base[:start] = False
    end_idx = n - 1
    if trade_end is not None:
        last = cal.searchsorted(pd.Timestamp(trade_end), side="right") - 1
        base[last:] = False                       # છેલ્લી entry પણ LOCKED ગાળામાં જ થાય
        end_idx = min(n - 1, last + 15)           # ખુલ્લા trades બંધ થવા માટે થોડા દિવસ
    return dict(cal=cal, tickers=tickers, sectors=sectors, sec_idx=sec_idx, A=A, rank=rank,
                base=base, filt=filt, frozen=base & filt, stop=A["C"] - ATR_MULT * A["atr"],
                start=start, end_idx=end_idx, nifty=nc, nifty_ret1=nifty_ret1,
                risk_on=risk_on, rs=rs_arr, up=up_arr, adx=adx_arr, slope=slope_arr,
                nifty_ema=nc.ewm(span=EMA_WIN, adjust=False).mean())
