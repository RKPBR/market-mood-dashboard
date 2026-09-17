"""
Market Mood Dashboard — single-page Streamlit app
====================================================
Consolidates everything into ONE page, like the RSI2 dashboard:
  - Layer 1: Nifty50 market filter
  - Layer 2: Top-3 sector ranking (relative strength)
  - Layer 3: Regime per top sector (ADX14 + 200EMA slope)
  - Layer 4: Entry scan (1990 for Uptrend sectors, Bollinger for Sideways sectors)

FILES NEEDED IN THE SAME FOLDER/REPO:
  - nifty500_stocklist.csv   (symbol, yfinance_symbol, company_name, industry)
  - bollinger_tradeable.csv  (the 20 validated Sideways-regime stocks)

DEPLOY THE SAME WAY YOU DEPLOYED THE RSI2 DASHBOARD:
  1. Put this file + the 2 CSVs + a requirements.txt (streamlit, pandas, numpy, yfinance)
     into a GitHub repo (a new one, or a new folder in your existing rsi2-dashboard repo).
  2. Go to share.streamlit.io, point it at this file, deploy.
  3. Open the link -> click "Run Today's Scan" -> everything shows on one page.

NOTE ON SPEED: scanning ~500 stocks + building 18 sector indices takes a few minutes
(same as it did in Colab). MAX_STOCKS_PER_SECTOR below keeps it from being too slow —
raise it later once you're comfortable with the runtime.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Market Mood Dashboard", layout="wide")

# ============================== TUNABLE PARAMETERS ==============================
STOCK_LIST_CSV        = "nifty500_stocklist.csv"
TRADEABLE_BOLLINGER   = "bollinger_tradeable.csv"
NIFTY50_TICKER        = "^NSEI"
PERIOD                = "3y"

RS_LOOKBACK_DAYS      = 21
TOP_N_SECTORS         = 3
MAX_STOCKS_PER_SECTOR = 15        # kept lower than the Colab version for web-app speed

RSI_PERIOD            = 2
RSI_ENTRY_MAX         = 10
EMA_TREND_WINDOW      = 200
ADX_WINDOW            = 14
ATR_WINDOW            = 14
ATR_SL_MULT           = 2.0

BB_WINDOW             = 20
BB_NDEV               = 1.5
ADX_STRONG            = 25
ADX_SIDEWAYS          = 20
# ==================================================================================


@st.cache_data(ttl=3600, show_spinner=False)
def load_universe():
    return pd.read_csv(STOCK_LIST_CSV)


@st.cache_data(ttl=3600, show_spinner=False)
def load_bollinger_list():
    try:
        return set(pd.read_csv(TRADEABLE_BOLLINGER)["symbol"].tolist())
    except Exception:
        return set()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc(symbol, period=PERIOD):
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < EMA_TREND_WINDOW + 20:
            return None
        return df[["High", "Low", "Close"]]
    except Exception:
        return None


def compute_adx(df, window=ADX_WINDOW):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close, prev_high, prev_low = close.shift(1), high.shift(1), low.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move, down_move = high - prev_high, prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/14, adjust=False).mean()


def compute_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, window=ATR_WINDOW):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def classify_regime(adx, slope):
    if pd.isna(adx) or pd.isna(slope):
        return None
    if adx > ADX_STRONG and slope > 0:
        return "Strong Uptrend"
    if adx > ADX_STRONG and slope < 0:
        return "Strong Downtrend"
    if adx < ADX_SIDEWAYS:
        return "Sideways/Choppy"
    return "Transitional"


def build_sector_index(symbols):
    highs, lows, closes = [], [], []
    used = 0
    for sym in symbols:
        ohlc = fetch_ohlc(sym)
        if ohlc is None:
            continue
        base = ohlc["Close"].iloc[0]
        if base <= 0 or pd.isna(base):
            continue
        highs.append(ohlc["High"] / base * 100)
        lows.append(ohlc["Low"] / base * 100)
        closes.append(ohlc["Close"] / base * 100)
        used += 1
    if used < 3:
        return None, used
    sector_df = pd.DataFrame({
        "High": pd.concat(highs, axis=1).mean(axis=1),
        "Low": pd.concat(lows, axis=1).mean(axis=1),
        "Close": pd.concat(closes, axis=1).mean(axis=1),
    }).dropna()
    return sector_df, used


def layer1_market_filter():
    nifty = fetch_ohlc(NIFTY50_TICKER)
    if nifty is None:
        return True, None, None
    ema200 = nifty["Close"].ewm(span=EMA_TREND_WINDOW, adjust=False).mean()
    last_close, last_ema = nifty["Close"].iloc[-1], ema200.iloc[-1]
    return last_close > last_ema, round(last_close, 1), round(last_ema, 1)


def scan_uptrend_stock(symbol):
    df = fetch_ohlc(symbol)
    if df is None:
        return None
    close = df["Close"]
    ema200 = close.ewm(span=EMA_TREND_WINDOW, adjust=False).mean()
    rsi2 = compute_rsi(close)
    atr = compute_atr(df)
    if close.iloc[-1] > ema200.iloc[-1] and rsi2.iloc[-1] < RSI_ENTRY_MAX:
        entry = close.iloc[-1]
        return round(entry, 2), round(entry - ATR_SL_MULT * atr.iloc[-1], 2)
    return None


def scan_sideways_stock(symbol):
    df = fetch_ohlc(symbol)
    if df is None:
        return None
    close = df["Close"]
    mid = close.rolling(BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW).std()
    lower = mid - BB_NDEV * std
    atr = compute_atr(df)
    if close.iloc[-1] < lower.iloc[-1]:
        entry = close.iloc[-1]
        return round(entry, 2), round(entry - ATR_SL_MULT * atr.iloc[-1], 2)
    return None


LOG_FILE = "market_mood_log.csv"
LOG_COLUMNS = ["symbol", "sector", "regime", "strategy", "entry_date", "entry_price",
               "stop_loss", "status", "exit_date", "exit_price", "exit_reason", "pnl_pct", "win"]


def load_log():
    try:
        df = pd.read_csv(LOG_FILE, parse_dates=["entry_date", "exit_date"])
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[LOG_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(df):
    df.to_csv(LOG_FILE, index=False)


def check_exit_1990(symbol, entry_price, stop_loss, entry_date):
    df = fetch_ohlc(symbol)
    if df is None:
        return None
    df_after = df[df.index.date > pd.Timestamp(entry_date).date()]  # never same-day exit
    if df_after.empty:
        return None
    close, rsi2, sma5 = df["Close"], compute_rsi(df["Close"]), df["Close"].rolling(5).mean()
    for date in df_after.index:
        c = close.loc[date]
        if c <= stop_loss:
            return date, stop_loss, "SL hit"
        if rsi2.loc[date] > 70:
            return date, c, "RSI2>70"
        if c > sma5.loc[date]:
            return date, c, "Close>5SMA"
    return None


def check_exit_bollinger(symbol, entry_price, stop_loss, entry_date):
    df = fetch_ohlc(symbol)
    if df is None:
        return None
    df_after = df[df.index.date > pd.Timestamp(entry_date).date()]
    if df_after.empty:
        return None
    close, mid = df["Close"], df["Close"].rolling(BB_WINDOW).mean()
    for date in df_after.index:
        c = close.loc[date]
        if c <= stop_loss:
            return date, stop_loss, "SL hit"
        if c > mid.loc[date]:
            return date, c, "Close>Mid-band"
    return None


def update_open_trades(log_df):
    for idx, row in log_df[log_df["status"] == "OPEN"].iterrows():
        checker = check_exit_1990 if str(row["strategy"]).startswith("1990") else check_exit_bollinger
        result = checker(row["symbol"], row["entry_price"], row["stop_loss"], row["entry_date"])
        if result:
            exit_date, exit_price, reason = result
            pnl_pct = round((exit_price - row["entry_price"]) / row["entry_price"] * 100, 2)
            log_df.loc[idx, ["status", "exit_date", "exit_price", "exit_reason", "pnl_pct", "win"]] = \
                ["CLOSED", exit_date, round(exit_price, 2), reason, pnl_pct, pnl_pct > 0]
    return log_df


def append_new_entries(log_df, entries_df, today):
    open_symbols = set(log_df[log_df["status"] == "OPEN"]["symbol"])
    new_rows = []
    for _, e in entries_df.iterrows():
        if e["symbol"] in open_symbols:
            continue  # already holding this one — same discipline as trade_tracker.py
        new_rows.append({
            "symbol": e["symbol"], "sector": e["sector"], "regime": e["regime"], "strategy": e["strategy"],
            "entry_date": today, "entry_price": e["entry"], "stop_loss": e["stop_loss"], "status": "OPEN",
            "exit_date": None, "exit_price": None, "exit_reason": None, "pnl_pct": None, "win": None,
        })
    if new_rows:
        log_df = pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)
    return log_df


def run_full_scan():
    universe = load_universe()
    bollinger_ok = load_bollinger_list()
    sector_stocks = universe.groupby("industry")["yfinance_symbol"].apply(list).to_dict()

    progress = st.progress(0, text="Layer 1: checking Nifty50...")
    risk_on, nifty_close, nifty_ema = layer1_market_filter()

    progress.progress(10, text="Layer 2: ranking sectors by relative strength...")
    scores = []
    sectors_list = list(sector_stocks.items())
    for i, (sector, symbols) in enumerate(sectors_list):
        syms = symbols[:MAX_STOCKS_PER_SECTOR]
        sector_df, used = build_sector_index(syms)
        if sector_df is not None and len(sector_df) > RS_LOOKBACK_DAYS + 5:
            ret = (sector_df["Close"].iloc[-1] / sector_df["Close"].iloc[-RS_LOOKBACK_DAYS] - 1) * 100
            scores.append({"sector": sector, "return_pct": round(ret, 2), "df": sector_df})
        progress.progress(10 + int(50 * (i + 1) / len(sectors_list)),
                           text=f"Layer 2: scanning sector {i+1}/{len(sectors_list)}...")

    scores.sort(key=lambda x: x["return_pct"], reverse=True)
    top_sectors = scores[:TOP_N_SECTORS]

    progress.progress(65, text="Layer 3: detecting regime for top sectors...")
    regime_rows = []
    for s in top_sectors:
        df = s["df"].copy()
        df["EMA200"] = df["Close"].ewm(span=EMA_TREND_WINDOW, adjust=False).mean()
        df["ADX14"] = compute_adx(df)
        df["EMA_SLOPE"] = df["EMA200"] - df["EMA200"].shift(10)
        latest = df.iloc[-1]
        regime = classify_regime(latest["ADX14"], latest["EMA_SLOPE"])
        s["regime"] = regime
        s["adx"] = round(latest["ADX14"], 2)
        s["slope"] = round(latest["EMA_SLOPE"], 3)
        regime_rows.append({"sector": s["sector"], "relative_strength_%": s["return_pct"],
                             "adx14": s["adx"], "ema_slope": s["slope"], "regime": regime})

    progress.progress(75, text="Layer 4: scanning for entries...")
    entries = []
    for s in top_sectors:
        sector, regime = s["sector"], s["regime"]
        if regime == "Strong Uptrend":
            for sym in sector_stocks[sector]:
                hit = scan_uptrend_stock(sym)
                if hit:
                    entries.append({"symbol": sym, "sector": sector, "regime": regime,
                                     "strategy": "1990 (RSI2+200EMA)", "entry": hit[0], "stop_loss": hit[1]})
        elif regime == "Sideways/Choppy":
            for sym in [s for s in sector_stocks[sector] if s in bollinger_ok]:
                hit = scan_sideways_stock(sym)
                if hit:
                    entries.append({"symbol": sym, "sector": sector, "regime": regime,
                                     "strategy": "Bollinger mean-reversion", "entry": hit[0], "stop_loss": hit[1]})

    progress.progress(100, text="Done!")
    progress.empty()

    return {
        "risk_on": risk_on, "nifty_close": nifty_close, "nifty_ema": nifty_ema,
        "sector_ranking": pd.DataFrame([{"sector": s["sector"], "return_pct": s["return_pct"]} for s in scores]),
        "regime_table": pd.DataFrame(regime_rows),
        "entries": pd.DataFrame(entries),
    }


# ================================== PAGE LAYOUT ==================================
st.title("📊 Market Mood Dashboard")
st.caption("Regime-adaptive scanner — Layer 1 (market filter) → Layer 2 (sector rank) → "
           "Layer 3 (regime) → Layer 4 (1990 / Bollinger entries). Runs in PARALLEL with 1990 — "
           "paper-track results before using real money.")

if st.button("🔍 Run Today's Scan", type="primary"):
    log_df = load_log()
    log_df = update_open_trades(log_df)          # check existing OPEN positions for exits first
    result = run_full_scan()
    today = pd.Timestamp.today().normalize()
    log_df = append_new_entries(log_df, result["entries"], today)   # log today's fresh entries
    save_log(log_df)
    st.session_state["result"] = result
    st.session_state["log"] = log_df

if "result" in st.session_state:
    r = st.session_state["result"]

    col1, col2, col3 = st.columns(3)
    with col1:
        status = "🟢 RISK-ON" if r["risk_on"] else "🔴 RISK-OFF"
        st.metric("Layer 1: Nifty50 vs 200EMA", status)
        if r["nifty_close"]:
            st.caption(f"Close: {r['nifty_close']} | 200EMA: {r['nifty_ema']}")
    with col2:
        st.metric("Layer 2: Top Sectors Today", ", ".join(r["regime_table"]["sector"].tolist()) if not r["regime_table"].empty else "—")
    with col3:
        st.metric("Layer 4: Entries Found", len(r["entries"]))

    if not r["risk_on"]:
        st.warning("Nifty50 is below its 200EMA — Master Plan says lean toward cash / reduce size today.")

    st.subheader("Sector Relative-Strength Ranking (all sectors)")
    st.dataframe(r["sector_ranking"], use_container_width=True, hide_index=True)

    st.subheader("Top 3 Sectors — Today's Regime")
    st.dataframe(r["regime_table"], use_container_width=True, hide_index=True)

    st.subheader("✅ TRADEABLE TODAY")
    if r["entries"].empty:
        st.info("No entries today across the top sectors.")
    else:
        st.dataframe(r["entries"], use_container_width=True, hide_index=True)
        st.download_button("Download as CSV", r["entries"].to_csv(index=False), "market_mood_today.csv")

    st.divider()
    st.subheader("📒 Paper Trade Log (all-time, this is how we judge performance)")
    log_df = st.session_state.get("log", load_log())
    closed = log_df[log_df["status"] == "CLOSED"]
    open_pos = log_df[log_df["status"] == "OPEN"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total logged", len(log_df))
    c2.metric("Closed trades", len(closed))
    c3.metric("Overall win rate", f"{closed['win'].mean()*100:.1f}%" if len(closed) else "—")

    if not closed.empty:
        st.write("**By strategy (this is the number that decides Step 5 — compare to 1990's own results):**")
        summary = closed.groupby("strategy").agg(
            trades=("win", "count"), win_rate=("win", "mean"), avg_pnl_pct=("pnl_pct", "mean")
        ).round(3)
        st.dataframe(summary, use_container_width=True)

    st.write(f"**Open positions ({len(open_pos)}):**")
    st.dataframe(open_pos.drop(columns=["exit_date", "exit_price", "exit_reason", "pnl_pct", "win"]),
                 use_container_width=True, hide_index=True)

    st.write(f"**Closed trades ({len(closed)}):**")
    st.dataframe(closed, use_container_width=True, hide_index=True)

    st.download_button("Download full log CSV", log_df.to_csv(index=False), "market_mood_log.csv")
    st.caption("⚠️ Same caveat as trade_log.csv on the RSI2 dashboard: this file's persistence on "
               "Streamlit Cloud's free tier isn't guaranteed across app restarts/redeploys. "
               "Download this CSV as a backup every few days until Google Sheets logging is set up.")
else:
    st.info("Click **Run Today's Scan** to check today's market mood.")
