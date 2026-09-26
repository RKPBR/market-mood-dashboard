"""
Market Mood dashboard — Frozen rules v1 (26 Sep 2026)

Every evening after 4:00 pm IST, "Run today's scan":
  1. downloads 5 years of daily prices for the current Nifty 500 + Nifty 50 (yfinance),
  2. runs the SAME frozen engine that passed the LOCKED test (mm_frozen.py),
  3. updates the system log (Google Sheet "market-mood-bot", tab "system_log_v1"),
  4. shows tomorrow's orders: BUY (with share count and stop-loss) and SELL.

Files needed in the repo: market_mood_app.py, mm_frozen.py, nifty500_current.csv,
requirements.txt, .streamlit/config.toml. Google credentials stay in Streamlit Secrets.
"""
import datetime as dt
import html
import logging
import os
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import mm_frozen as F

warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty-sector means on early dates

st.set_page_config(page_title="Market Mood", page_icon="🟣", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ settings
OWNER = "Kaushik J. Tanna"
EMAIL = "kj.tanna@gmail.com"
IST = ZoneInfo("Asia/Kolkata")
LIST_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nifty500_current.csv")
SPREADSHEET_NAME = "market-mood-bot"
LOG_TAB = "system_log_v1"
HISTORY_YEARS = 5
MARKET_OPEN, MARKET_CLOSE, SCAN_AFTER = dt.time(9, 15), dt.time(15, 30), dt.time(16, 0)
GTT_LIMIT_BELOW = 0.05          # GTT limit price 5% below the trigger, so gap-downs still fill

# NSE equity holidays 2026 (official circular). Add the 2027 list when NSE publishes it in December.
NSE_HOLIDAYS = {
    dt.date(2026, 1, 15): "Maharashtra municipal election", dt.date(2026, 1, 26): "Republic Day",
    dt.date(2026, 3, 3): "Holi", dt.date(2026, 3, 26): "Shri Ram Navami",
    dt.date(2026, 3, 31): "Shri Mahavir Jayanti", dt.date(2026, 4, 3): "Good Friday",
    dt.date(2026, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti", dt.date(2026, 5, 1): "Maharashtra Day",
    dt.date(2026, 5, 28): "Bakri Id", dt.date(2026, 6, 26): "Muharram",
    dt.date(2026, 9, 14): "Ganesh Chaturthi", dt.date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    dt.date(2026, 10, 20): "Dussehra", dt.date(2026, 11, 10): "Diwali Balipratipada",
    dt.date(2026, 11, 24): "Guru Nanak Jayanti", dt.date(2026, 12, 25): "Christmas",
}

LOG_COLS = ["symbol", "sector", "signal_date", "qty", "stop_loss", "status", "entry_date",
            "entry_price", "exit_date", "exit_price", "exit_reason", "pnl_pct", "pnl_rs"]

# test hooks (never set on Streamlit Cloud)
TEST_PARQUET = os.environ.get("MM_TEST_PARQUET")
TEST_NOW = os.environ.get("MM_TEST_NOW")
TEST_LOG = os.environ.get("MM_TEST_LOG")


def now_ist():
    return pd.Timestamp(TEST_NOW, tz=IST) if TEST_NOW else pd.Timestamp.now(tz=IST)


def is_trading_day(d):
    return d.weekday() < 5 and d not in NSE_HOLIDAYS


def next_trading_day(d):
    d = d + dt.timedelta(days=1)
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def indian(x, dec=0):
    """1,50,000 style grouping."""
    neg = x < 0
    s = f"{abs(x):.{dec}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups) + "," + tail
    return ("-" if neg else "") + whole + ("." + frac if frac else "")


def inr(x, dec=0):
    return ("-₹" if x < 0 else "₹") + indian(abs(x), dec)


def esc(s):
    return html.escape(str(s))


def short(sym):
    return str(sym).replace(".NS", "")


REASON_TEXT = {"RSI2>70": "RSI(2) above 70", "Close>SMA5": "close above 5-day average",
               "MaxHold": "10-day limit reached", "SL": "stop-loss", "SL_gap": "stop-loss, gap down"}


def pretty_date(x):
    try:
        return pd.Timestamp(x).strftime("%d %b")
    except (TypeError, ValueError):
        return str(x)


# ------------------------------------------------------------------ style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&family=Noto+Sans+Gujarati:wght@400;600&display=swap');
:root{--ink:#0E0E12;--card:#18181E;--line:#26262F;--lav:#C9B8FF;--butter:#F3E28C;--sky:#9FD4FF;
--mint:#8BE3B4;--coral:#FF9C8A;--text:#F5F4F8;--muted:#8E8C9B;}
html,body,.stApp,[class*="css"]{font-family:'Manrope','Noto Sans Gujarati',sans-serif;}
.stApp{background:linear-gradient(135deg,#DCD3FF 0%,#C9D8FF 55%,#E4D7FF 100%);}
[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container,[data-testid="stMainBlockContainer"]{background:var(--ink);border-radius:30px;
  max-width:1320px;margin:22px auto 30px auto;padding:26px 30px 30px 30px !important;
  box-shadow:0 30px 80px rgba(40,20,90,.35);}
@media (max-width:760px){.block-container,[data-testid="stMainBlockContainer"]{margin:0;border-radius:0;padding:16px 14px !important;}}
.mm-brand{font-size:30px;font-weight:600;letter-spacing:-.02em;color:var(--text);line-height:1;padding-top:6px;}
.mm-brand small{display:block;font-size:13px;font-weight:400;color:var(--muted);letter-spacing:0;margin-top:6px;}
.mm-profile{display:flex;align-items:center;gap:12px;justify-content:flex-end;background:var(--card);
  border:1px solid var(--line);border-radius:999px;padding:7px 18px 7px 8px;width:fit-content;margin-left:auto;}
.mm-avatar{width:40px;height:40px;border-radius:50%;background:var(--lav);color:var(--ink);display:flex;
  align-items:center;justify-content:center;font-weight:700;font-size:15px;}
.mm-pname{color:var(--text);font-weight:600;font-size:15px;line-height:1.1;}
.mm-pmail{color:var(--muted);font-size:12px;}
.mm-pmail a{color:var(--muted);text-decoration:none;}
.stButton>button{background:var(--lav)!important;color:var(--ink)!important;border:none!important;
  border-radius:999px!important;font-weight:700!important;padding:.62rem 1.2rem!important;}
.stButton>button:hover{filter:brightness(1.06);}
.stButton>button:focus-visible{outline:3px solid var(--butter)!important;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--card);border:1px solid var(--line);border-radius:999px;
  padding:5px;width:fit-content;margin:6px 0 14px 0;}
.stTabs [data-baseweb="tab"]{border-radius:999px;padding:8px 20px;color:var(--muted);background:transparent;height:auto;}
.stTabs [aria-selected="true"]{background:var(--text)!important;color:var(--ink)!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}
.mm-card{background:var(--card);border:1px solid var(--line);border-radius:24px;padding:20px 22px;min-height:100%;}
.mm-card.tall{min-height:410px;}
.mm-card.mid{min-height:360px;}
.mm-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}
.mm-t{color:var(--text);font-weight:600;font-size:16px;}
.mm-chip{border:1px solid var(--line);color:var(--muted);border-radius:999px;padding:4px 12px;font-size:12px;}
.mm-big{color:var(--text);font-weight:300;font-size:46px;letter-spacing:-.03em;line-height:1;}
.mm-sub{color:var(--muted);font-size:13px;margin-top:6px;}
.b-up,.b-dn,.b-lav,.b-but{display:inline-block;border-radius:9px;padding:3px 9px;font-size:12px;font-weight:700;color:var(--ink);}
.b-up{background:var(--mint);} .b-dn{background:var(--coral);} .b-lav{background:var(--lav);} .b-but{background:var(--butter);}
.mm-verdict{margin-top:14px;padding:12px 14px;border-radius:16px;background:#202028;color:var(--text);font-size:14px;}
.mm-verdict b{color:var(--lav);}
.mm-row{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:16px;background:#202028;margin-bottom:8px;}
.mm-row .grow{flex:1;min-width:0;}
.mm-row .nm{color:var(--text);font-weight:600;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.mm-row .dt{color:var(--muted);font-size:12px;}
.mm-row .val{background:#2A2A33;color:var(--text);border-radius:12px;padding:8px 10px;font-size:13px;font-weight:600;text-align:right;white-space:nowrap;}
.mm-empty{color:var(--muted);font-size:14px;padding:18px 4px;line-height:1.5;}
.mm-slots{display:flex;align-items:flex-end;gap:10px;height:230px;margin-top:8px;}
.mm-slot{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;}
.mm-slot .bar{width:100%;border-radius:14px;}
.mm-slot .lab{color:var(--muted);font-size:11px;margin-top:8px;white-space:nowrap;}
.mm-slot .top{color:var(--text);font-size:11px;font-weight:700;margin-bottom:6px;background:#2A2A33;border-radius:8px;padding:2px 6px;}
.bar.open{background:var(--lav);} .bar.sell{background:var(--butter);}
.bar.buy{background:repeating-linear-gradient(135deg,var(--butter) 0 6px,#3a3726 6px 11px);}
.bar.free{background:repeating-linear-gradient(135deg,#2c2c35 0 6px,#1d1d23 6px 11px);height:34%;}
.mm-legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:12px;color:var(--muted);font-size:12px;}
.mm-legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px;}
.mm-bubbles{position:relative;height:250px;}
.mm-bub{position:absolute;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;
  color:var(--ink);text-align:center;padding:10px;}
.mm-bub b{font-weight:600;letter-spacing:-.02em;}
.mm-bub span{font-size:11px;line-height:1.2;}
.mm-foot{margin-top:26px;padding:18px 22px;border-radius:20px;border:1px solid var(--line);color:var(--muted);font-size:12.5px;line-height:1.6;}
.mm-foot b{color:var(--text);}
.mm-foot a{color:var(--lav);}
.mm-note{color:var(--muted);font-size:12px;margin-top:10px;}
div[data-testid="stAlert"]{border-radius:16px;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
def _extract(data, t):
    if data is None or len(data) == 0:
        return None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if t in data.columns.get_level_values(0):
                sub = data.xs(t, axis=1, level=0)
            elif t in data.columns.get_level_values(1):
                sub = data.xs(t, axis=1, level=1)
            else:
                return None
        else:
            sub = data
        sub = sub.rename(columns={"Adj Close": "AdjClose"})
        need = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
        if any(c not in sub.columns for c in need):
            return None
        sub = sub[need].copy()
        sub = sub[sub["Close"].notna()]
        if len(sub) == 0:
            return None
        idx = pd.DatetimeIndex(pd.to_datetime(sub.index))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        sub.index = idx.normalize()
        sub = sub[~sub.index.duplicated(keep="last")]
        sub.index.name = "Date"
        sub = sub.reset_index()
        sub.insert(1, "Ticker", t)
        return sub
    except Exception:
        return None


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_prices(tickers, start, end, session_tag):
    """session_tag separates before-close and after-close downloads of the same day."""
    if TEST_PARQUET:
        w = pd.read_parquet(TEST_PARQUET)
        return w[(w["Date"] >= start) & (w["Date"] < end) & (w["Ticker"].isin(tickers))].copy()
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    def fetch(batch):
        for attempt in range(3):
            try:
                return yf.download(list(batch), start=start, end=end, interval="1d", auto_adjust=False,
                                   actions=False, group_by="ticker", threads=True, progress=False)
            except Exception:
                time.sleep(5 * (attempt + 1))
        return None

    tickers = list(tickers)
    frames, got = [], set()
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        data = fetch(batch)
        for t in batch:
            sub = _extract(data, t)
            if sub is not None:
                frames.append(sub)
                got.add(t)
        time.sleep(0.5)
    for t in [t for t in tickers if t not in got][:40]:
        sub = _extract(fetch([t]), t)
        if sub is not None:
            frames.append(sub)
            got.add(t)
    if not frames:
        return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"])
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_universe():
    cur = pd.read_csv(LIST_CSV)
    return cur[~cur["symbol"].astype(str).str.upper().str.startswith("DUMMY")].reset_index(drop=True)


# ------------------------------------------------------------------ system log (Google Sheet)
def _clean_log(df):
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[LOG_COLS].copy().astype(object)
    return df.where(df.notna(), "")


@st.cache_resource(show_spinner=False)
def _worksheet():
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scopes)
    sh = gspread.authorize(creds).open(SPREADSHEET_NAME)
    try:
        return sh.worksheet(LOG_TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=LOG_TAB, rows=2000, cols=len(LOG_COLS))
        ws.update([LOG_COLS])
        return ws


def load_log():
    try:
        if TEST_LOG:
            df = pd.read_csv(TEST_LOG, dtype=str) if os.path.exists(TEST_LOG) else pd.DataFrame(columns=LOG_COLS)
        else:
            recs = _worksheet().get_all_records()
            df = pd.DataFrame(recs) if recs else pd.DataFrame(columns=LOG_COLS)
        return _clean_log(df), None
    except Exception as e:
        return pd.DataFrame(columns=LOG_COLS), f"The trade log could not be read ({e})."


def save_log(df):
    try:
        out = _clean_log(df.copy())
        if TEST_LOG:
            out.to_csv(TEST_LOG, index=False)
            return None
        rows = [[(x.item() if hasattr(x, "item") else x) for x in r] for r in out.values.tolist()]
        ws = _worksheet()
        ws.clear()
        ws.update([LOG_COLS] + rows)
        return None
    except Exception as e:
        return f"The trade log could not be saved ({e})."


# ------------------------------------------------------------------ frozen-rule position tracking
def _num(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def update_positions(log, D, L):
    """Replays every non-closed row from its signal date with the frozen exit rules (raw rupee prices)."""
    A, cal = D["A"], D["cal"]
    jmap = {t: i for i, t in enumerate(D["tickers"])}
    with np.errstate(all="ignore"):
        rf = A["rawC"] / A["C"]
    rows, stopped_today = [], []
    for r in log.to_dict("records"):
        if r["status"] in ("CLOSED", "SKIPPED"):
            rows.append(r)
            continue
        j = jmap.get(r["symbol"])
        sdt = pd.Timestamp(r["signal_date"]) if r["signal_date"] else None
        sd = cal.searchsorted(sdt) if sdt is not None else len(cal)
        if j is None or sd >= len(cal) or cal[sd] != sdt:
            r["status"] = r["status"] or "PENDING"
            rows.append(r)
            continue
        stop, qty = _num(r["stop_loss"]), int(_num(r["qty"], 0))
        r.update(entry_date="", entry_price="", exit_date="", exit_price="", exit_reason="", pnl_pct="", pnl_rs="")
        if sd >= L:
            r["status"] = "PENDING"
            rows.append(r)
            continue
        ed = sd + 1
        o_in = A["O"][ed, j] * rf[ed, j]
        if not np.isfinite(o_in):
            r.update(status="SKIPPED", exit_reason="no trading on entry day")
            rows.append(r)
            continue
        if o_in <= stop:
            r.update(status="SKIPPED", exit_reason="opened below stop-loss")
            rows.append(r)
            continue
        r.update(status="OPEN", entry_date=cal[ed].strftime("%Y-%m-%d"), entry_price=round(o_in, 2))
        why, exit_px, exit_di, reason = None, None, None, None
        for di in range(ed, L + 1):
            if not np.isfinite(A["C"][di, j]):
                continue
            o, lo = A["O"][di, j] * rf[di, j], A["L"][di, j] * rf[di, j]
            if why:
                exit_px, exit_di, reason = o, di, why
                break
            if lo <= stop:
                gap = di > ed and o <= stop
                exit_px, exit_di, reason = (o if gap else stop), di, ("SL_gap" if gap else "SL")
                break
            if A["rsi"][di, j] > F.RSI_EXIT:
                why = "RSI2>70"
            elif A["C"][di, j] > A["sma5"][di, j]:
                why = "Close>SMA5"
            elif di - ed + 1 >= F.MAX_HOLD:
                why = "MaxHold"
        if exit_px is not None:
            gross = qty * (exit_px - o_in)
            net = gross - F.SIDE_COST * qty * (o_in + exit_px) - F.DP
            r.update(status="CLOSED", exit_date=cal[exit_di].strftime("%Y-%m-%d"), exit_price=round(exit_px, 2),
                     exit_reason=reason, pnl_pct=round(100 * (exit_px / o_in - 1), 2), pnl_rs=round(net, 0))
            if exit_di == L and reason.startswith("SL"):
                stopped_today.append(r)
        else:
            last = A["C"][L, j] * rf[L, j]
            r.update(status="EXIT_PENDING" if why else "OPEN", exit_reason=why or "",
                     pnl_pct=round(100 * (last / o_in - 1), 2),
                     pnl_rs=round(qty * (last - o_in), 0))
        rows.append(r)
    return pd.DataFrame(rows, columns=LOG_COLS), stopped_today


def pick_orders(log, D, L):
    """Frozen priority: sector rank, then lowest RSI(2); 6 slots of Rs 25,000."""
    A = D["A"]
    held = log[log["status"].isin(["OPEN", "EXIT_PENDING", "PENDING"])]
    held_syms = set(held["symbol"])
    free = max(0, F.MAX_SLOTS - len(held))
    cands = [int(j) for j in np.flatnonzero(D["frozen"][L]) if D["tickers"][j] not in held_syms]
    cands.sort(key=lambda j: (D["rank"][L, D["sec_idx"][j]], A["rsi"][L, j]))
    buys, skipped = [], []
    for j in cands:
        raw = A["rawC"][L, j]
        item = dict(symbol=D["tickers"][j], sector=D["sectors"][D["sec_idx"][j]],
                    rank=int(D["rank"][L, D["sec_idx"][j]]), rsi=float(A["rsi"][L, j]), close=float(raw))
        qty = int(F.SLOT // raw)
        if qty < 1:
            skipped.append(dict(item, why="share price above ₹25,000"))
            continue
        if len(buys) >= free:
            skipped.append(dict(item, why="no free slot"))
            continue
        stop = float(D["stop"][L, j] * raw / A["C"][L, j])
        buys.append(dict(item, qty=qty, stop=round(stop, 2), value=qty * raw))
    return buys, skipped, free


# ------------------------------------------------------------------ scan
def regime_name(adx, slope):
    if not np.isfinite(adx) or not np.isfinite(slope):
        return "No data"
    if adx > F.ADX_STRONG and slope > 0:
        return "Strong uptrend"
    if adx > F.ADX_STRONG and slope < 0:
        return "Strong downtrend"
    if adx < 20:
        return "Sideways"
    return "Transitional"


def run_scan():
    now = now_ist()
    today = now.date()
    in_session = is_trading_day(today) and MARKET_OPEN <= now.time() < SCAN_AFTER
    after_close = is_trading_day(today) and now.time() >= SCAN_AFTER
    cur = load_universe()
    log, log_err = load_log()
    extra = [s for s in log.loc[log["status"].isin(["OPEN", "EXIT_PENDING", "PENDING"]), "symbol"]
             if s and s not in set(cur["yfinance_symbol"])]
    tickers = tuple(["^NSEI"] + list(cur["yfinance_symbol"]) + sorted(set(extra)))
    end = today + dt.timedelta(days=1)
    start = end - pd.DateOffset(years=HISTORY_YEARS)
    w = load_prices(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                    "after" if now.time() >= SCAN_AFTER else "before")
    if not after_close:
        w = w[w["Date"] < pd.Timestamp(today)]          # never use an unfinished day
    if "^NSEI" not in set(w["Ticker"]):
        return dict(error="Nifty 50 prices could not be downloaded, so the market filter is treated as "
                          "RISK-OFF. Try the scan again in a few minutes.")
    D = F.build(w, cur, str(w["Date"].min().date()))
    cal = D["cal"]
    L = len(cal) - 1
    as_of = cal[L].date()
    mode = "orders"
    if in_session:
        mode = "market_hours"
    elif after_close and as_of < today:
        mode = "stale"

    if mode == "orders":   # tonight's picks are rebuilt from scratch, so a second run changes nothing
        log = log[~((log["status"] == "PENDING") & (log["signal_date"] == as_of.strftime("%Y-%m-%d")))]
    log, stopped_today = update_positions(log, D, L)
    buys, skipped = [], []
    free = max(0, F.MAX_SLOTS - int(log["status"].isin(["OPEN", "EXIT_PENDING", "PENDING"]).sum()))
    if mode == "orders":
        buys, skipped, free = pick_orders(log, D, L)
        new = pd.DataFrame([dict(symbol=b["symbol"], sector=b["sector"], signal_date=as_of.strftime("%Y-%m-%d"),
                                 qty=b["qty"], stop_loss=b["stop"], status="PENDING") for b in buys],
                           columns=LOG_COLS)
        log = _clean_log(pd.concat([log, new], ignore_index=True)) if len(new) else log
    save_err = save_log(log)

    n = D["nifty"]
    ema = D["nifty_ema"]
    sec_rows = []
    for si, s in enumerate(D["sectors"]):
        rk = int(D["rank"][L, si])
        sec_rows.append(dict(Rank=rk if rk < 99 else None, Sector=s,
                             **{"21-day move %": round(100 * D["rs"][L, si], 2) if np.isfinite(D["rs"][L, si]) else None},
                             ADX=round(float(D["adx"][L, si]), 1) if np.isfinite(D["adx"][L, si]) else None,
                             Trend=regime_name(D["adx"][L, si], D["slope"][L, si]),
                             Eligible="Yes" if (rk <= F.TOP_N and D["up"][L, si]) else ""))
    sectors = pd.DataFrame(sec_rows).sort_values("Rank", na_position="last").reset_index(drop=True)
    A = D["A"]
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(A["C"][L]) & (A["bars"][L] >= F.MIN_BARS)
        above = ok & (A["C"][L] > A["ema"][L])
    order = np.argsort(D["sec_idx"], kind="stable")
    dots = [(bool(above[j]) if ok[j] else None) for j in order]
    sells = log[log["status"] == "EXIT_PENDING"].to_dict("records")
    return dict(
        error=None, mode=mode, as_of=as_of, next_session=next_trading_day(as_of),
        nifty=dict(close=float(n.iloc[-1]), prev=float(n.iloc[-2]), ema=float(ema.iloc[-1]),
                   risk_on=bool(D["risk_on"][L]), ret=float(D["nifty_ret1"][L]),
                   dates=[d.strftime("%d %b") for d in n.index[-120:]],
                   series=n.iloc[-120:].round(2).tolist(), ema_series=ema.iloc[-120:].round(2).tolist()),
        sectors=sectors, dots=dots, breadth=(int(above.sum()), int(ok.sum())),
        buys=buys, skipped=skipped, free=free, sells=sells, stopped_today=stopped_today,
        log=log, log_err=log_err, save_err=save_err,
        n_signals=int(D["frozen"][L].sum()), n_stocks=int(np.isfinite(A["C"][L]).sum()), n_total=len(D["tickers"]),
    )


# ------------------------------------------------------------------ pieces of UI
def svg_nifty(series, ema_series, dates):
    w, h, pad = 620, 190, 6
    vals = [v for v in series + ema_series if v == v]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1

    def pts(arr):
        n = len(arr)
        return [(pad + i * (w - 2 * pad) / max(n - 1, 1), pad + (hi - v) * (h - 2 * pad) / rng) for i, v in enumerate(arr)]

    p1, p2 = pts(series), pts(ema_series)
    line = lambda p: "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p)
    area = line(p1) + f" L{p1[-1][0]:.1f},{h} L{p1[0][0]:.1f},{h} Z"
    lx, ly = p1[-1]
    ticks = "".join(f'<text x="{pad + i * (w - 2 * pad) / 4:.0f}" y="{h + 16}" fill="#8E8C9B" font-size="11" '
                    f'text-anchor="middle">{esc(dates[min(len(dates) - 1, round(i * (len(dates) - 1) / 4))])}</text>'
                    for i in range(5))
    return f"""<svg viewBox="0 0 {w} {h + 22}" width="100%" role="img" aria-label="Nifty 50 and its 200-day average">
<defs><pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="1.3" height="4" fill="#C9B8FF" opacity=".28"/></pattern></defs>
<path d="{area}" fill="url(#hatch)"/><path d="{line(p2)}" fill="none" stroke="#F3E28C" stroke-width="2"/>
<path d="{line(p1)}" fill="none" stroke="#C9B8FF" stroke-width="2.4"/>
<line x1="{lx:.1f}" y1="0" x2="{lx:.1f}" y2="{h}" stroke="#8E8C9B" stroke-dasharray="3 4"/>
<circle cx="{lx:.1f}" cy="{ly:.1f}" r="6" fill="#F5F4F8" stroke="#C9B8FF" stroke-width="3"/>{ticks}</svg>"""


def card_nifty(res):
    if not res:
        return ('<div class="mm-card tall"><div class="mm-h"><span class="mm-t">Nifty 50</span></div>'
                '<div class="mm-empty">Run today\'s scan to load the market. After 4:00 pm IST it also '
                'prepares tomorrow\'s orders.</div></div>')
    nf = res["nifty"]
    badge = "b-up" if nf["ret"] >= 0 else "b-dn"
    arrow = "↗" if nf["ret"] >= 0 else "↘"
    risk = '<span class="b-up">Risk-on</span>' if nf["risk_on"] else '<span class="b-dn">Risk-off</span>'
    dip = nf["ret"] <= F.NIFTY_DIP
    if not nf["risk_on"]:
        verdict = "<b>No buys.</b> Nifty is below its 200-day average (risk-off)."
    elif not dip:
        verdict = f"<b>No new buys.</b> Nifty moved {nf['ret'] * 100:+.2f}%; buys need a fall of 0.5% or more."
    else:
        k = res["n_signals"]
        verdict = (f"<b>Buy day.</b> Nifty fell {abs(nf['ret']) * 100:.2f}% in a risk-on market; "
                   f"{k} {'stock' if k == 1 else 'stocks'} qualified.")
    return f"""<div class="mm-card tall">
<div class="mm-h"><span class="mm-t">Nifty 50</span><span class="mm-chip">Close {res['as_of'].strftime('%d %b %Y')}</span></div>
<div style="display:flex;gap:26px;align-items:flex-end;flex-wrap:wrap">
 <div><div class="mm-big">{indian(nf['close'], 2)}</div>
  <div class="mm-sub"><span class="{badge}">{arrow} {nf['ret'] * 100:+.2f}%</span>&nbsp; today</div></div>
 <div><div class="mm-big" style="font-size:30px">{indian(nf['ema'], 0)}</div>
  <div class="mm-sub">200-day average &nbsp;{risk}</div></div>
</div>
<div style="margin-top:10px">{svg_nifty(nf['series'], nf['ema_series'], nf['dates'])}</div>
<div class="mm-verdict">{verdict}</div></div>"""


def calendar_html(now):
    today = now.date()
    first = today.replace(day=1)
    nxt = (first + dt.timedelta(days=32)).replace(day=1)
    days = (nxt - first).days
    cells = ['<div class="wd">' + d + "</div>" for d in ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]]
    cells += ['<div class="d blank"></div>'] * first.weekday()
    for k in range(days):
        d = first + dt.timedelta(days=k)
        cls = ["d"]
        tip = ""
        if d.weekday() >= 5:
            cls.append("we")
        if d in NSE_HOLIDAYS:
            cls.append("hol")
            tip = f' title="{esc(NSE_HOLIDAYS[d])}"'
        if d == today:
            cls.append("today")
        dot = '<i></i>' if d in NSE_HOLIDAYS else ""
        cells.append(f'<div class="{" ".join(cls)}"{tip}>{d.day}{dot}</div>')
    upcoming = sorted(d for d in NSE_HOLIDAYS if d >= today)
    if upcoming:
        h = upcoming[0]
        n_days = (h - today).days
        when = "today" if n_days == 0 else ("tomorrow" if n_days == 1 else f"in {n_days} days")
        nxt_html = f'<div class="big">{h.strftime("%d %b")}</div><div class="sub">{esc(NSE_HOLIDAYS[h])}, {when}</div>'
    else:
        nxt_html = '<div class="big">—</div><div class="sub">Add the 2027 NSE holiday list</div>'
    hol_js = ",".join(f'"{d.isoformat()}"' for d in NSE_HOLIDAYS)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
body{{margin:0;background:transparent;font-family:'Manrope',sans-serif;color:#F5F4F8;}}
.card{{background:#18181E;border:1px solid #26262F;border-radius:24px;padding:18px 18px 16px 18px;height:372px;box-sizing:border-box;}}
.top{{display:flex;justify-content:space-between;align-items:center;}}
.month{{border:1px solid #26262F;border-radius:999px;padding:6px 14px;font-size:13px;}}
.clock{{font-size:22px;font-weight:300;letter-spacing:-.02em;}}
.clock small{{font-size:11px;color:#8E8C9B;margin-left:4px;}}
.state{{font-size:12px;margin:10px 0 8px 2px;color:#8E8C9B;}}
.state b{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:#8E8C9B;}}
.state.open b{{background:#8BE3B4;}}
.grid{{display:grid;grid-template-columns:repeat(7,1fr);gap:5px;}}
.wd{{font-size:10px;color:#8E8C9B;text-align:center;padding-bottom:2px;}}
.d{{position:relative;background:#222229;border-radius:10px;height:27px;display:flex;align-items:center;
   justify-content:center;font-size:12px;color:#C9C7D3;}}
.d.blank{{background:transparent;}} .d.we{{color:#5E5C6A;background:#1C1C22;}}
.d.hol{{background:#C9B8FF;color:#0E0E12;font-weight:700;}}
.d i{{position:absolute;bottom:3px;width:4px;height:4px;border-radius:50%;background:#0E0E12;}}
.d.today{{outline:2px solid #F5F4F8;outline-offset:-2px;font-weight:700;}}
.next{{margin-top:12px;background:#222229;border-radius:16px;padding:10px 14px;display:flex;align-items:baseline;gap:12px;}}
.next .big{{font-size:28px;font-weight:300;letter-spacing:-.03em;}}
.next .sub{{font-size:12px;color:#8E8C9B;}}
.next .lab{{font-size:11px;color:#8E8C9B;}}
</style></head><body><div class="card">
<div class="top"><span class="month">{now.strftime('%B %Y')}</span><span class="clock" id="clk">--:--<small>IST</small></span></div>
<div class="state" id="st"><b></b>…</div>
<div class="grid">{''.join(cells)}</div>
<div class="next"><div><div class="lab">Next NSE holiday</div>{nxt_html}</div></div>
</div>
<script>
const HOL=[{hol_js}];
function tick(){{
 const now=new Date();
 const f=new Intl.DateTimeFormat('en-GB',{{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false,weekday:'short',year:'numeric',month:'2-digit',day:'2-digit'}});
 const p={{}}; f.formatToParts(now).forEach(x=>p[x.type]=x.value);
 const hh=+p.hour, mm=+p.minute, iso=p.year+'-'+p.month+'-'+p.day;
 const h12=((hh%12)||12), ap=hh<12?'AM':'PM';
 document.getElementById('clk').innerHTML=h12+':'+p.minute+':'+p.second+' '+ap+'<small>IST</small>';
 const wk=!['Sat','Sun'].includes(p.weekday), hol=HOL.includes(iso), t=hh*60+mm;
 const open=wk&&!hol&&t>=555&&t<930;
 const el=document.getElementById('st');
 el.className='state'+(open?' open':'');
 el.innerHTML='<b></b>'+(open?'Market open, closes 3:30 pm':(hol?'Market holiday today':(wk&&t<555?'Market opens 9:15 am':'Market closed')));
}}
tick(); setInterval(tick,1000);
</script></body></html>"""


def card_slots(res, log):
    rows = log[log["status"].isin(["OPEN", "EXIT_PENDING", "PENDING"])].to_dict("records") if log is not None else []
    bars = []
    for r in rows[:F.MAX_SLOTS]:
        st_ = r["status"]
        held = 1
        if st_ != "PENDING" and r.get("entry_date"):
            held = max(1, np.busday_count(pd.Timestamp(r["entry_date"]).date(), (res["as_of"] if res else dt.date.today()) + dt.timedelta(days=1)))
        hpct = 18 + 82 * min(held, F.MAX_HOLD) / F.MAX_HOLD if st_ != "PENDING" else 30
        cls = {"OPEN": "open", "EXIT_PENDING": "sell", "PENDING": "buy"}[st_]
        pnl = _num(r.get("pnl_pct"))
        top = "buy" if st_ == "PENDING" else (f"{pnl:+.1f}%" if np.isfinite(pnl) else "—")
        bars.append(f'<div class="mm-slot"><span class="top">{esc(top)}</span><div class="bar {cls}" style="height:{hpct:.0f}%"></div>'
                    f'<span class="lab">{esc(short(r["symbol"]))[:10]}</span></div>')
    for _ in range(F.MAX_SLOTS - len(bars)):
        bars.append('<div class="mm-slot"><span class="top" style="background:transparent;color:#8E8C9B">free</span>'
                    '<div class="bar free"></div><span class="lab">—</span></div>')
    used = min(len(rows), F.MAX_SLOTS)
    return f"""<div class="mm-card tall">
<div class="mm-h"><span class="mm-t">Slots</span><span class="mm-chip">6 × ₹25,000</span></div>
<div class="mm-big">{F.MAX_SLOTS - used}<span style="font-size:18px;color:#8E8C9B"> of 6 free</span></div>
<div class="mm-slots">{''.join(bars)}</div>
<div class="mm-legend"><span><i style="background:#C9B8FF"></i>holding (height = days held)</span>
<span><i style="background:#F3E28C"></i>sell at next open</span><span><i style="background:repeating-linear-gradient(135deg,#F3E28C 0 3px,#3a3726 3px 6px)"></i>buy at next open</span></div>
</div>"""


def card_orders(res):
    if not res:
        return ('<div class="mm-card mid"><div class="mm-h"><span class="mm-t">Next orders</span></div>'
                '<div class="mm-empty">Orders appear here after the evening scan.</div></div>')
    head = f'Orders for {res["next_session"].strftime("%a, %d %b")} at 9:15 am'
    rows = []
    for s in res["sells"]:
        rows.append(f'<div class="mm-row"><span class="b-but">SELL</span><div class="grow"><div class="nm">{esc(short(s["symbol"]))}</div>'
                    f'<div class="dt">{esc(REASON_TEXT.get(s["exit_reason"], s["exit_reason"]))}, {esc(s["qty"])} shares</div></div>'
                    f'<div class="val">market order</div></div>')
    for b in res["buys"]:
        lim = b["stop"] * (1 - GTT_LIMIT_BELOW)
        rows.append(f'<div class="mm-row"><span class="b-lav">BUY</span><div class="grow"><div class="nm">{esc(short(b["symbol"]))}</div>'
                    f'<div class="dt">{esc(b["sector"])}, sector {b["rank"]}, RSI(2) {b["rsi"]:.1f}</div></div>'
                    f'<div class="val">{b["qty"]} sh ≈ {inr(b["value"])}<br><span style="color:#8E8C9B;font-weight:500">'
                    f'GTT {indian(b["stop"], 2)} / limit {indian(lim, 2)}</span></div></div>')
    if res["mode"] == "market_hours":
        body = '<div class="mm-empty">The market day is not finished. Tomorrow\'s orders appear after 4:00 pm IST.</div>'
    elif res["mode"] == "stale":
        body = '<div class="mm-empty">Today\'s closing prices are not available yet. Run the scan again in 15 minutes.</div>'
    elif not rows:
        body = '<div class="mm-empty">No orders. Nothing to buy or sell at the next open.</div>'
    else:
        body = "".join(rows)
    extra = ""
    if res["skipped"]:
        names = ", ".join(f'{short(s["symbol"])} ({s["why"]})' for s in res["skipped"][:6])
        extra = f'<div class="mm-note">Also qualified but not bought: {esc(names)}</div>'
    if res["stopped_today"]:
        names = ", ".join(short(s["symbol"]) for s in res["stopped_today"])
        extra += f'<div class="mm-note">Stop-loss hit today: {esc(names)}. Check that the GTT order sold them.</div>'
    return f'<div class="mm-card mid"><div class="mm-h"><span class="mm-t">{esc(head)}</span></div>{body}{extra}</div>'


def card_sectors(res):
    if not res:
        return '<div class="mm-card mid"><div class="mm-h"><span class="mm-t">Strongest sectors</span></div><div class="mm-empty">Sector strength loads with the scan.</div></div>'
    top = res["sectors"].dropna(subset=["Rank"]).head(3).to_dict("records")
    spots = [(0, 16, 170, "#C9B8FF"), (150, 110, 130, "#F3E28C"), (160, 0, 104, "#9FD4FF")]
    bub = []
    for (x, y, size, col), s in zip(spots, top):
        ok = "Top-2, uptrend" if s["Eligible"] else s["Trend"]
        move = s["21-day move %"]
        bub.append(f'<div class="mm-bub" style="left:{x}px;top:{y}px;width:{size}px;height:{size}px;background:{col}">'
                   f'<b style="font-size:{22 if size > 120 else 17}px">{move:+.1f}%</b><span>{esc(s["Sector"])}</span>'
                   f'<span style="opacity:.75">{esc(ok)}</span></div>')
    return (f'<div class="mm-card mid"><div class="mm-h"><span class="mm-t">Strongest sectors</span>'
            f'<span class="mm-chip">21-day move</span></div><div class="mm-bubbles">{"".join(bub)}</div></div>')


def card_breadth(res):
    if not res:
        return '<div class="mm-card mid"><div class="mm-h"><span class="mm-t">Market breadth</span></div><div class="mm-empty">Breadth loads with the scan.</div></div>'
    up, tot = res["breadth"]
    cols, r, gap = 25, 4.2, 12.4
    circles = []
    for i, v in enumerate(res["dots"]):
        cx, cy = 6 + (i % cols) * gap, 6 + (i // cols) * gap
        fill = "#3A3A45" if v is None else ("#C9B8FF" if v else "#2A2A33")
        circles.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"/>')
    h = 12 + ((len(res["dots"]) - 1) // cols) * gap
    pct = 100 * up / tot if tot else 0
    return f"""<div class="mm-card mid"><div class="mm-h"><span class="mm-t">Market breadth</span><span class="mm-chip">above 200-day average</span></div>
<div class="mm-big" style="font-size:38px">{pct:.0f}%<span style="font-size:15px;color:#8E8C9B"> &nbsp;{up} of {tot} stocks</span></div>
<svg viewBox="0 0 {12 + (cols - 1) * gap:.0f} {h:.0f}" width="100%" style="margin-top:12px" role="img" aria-label="Nifty 500 breadth, grouped by sector">{''.join(circles)}</svg>
<div class="mm-note">Each dot is one Nifty 500 stock, grouped by sector. Lavender: above its 200-day average.</div></div>"""


def card_positions(res, log):
    rows = log[log["status"].isin(["OPEN", "EXIT_PENDING"])].to_dict("records") if log is not None else []
    items = []
    for r in rows:
        pnl = _num(r.get("pnl_pct"))
        b = ("b-up" if pnl >= 0 else "b-dn") if np.isfinite(pnl) else "b-lav"
        txt = f"{pnl:+.2f}%" if np.isfinite(pnl) else "—"
        items.append(f'<div class="mm-row"><div class="grow"><div class="nm">{esc(short(r["symbol"]))}</div>'
                     f'<div class="dt">Bought {esc(pretty_date(r.get("entry_date", "")))} at {indian(_num(r.get("entry_price"), 0), 2)}, '
                     f'stop {indian(_num(r.get("stop_loss"), 0), 2)}</div></div>'
                     f'<span class="{b}">{txt}</span></div>')
    body = "".join(items) if items else '<div class="mm-empty">No open positions.</div>'
    closed = log[log["status"] == "CLOSED"] if log is not None else pd.DataFrame()
    total = pd.to_numeric(closed["pnl_rs"], errors="coerce").sum() if len(closed) else 0.0
    tail = f'<div class="mm-note">Closed trades: {len(closed)}, net {inr(total)} after costs</div>' if len(closed) else ""
    return f'<div class="mm-card mid"><div class="mm-h"><span class="mm-t">Open positions</span></div>{body}{tail}</div>'


FOOTER = f"""<div class="mm-foot">
<b>Disclaimer.</b> {OWNER} is not a SEBI-registered investment adviser or research analyst. This dashboard is a
personal research tool that shows the output of a rule-based system. It is not investment advice and not a
recommendation to buy or sell any security. Backtested or past results do not guarantee future returns.
Investing in the stock market involves risk, including the loss of capital. Anyone who invests does so with
their own money and at their own risk; {OWNER} accepts no responsibility for any profit or loss arising from
the use of this information.<br><br>
Any query? Please contact <a href="mailto:{EMAIL}">{EMAIL}</a>
</div>"""


# ------------------------------------------------------------------ page
now = now_ist()
c1, c2, c3 = st.columns([2.2, 1.05, 1.9], vertical_alignment="center")
with c1:
    st.markdown('<div class="mm-brand">Market Mood<small>Rule-based swing system for Nifty 500 stocks, frozen rules v1</small></div>',
                unsafe_allow_html=True)
with c2:
    run = st.button("Run today's scan", type="primary", width="stretch")
with c3:
    st.markdown(f"""<div class="mm-profile"><div class="mm-avatar">KT</div><div>
<div class="mm-pname">{OWNER}</div><div class="mm-pmail"><a href="mailto:{EMAIL}">{EMAIL}</a></div></div></div>""",
                unsafe_allow_html=True)

if run:
    with st.spinner("Downloading 5 years of prices for 500 stocks and running the frozen rules (1–3 minutes)…"):
        st.session_state["res"] = run_scan()

res = st.session_state.get("res")
if res and res.get("error"):
    st.error(res["error"])
    res = None
if res:
    log = res["log"]
    if res["log_err"]:
        st.warning(res["log_err"])
    if res["save_err"]:
        st.warning(res["save_err"])
    if res["n_stocks"] < 0.9 * res["n_total"]:
        st.warning(f"Only {res['n_stocks']} of {res['n_total']} stocks have prices for {res['as_of']:%d %b}. "
                   "Yahoo may be slow; run the scan again later before placing orders.")
else:
    log, err = load_log()
    if err:
        st.warning(err)

tab_over, tab_pos, tab_sec, tab_rules = st.tabs(["Overview", "Positions", "Sectors", "Rules"])

with tab_over:
    a, b, c = st.columns([1.55, 1.05, 1.15], gap="medium")
    with a:
        st.markdown(card_nifty(res), unsafe_allow_html=True)
    with b:
        if hasattr(st, "iframe"):
            st.iframe(calendar_html(now), height=376)
        else:
            components.html(calendar_html(now), height=376)
    with c:
        st.markdown(card_slots(res, log), unsafe_allow_html=True)
    st.write("")
    d, e, f, g = st.columns([1.35, 0.95, 1.1, 1.0], gap="medium")
    with d:
        st.markdown(card_orders(res), unsafe_allow_html=True)
    with e:
        st.markdown(card_sectors(res), unsafe_allow_html=True)
    with f:
        st.markdown(card_breadth(res), unsafe_allow_html=True)
    with g:
        st.markdown(card_positions(res, log), unsafe_allow_html=True)

with tab_pos:
    view = (log.copy() if log is not None else pd.DataFrame(columns=LOG_COLS)).astype(str)
    view["symbol"] = view["symbol"].map(short)
    live = view[view["status"].isin(["PENDING", "OPEN", "EXIT_PENDING"])]
    done = view[view["status"].isin(["CLOSED", "SKIPPED"])]
    st.markdown("#### Current positions")
    if len(live):
        st.dataframe(live, hide_index=True, width="stretch")
    else:
        st.caption("No open or pending positions.")
    st.markdown("#### Closed trades")
    if len(done):
        cl = done[done["status"] == "CLOSED"]
        pnl = pd.to_numeric(cl["pnl_rs"], errors="coerce")
        m1, m2, m3 = st.columns(3)
        m1.metric("Closed trades", len(cl))
        m2.metric("Winning trades", f"{100 * (pnl > 0).mean():.0f}%" if len(cl) else "—")
        m3.metric("Net profit (after costs)", inr(pnl.sum()) if len(cl) else "—")
        st.dataframe(done.iloc[::-1], hide_index=True, width="stretch")
    else:
        st.caption("No closed trades yet.")
    st.caption("The log lives in the Google Sheet 'market-mood-bot', tab 'system_log_v1'. It follows the frozen "
               "rules exactly; if you skip a trade in real life, delete its row there.")

with tab_sec:
    if res:
        st.dataframe(res["sectors"], hide_index=True, width="stretch")
        st.caption("Buys are allowed only in sectors ranked 1 or 2 that are also in a strong uptrend "
                   "(ADX above 25 and a rising 200-day average).")
    else:
        st.caption("Run today's scan to see all sectors.")

with tab_rules:
    st.markdown("""
#### Frozen rules v1 (26 Sep 2026)
**When to buy** (checked after the close, bought at the next day's open):
1. Nifty 50 closes above its 200-day average (risk-on).
2. Nifty 50 closes 0.5% or more below the previous close (a market-wide dip).
3. The stock's sector ranks 1 or 2 by 21-day strength and is in a strong uptrend.
4. The stock closes above its 200-day average with RSI(2) below 10.

**How much:** 6 slots of ₹25,000. Share count = ₹25,000 ÷ signal-day close. When more stocks qualify than
free slots, the stronger sector goes first, then the lower RSI(2).

**When to sell** (whichever comes first):
1. Stop-loss = signal-day close − 2 × ATR(14), active from the day of purchase (GTT order).
2. RSI(2) above 70, or a close above the 5-day average → sell at the next open.
3. Still holding after 10 trading days → sell at the next open.

**Test results before going live** (costs included): 2016–2023 locked test, never used to build the rules:
493 trades, +0.77% per trade, 6.6% a year, largest fall 9.4% (Nifty's was 38.4%). All pass criteria met.
Past and tested results do not guarantee future returns.
""")

st.markdown(FOOTER, unsafe_allow_html=True)
