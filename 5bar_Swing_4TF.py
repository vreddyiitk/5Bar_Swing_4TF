"""
chart_generator_4tf.py
======================
Reads scrip symbols from 'Scrips_500.xlsx', downloads OHLC data across four
timeframes, and saves a single TradingView-style (Light/White theme) PNG per stock.

┌─────────────────────┬─────────────────────┐
│  TOP-LEFT           │  TOP-RIGHT          │
│  Daily  – 100 bars  │  Weekly – 100 bars  │
│  Price + EMA9       │  Price + EMA9       │
│  Swing TSL + Signals│  Swing TSL + Signals│
│  MACD (12,26,9)     │  MACD (12,26,9)     │
├─────────────────────┼─────────────────────┤
│  BOTTOM-LEFT        │  BOTTOM-RIGHT       │
│  Monthly– 100 bars  │  Hourly – 100 bars  │
│  Price + EMA9       │  Price + EMA9       │
│  Swing TSL + Signals│  Swing TSL + Signals│
│  MACD (12,26,9)     │  MACD (12,26,9)     │
└─────────────────────┴─────────────────────┘

Accurate Swing Trading System [Pine Script v4 port]
────────────────────────────────────────────────────
  no   = 5  (swing period)

  res  = highest(high, no)          ← rolling 5-bar high
  sup  = lowest(low,  no)           ← rolling 5-bar low

  avd  = +1 if close > res[1]       ← bullish breakout
         -1 if close < sup[1]       ← bearish breakdown
          0 otherwise               ← no new signal

  avn  = valuewhen(avd≠0, avd, 0)   ← holds last directional value
         (equivalent: forward-fill avd, ignore 0s)

  tsl  = sup  if avn==+1            ← trail stop = recent low  in bull
         res  if avn==-1            ← trail stop = recent high in bear

  BUY  = close crosses OVER  tsl
  SELL = close crosses UNDER tsl

  TSL line colour  : green if close ≥ tsl, red if close ≤ tsl
  Bar colour       : same green/red as TSL

  On-chart markers:
    ▲ BUY  label (green box, black text) below bar low
    ▼ SELL label (red   box, black text) above bar high
    ● MACD turns +ve : green circle below bar
    ● MACD turns -ve : red   circle above bar

Requirements:
    pip install yfinance pandas openpyxl matplotlib

Usage:
    Place Scrips_500.xlsx in the same directory and run:
        python chart_generator_4tf.py
"""

import os
import time
import warnings
import traceback
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as _Rect
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EXCEL_FILE   = "Scrips_500.xlsx"
SYMBOL_COL   = None
EXCHANGE_SFX = ".NS"
OUTPUT_DIR   = "YF_4TF_500_5bar"

N_BARS      = 100
EMA_PERIOD  = 9
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# ── Swing Trading System parameter ───────────
SWING_NO    = 5      # Pine: no=5

MAX_RETRIES = 3
RETRY_DELAY = 5

LOOKBACK = {
    "1d":  180,
    "1wk": 730,
    "1mo": 3650,
    "1h":  120,
}

# ── White / Light theme ───────────────────────
STYLE = {
    "bg":             "#FFFFFF",
    "panel_bg":       "#FAFAFA",
    "grid":           "#E0E0E0",
    "text":           "#1A1A2E",
    "subtext":        "#5A5A72",
    "border":         "#CCCCCC",
    "zero_line":      "#999999",
    "up":             "#26A69A",   # candle up
    "dn":             "#EF5350",   # candle down
    "swing_bull":     "#00C853",   # TSL green
    "swing_bear":     "#D50000",   # TSL red
    "buy_lbl_bg":     "#00C853",
    "sell_lbl_bg":    "#D50000",
    "macd_turn_bull": "#00C853",
    "macd_turn_bear": "#D50000",
    # per-TF EMA / MACD
    "d_ema":  "#E65100", "d_macd": "#1565C0", "d_sig": "#E65100",
    "d_hu":   "#26A69A", "d_hd":   "#EF5350",
    "w_ema":  "#6A1B9A", "w_macd": "#0277BD", "w_sig": "#AD1457",
    "w_hu":   "#26A69A", "w_hd":   "#EF5350",
    "m_ema":  "#00695C", "m_macd": "#4527A0", "m_sig": "#BF360C",
    "m_hu":   "#26A69A", "m_hd":   "#EF5350",
    "h_ema":  "#1B5E20", "h_macd": "#0D47A1", "h_sig": "#B71C1C",
    "h_hu":   "#26A69A", "h_hd":   "#EF5350",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def read_symbols(path, col):
    df = pd.read_excel(path, sheet_name=0)
    if col and col in df.columns:
        return df[col].dropna().astype(str).str.strip().tolist()
    for c in df.columns:
        sample = df[c].dropna().astype(str).str.strip()
        mask   = sample.str.match(r'^[A-Z0-9&\-]{2,20}$')
        if mask.sum() > len(sample) * 0.5:
            print(f"  Auto-detected symbol column: '{c}'")
            return sample[mask].tolist()
    return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def macd_calc(close, fast=12, slow=26, signal=9):
    ml = ema(close, fast) - ema(close, slow)
    sl = ema(ml, signal)
    return ml, sl, ml - sl


def download_with_retry(ticker, start, end, interval):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval=interval, auto_adjust=True,
                             progress=False)
            return df
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  ! retry {attempt}/{MAX_RETRIES} in {wait}s …",
                      end="", flush=True)
                time.sleep(wait)
    raise last_exc


def fetch_ohlc(ticker, interval):
    end_dt   = datetime.today() + timedelta(days=1)
    start_dt = end_dt - timedelta(days=LOOKBACK[interval])
    try:
        df = download_with_retry(ticker,
                                 start=start_dt.strftime("%Y-%m-%d"),
                                 end=end_dt.strftime("%Y-%m-%d"),
                                 interval=interval)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        if interval == "1h":
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("Asia/Kolkata")
            df = df.between_time("09:15","15:30")
            df.index = df.index.tz_localize(None)
        min_bars = 1 if interval == "1mo" else MACD_SLOW + MACD_SIGNAL + SWING_NO + 2
        if len(df) < min_bars:
            return None
        return df.tail(N_BARS)
    except Exception:
        return None


# ─────────────────────────────────────────────
# ACCURATE SWING TRADING SYSTEM  (Pine → Python)
# ─────────────────────────────────────────────

def calc_swing(df, no=SWING_NO):
    """
    Exact port of Pine Script v4 "Accurate Swing Trading System".

    Returns dict:
        tsl        : pd.Series  trailing stop level
        tsl_color  : list[str]  'green' or 'red' per bar
        buy_sig    : np.ndarray[bool]  BUY  signal (close crosses over  tsl)
        sell_sig   : np.ndarray[bool]  SELL signal (close crosses under tsl)
        bar_color  : list[str]  per-bar hex colour
    """
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n     = len(df)

    # Rolling highest high / lowest low of last `no` bars (inclusive)
    res = pd.Series(high).rolling(no, min_periods=1).max().values   # highest(high, no)
    sup = pd.Series(low).rolling(no, min_periods=1).min().values    # lowest(low,  no)

    # avd: +1 / -1 / 0  (compare close to res[1] / sup[1])
    avd = np.zeros(n, dtype=int)
    for i in range(1, n):
        if   close[i] > res[i-1]: avd[i] =  1
        elif close[i] < sup[i-1]: avd[i] = -1
        else:                     avd[i] =  0

    # avn = valuewhen(avd != 0, avd, 0)
    # → forward-fill the last non-zero avd value; starts at 0
    avn = np.zeros(n, dtype=int)
    last_nonzero = 0
    for i in range(n):
        if avd[i] != 0:
            last_nonzero = avd[i]
        avn[i] = last_nonzero

    # tsl: avn==+1 → sup (trail below); avn==-1 → res (trail above)
    tsl = np.where(avn == 1, sup, res)

    # BUY  = crossover (close, tsl):  close[i-1] <= tsl[i-1]  AND  close[i] > tsl[i]
    # SELL = crossunder(close, tsl):  close[i-1] >= tsl[i-1]  AND  close[i] < tsl[i]
    buy_sig  = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if close[i-1] <= tsl[i-1] and close[i] > tsl[i]:
            buy_sig[i]  = True
        if close[i-1] >= tsl[i-1] and close[i] < tsl[i]:
            sell_sig[i] = True

    # TSL colour + bar colour: green if close >= tsl, red if close <= tsl
    tsl_color = []
    bar_color = []
    for i in range(n):
        if   close[i] >= tsl[i]: c = STYLE["swing_bull"]
        elif close[i] <= tsl[i]: c = STYLE["swing_bear"]
        else:                    c = STYLE["subtext"]
        tsl_color.append(c)
        bar_color.append(c)

    return dict(
        tsl       = pd.Series(tsl,  index=df.index),
        tsl_color = tsl_color,
        buy_sig   = buy_sig,
        sell_sig  = sell_sig,
        bar_color = bar_color,
    )


# ─────────────────────────────────────────────
# DRAW ONE QUARTER
# ─────────────────────────────────────────────

def draw_quarter(ax_price, ax_macd, df, label, colors, date_fmt, y_side="right"):
    s   = STYLE
    n   = len(df)
    xs  = np.arange(n)

    ema9 = ema(df["Close"], EMA_PERIOD)
    sw   = calc_swing(df)

    has_macd = n >= MACD_SLOW + MACD_SIGNAL + 2
    if has_macd:
        macd_l, sig, hist = macd_calc(df["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        hist_vals = hist.values
        macd_bull_turn = np.zeros(n, dtype=bool)
        macd_bear_turn = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if hist_vals[i-1] <= 0 and hist_vals[i] > 0: macd_bull_turn[i] = True
            if hist_vals[i-1] >= 0 and hist_vals[i] < 0: macd_bear_turn[i] = True
    else:
        macd_l = sig = hist = hist_vals = None
        macd_bull_turn = macd_bear_turn = np.zeros(n, dtype=bool)

    # ── Style axes ────────────────────────────
    for ax in (ax_price, ax_macd):
        ax.set_facecolor(s["panel_bg"])
        ax.tick_params(colors=s["text"], labelsize=7, width=1.0, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(s["border"])
            spine.set_linewidth(0.8)
        ax.grid(True, color=s["grid"], linewidth=0.5, alpha=0.9)

    # ── Candlesticks — coloured by Swing TSL state ─────────
    avg_range  = (df["High"] - df["Low"]).mean()
    min_body_h = avg_range * 0.04
    BODY_W     = 0.6

    for i, (_, row) in enumerate(df.iterrows()):
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        bar_col = sw["bar_color"][i]
        body_h  = max(abs(c - o), min_body_h)

        ax_price.vlines(i, l, h, color=bar_col, linewidth=1.2, zorder=2)
        ax_price.add_patch(_Rect(
            (i - BODY_W/2, min(o, c)), BODY_W, body_h,
            facecolor=bar_col, edgecolor=bar_col,
            linewidth=0, zorder=3))

    # ── EMA ───────────────────────────────────
    ax_price.plot(xs, ema9.values, color=colors["ema"],
                  linewidth=1.6, zorder=4, label=f"EMA {EMA_PERIOD}")

    # ── TSL line (segmented by colour) ────────
    # Draw each bar-to-bar segment in its own colour
    tsl_vals  = sw["tsl"].values
    tsl_colors = sw["tsl_color"]
    for i in range(1, n):
        col = tsl_colors[i]
        ax_price.plot([xs[i-1], xs[i]], [tsl_vals[i-1], tsl_vals[i]],
                      color=col, linewidth=2.2, solid_capstyle="round",
                      zorder=5)

    # ── BUY / SELL labels ─────────────────────
    price_range  = df["High"].max() - df["Low"].min()
    lbl_off      = price_range * 0.012

    for i in range(n):
        if sw["buy_sig"][i]:
            ax_price.annotate(
                "BUY",
                xy=(xs[i], df["Low"].iloc[i] - lbl_off),
                fontsize=6.5, fontweight="bold",
                color="#000000", ha="center", va="top", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=s["buy_lbl_bg"],
                          edgecolor="none", alpha=0.92))
        if sw["sell_sig"][i]:
            ax_price.annotate(
                "SELL",
                xy=(xs[i], df["High"].iloc[i] + lbl_off),
                fontsize=6.5, fontweight="bold",
                color="#000000", ha="center", va="bottom", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=s["sell_lbl_bg"],
                          edgecolor="none", alpha=0.92))

    # ── MACD direction-change circles ─────────
    circle_off = price_range * 0.022
    for i in range(n):
        if macd_bull_turn[i]:
            ax_price.plot(xs[i], df["Low"].iloc[i] - circle_off - lbl_off * 1.5,
                          marker="o", markersize=7,
                          color=s["macd_turn_bull"], markeredgecolor="white",
                          markeredgewidth=0.7, zorder=8, linestyle="None")
        if macd_bear_turn[i]:
            ax_price.plot(xs[i], df["High"].iloc[i] + circle_off + lbl_off * 1.5,
                          marker="o", markersize=7,
                          color=s["macd_turn_bear"], markeredgecolor="white",
                          markeredgewidth=0.7, zorder=8, linestyle="None")

    # ── Axes limits ───────────────────────────
    ax_price.set_xlim(-1, n + 1)
    pad = price_range * 0.08   # extra room for labels above/below
    ax_price.set_ylim(df["Low"].min() - pad, df["High"].max() + pad)
    ax_price.yaxis.set_label_position(y_side)
    ax_price.yaxis.tick_right() if y_side == "right" else ax_price.yaxis.tick_left()
    ax_price.set_ylabel("Price (₹)", color=s["text"], fontsize=7)
    ax_price.yaxis.set_tick_params(labelsize=7, labelcolor=s["text"])

    ax_macd.yaxis.set_label_position(y_side)
    ax_macd.yaxis.tick_right() if y_side == "right" else ax_macd.yaxis.tick_left()
    ax_macd.set_ylabel("MACD", color=s["text"], fontsize=7)
    ax_macd.yaxis.set_tick_params(labelsize=6, labelcolor=s["text"])

    # ── Last-price pills ──────────────────────
    last_close = df["Close"].iloc[-1]
    last_ema   = ema9.iloc[-1]
    close_col  = sw["bar_color"][-1]
    for val, col in [(last_close, close_col), (last_ema, colors["ema"])]:
        ax_price.annotate(f"₹{val:,.2f}",
                          xy=(1, val), xycoords=("axes fraction","data"),
                          xytext=(4,0), textcoords="offset points",
                          fontsize=7.5, fontweight="bold", color="#FFFFFF",
                          ha="left", va="center",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor=col, edgecolor="none", alpha=0.97),
                          annotation_clip=False)

    # ── Legend ────────────────────────────────
    leg = [
        mpatches.Patch(facecolor=s["up"], label="Bull candle"),
        mpatches.Patch(facecolor=s["dn"], label="Bear candle"),
        Line2D([0],[0], color=colors["ema"], linewidth=1.5,
               label=f"EMA {EMA_PERIOD}"),
        Line2D([0],[0], color=s["swing_bull"], linewidth=2.2,
               label="TSL Bull"),
        Line2D([0],[0], color=s["swing_bear"], linewidth=2.2,
               label="TSL Bear"),
        Line2D([0],[0], marker="^", color="w",
               markerfacecolor=s["buy_lbl_bg"],  markersize=7,
               linestyle="None", label="BUY"),
        Line2D([0],[0], marker="v", color="w",
               markerfacecolor=s["sell_lbl_bg"], markersize=7,
               linestyle="None", label="SELL"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=s["macd_turn_bull"], markersize=6,
               linestyle="None", label="MACD +ve"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=s["macd_turn_bear"], markersize=6,
               linestyle="None", label="MACD -ve"),
    ]
    ax_price.legend(handles=leg, loc="upper left", fontsize=5.8,
                    framealpha=0.85, facecolor=s["bg"],
                    edgecolor=s["border"], labelcolor=s["text"], ncol=3)

    # ── Timeframe badge ───────────────────────
    ax_price.set_title(f"  {label}  ",
                       loc="left", color="#FFFFFF",
                       fontsize=10, fontweight="bold", pad=3,
                       bbox=dict(boxstyle="round,pad=0.35",
                                 facecolor=colors["ema"],
                                 edgecolor="none", alpha=0.95))

    # ── MACD panel ────────────────────────────
    if has_macd:
        hcols = [colors["hu"] if v >= 0 else colors["hd"] for v in hist_vals]
        ax_macd.bar(xs, hist_vals, color=hcols, alpha=0.80, width=0.65, zorder=2)
        ax_macd.plot(xs, macd_l.values, color=colors["macd"], linewidth=1.0,
                     zorder=3, label="MACD")
        ax_macd.plot(xs, sig.values,    color=colors["sig"],  linewidth=0.9,
                     zorder=3, label="Signal")
        ax_macd.axhline(0, color=s["zero_line"], linewidth=0.8, linestyle="--")
        ax_macd.legend(loc="upper left", fontsize=6.5, framealpha=0.85,
                       facecolor=s["bg"], edgecolor=s["border"],
                       labelcolor=s["text"])
    else:
        ax_macd.set_facecolor(s["panel_bg"])
        ax_macd.set_xticks([]); ax_macd.set_yticks([])
        ax_macd.text(0.5, 0.5,
                     f"MACD needs ≥{MACD_SLOW+MACD_SIGNAL+2} bars (have {n})",
                     transform=ax_macd.transAxes,
                     color=s["subtext"], fontsize=7, ha="center", va="center",
                     style="italic")

    # ── X-axis labels ─────────────────────────
    step = max(n // 8, 1)
    ax_macd.set_xticks(xs[::step])
    ax_macd.set_xticklabels(
        [df.index[i].strftime(date_fmt) for i in range(0, n, step)],
        rotation=25, ha="right", fontsize=7, color=s["text"])
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # ── % change caption ──────────────────────
    lc = df["Close"].iloc[-1]; fc = df["Close"].iloc[0]
    pct  = (lc - fc) / fc * 100
    sign = "+" if pct >= 0 else ""
    ccol = s["swing_bull"] if pct >= 0 else s["swing_bear"]
    ax_price.text(0.99, 0.02, f"{sign}{pct:.2f}%  ({n} bars)",
                  transform=ax_price.transAxes,
                  color=ccol, fontsize=7.5, fontweight="bold",
                  ha="right", va="bottom")


# ─────────────────────────────────────────────
# MASTER CHART  (4 quarters)
# ─────────────────────────────────────────────

def plot_chart(symbol, d_df, w_df, m_df, h_df, output_path):
    s = STYLE

    fig = plt.figure(figsize=(19.2, 10.4), dpi=100, facecolor=s["bg"])
    fig.patch.set_facecolor(s["bg"])

    gs = gridspec.GridSpec(
        4, 2,
        height_ratios=[7, 3, 7, 3],
        width_ratios=[1, 1],
        hspace=0.10, wspace=0.14,
        top=0.93, bottom=0.05,
        left=0.02, right=0.97,
    )

    ax_d_p  = fig.add_subplot(gs[0, 0])
    ax_d_m  = fig.add_subplot(gs[1, 0], sharex=ax_d_p)
    ax_w_p  = fig.add_subplot(gs[0, 1])
    ax_w_m  = fig.add_subplot(gs[1, 1], sharex=ax_w_p)
    ax_mo_p = fig.add_subplot(gs[2, 0])
    ax_mo_m = fig.add_subplot(gs[3, 0], sharex=ax_mo_p)
    ax_h_p  = fig.add_subplot(gs[2, 1])
    ax_h_m  = fig.add_subplot(gs[3, 1], sharex=ax_h_p)

    tf_colors = {
        "d":  dict(up=s["up"], dn=s["dn"], ema=s["d_ema"],
                   macd=s["d_macd"], sig=s["d_sig"], hu=s["d_hu"], hd=s["d_hd"]),
        "w":  dict(up=s["up"], dn=s["dn"], ema=s["w_ema"],
                   macd=s["w_macd"], sig=s["w_sig"], hu=s["w_hu"], hd=s["w_hd"]),
        "mo": dict(up=s["up"], dn=s["dn"], ema=s["m_ema"],
                   macd=s["m_macd"], sig=s["m_sig"], hu=s["m_hu"], hd=s["m_hd"]),
        "h":  dict(up=s["up"], dn=s["dn"], ema=s["h_ema"],
                   macd=s["h_macd"], sig=s["h_sig"], hu=s["h_hu"], hd=s["h_hd"]),
    }

    def no_data(ax_p, ax_m, label, key):
        col = tf_colors[key]["ema"]
        for ax in (ax_p, ax_m):
            ax.set_facecolor(s["panel_bg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(s["border"])
            ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(left=False, right=False, bottom=False,
                           labelbottom=False, labelleft=False, labelright=False)
        ax_p.set_title(f"  {label}  ", loc="left", color="#FFFFFF",
                       fontsize=10, fontweight="bold", pad=3,
                       bbox=dict(boxstyle="round,pad=0.35",
                                 facecolor=col, edgecolor="none", alpha=0.95))
        ax_p.text(0.5, 0.5, "No data\n(recently listed stock)",
                  transform=ax_p.transAxes,
                  color=s["subtext"], fontsize=10,
                  ha="center", va="center", style="italic")
        ax_m.text(0.5, 0.5, "—", transform=ax_m.transAxes,
                  color=s["subtext"], fontsize=10, ha="center", va="center")

    quarters = [
        (ax_d_p,  ax_d_m,  d_df, "Daily",   "d",  "%d %b '%y",   "left"),
        (ax_w_p,  ax_w_m,  w_df, "Weekly",  "w",  "%d %b '%y",   "right"),
        (ax_mo_p, ax_mo_m, m_df, "Monthly", "mo", "%b '%y",      "left"),
        (ax_h_p,  ax_h_m,  h_df, "Hourly",  "h",  "%d %b %H:%M", "right"),
    ]

    for ax_p, ax_m, df, label, key, dfmt, yside in quarters:
        if df is not None and len(df) >= 1:
            draw_quarter(ax_p, ax_m, df, label, tf_colors[key], dfmt, yside)
        else:
            no_data(ax_p, ax_m, label, key)

    # ── Master title ──────────────────────────
    latest = (d_df.index[-1].strftime("%d %b %Y")
              if d_df is not None and not d_df.empty else "–")
    lc = (float(d_df["Close"].iloc[-1]) if d_df is not None and not d_df.empty else 0)

    fig.text(0.03, 0.965,
             f"{symbol}  |  NSE  |  Multi-Timeframe  |  Accurate Swing System",
             color=s["text"], fontsize=11, fontweight="bold")
    fig.text(0.03, 0.945,
             f"₹{lc:,.2f}  |  {N_BARS} bars/TF  "
             f"|  TSL(swing={SWING_NO})  "
             f"|  ▲BUY  ▼SELL  ●MACD±",
             color=s["subtext"], fontsize=8)
    fig.text(0.97, 0.965, f"Latest: {latest}",
             color=s["text"], fontsize=8, ha="right", fontweight="bold")
    fig.text(0.97, 0.945,
             f"EMA {EMA_PERIOD}  |  MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})"
             f"  |  Data: yfinance",
             color=s["subtext"], fontsize=7, ha="right")

    # ── Quarter dividers ──────────────────────
    for xy in [([0.02,0.97],[0.50,0.50]), ([0.50,0.50],[0.05,0.93])]:
        fig.add_artist(plt.Line2D(xy[0], xy[1],
                                  transform=fig.transFigure,
                                  color=s["border"], linewidth=1.5, alpha=0.8))

    plt.savefig(output_path, dpi=100, facecolor=s["bg"], edgecolor="none")
    plt.close(fig)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  Multi-TF Chart Generator  –  NSE  |  White Theme")
    print(f"  + Accurate Swing Trading System (swing={SWING_NO})")
    print(f"  Quarters: Daily · Weekly · Monthly · Hourly  |  {N_BARS} bars")
    print(f"{'='*65}")

    if not os.path.exists(EXCEL_FILE):
        print(f"\n[ERROR] '{EXCEL_FILE}' not found.")
        return

    symbols = read_symbols(EXCEL_FILE, SYMBOL_COL)
    print(f"\n  Loaded {len(symbols)} symbols from '{EXCEL_FILE}'")

    success, failed = [], []

    for i, sym in enumerate(symbols, 1):
        ticker = sym if sym.endswith(EXCHANGE_SFX) else sym + EXCHANGE_SFX
        print(f"\n[{i:>3}/{len(symbols)}]  {ticker:<22}", end="", flush=True)
        try:
            d_df = fetch_ohlc(ticker, "1d")
            w_df = fetch_ohlc(ticker, "1wk")
            m_df = fetch_ohlc(ticker, "1mo")
            h_df = fetch_ohlc(ticker, "1h")

            counts = (f"D:{len(d_df) if d_df is not None else 0}"
                      f"  W:{len(w_df) if w_df is not None else 0}"
                      f"  M:{len(m_df) if m_df is not None else 0}"
                      f"  H:{len(h_df) if h_df is not None else 0}")
            print(f"  {counts}", end="", flush=True)

            out = os.path.join(OUTPUT_DIR, f"{sym}.png")
            plot_chart(sym, d_df, w_df, m_df, h_df, out)
            print(f"  →  {out}")
            success.append(sym)

        except Exception:
            print(f"  ✗  Error")
            traceback.print_exc()
            failed.append(sym)

    print(f"\n{'='*65}")
    print(f"  Done!  {len(success)} charts saved to '{OUTPUT_DIR}/'")
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(failed[:20])}"
              + (" …" if len(failed) > 20 else ""))
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()