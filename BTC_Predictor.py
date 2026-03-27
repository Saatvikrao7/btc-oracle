"""
Bitcoin 15-Minute Price Predictor
Combines statistical/ML signals with Claude AI analysis.
Tracks predictions, win rate, confidence calibration, and simulated Kalshi P&L.

Requirements:
    pip install requests anthropic numpy pandas scikit-learn rich

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python BTC_Predictor.py                   # AI mode, runs every 15 min
    python BTC_Predictor.py --no-ai           # rule-based only, no API key needed
    python BTC_Predictor.py --once            # single prediction then exit
    python BTC_Predictor.py --interval 5      # run every 5 minutes
    python BTC_Predictor.py --stats           # show win rate stats and exit
"""

import os
import sys
import time
import json
import re
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from dotenv import load_dotenv

# Load .env file from same directory as this script
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_script_dir, ".env"), override=True)

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL           = "XBTUSD"
INTERVAL         = 15
CANDLES_FETCH    = 100
SHORT_MA         = 9
LONG_MA          = 21
RSI_PERIOD       = 14
PREDICTIONS_FILE = os.path.join(os.path.dirname(__file__), "btc_predictions_log.json")
MODEL_FILE       = os.path.join(os.path.dirname(__file__), "btc_xgb_model.pkl")

# Feature order must match train_model.py FEATURE_COLS exactly
XGB_FEATURE_COLS = [
    "rsi", "macd", "bb_pct",
    "price_vs_ma9", "price_vs_ma21", "ma_cross",
    "vol_trend", "price_change_1h", "price_change_4h",
    "ml_slope_pct", "ml_confidence", "ml_dir_up", "strategy_ai",
]

# Simulated Kalshi contract size ($1 payout per contract)
KALSHI_CONTRACT  = 1.0

console = Console()

# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_klines(symbol: str = SYMBOL, interval: int = INTERVAL, limit: int = CANDLES_FETCH) -> pd.DataFrame:
    """Fetch OHLCV candles from Kraken public API (no key needed)."""
    url = "https://api.kraken.com/0/public/OHLC"
    r = requests.get(url, params={"pair": symbol, "interval": interval}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise ValueError(f"Kraken error: {data['error']}")
    pair_key = list(data["result"].keys())[0]
    raw = data["result"][pair_key][-limit:]
    df = pd.DataFrame(raw, columns=["time","open","high","low","close","vwap","volume","count"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    return df[["open_time","open","high","low","close","volume"]]


# ── Technical Indicators ──────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> dict:
    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    vols   = df["volume"].values

    ema_fn = lambda arr, n: pd.Series(arr).ewm(span=n, adjust=False).mean().values

    # ── Moving Averages ──────────────────────────────────────────────────────
    ma_short = float(np.mean(closes[-SHORT_MA:]))
    ma_long  = float(np.mean(closes[-LONG_MA:]))

    # ── RSI ──────────────────────────────────────────────────────────────────
    deltas   = np.diff(closes[-RSI_PERIOD - 1:])
    gains    = np.where(deltas > 0, deltas, 0)
    losses   = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)  or 1e-9
    avg_loss = np.mean(losses) or 1e-9
    rsi      = 100 - (100 / (1 + avg_gain / avg_loss))

    # ── Bollinger Bands (20-period, 2σ) ──────────────────────────────────────
    bb_window = closes[-20:]
    bb_mid    = float(np.mean(bb_window))
    bb_std    = float(np.std(bb_window))
    bb_upper  = bb_mid + 2 * bb_std
    bb_lower  = bb_mid - 2 * bb_std
    bb_pct    = (closes[-1] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    bb_width  = (bb_upper - bb_lower) / bb_mid * 100   # % width — squeeze detection

    # ── MACD (12/26/9) with signal line & histogram ──────────────────────────
    ema12     = ema_fn(closes, 12)
    ema26     = ema_fn(closes, 26)
    macd_line = float(ema12[-1] - ema26[-1])
    macd_signal = float(pd.Series(ema12 - ema26).ewm(span=9, adjust=False).mean().values[-1])
    macd_hist   = macd_line - macd_signal

    # ── Stochastic Oscillator (14-period) ────────────────────────────────────
    stoch_period = 14
    stoch_high   = float(np.max(highs[-stoch_period:]))
    stoch_low    = float(np.min(lows[-stoch_period:]))
    stoch_k      = (closes[-1] - stoch_low) / (stoch_high - stoch_low + 1e-9) * 100
    # %D = 3-period SMA of %K (approximate from last 3 closes)
    stoch_k_vals = [(closes[-i] - stoch_low) / (stoch_high - stoch_low + 1e-9) * 100 for i in range(3, 0, -1)]
    stoch_d      = float(np.mean(stoch_k_vals))

    # ── ATR (Average True Range, 14-period) — volatility measure ─────────────
    tr_values = []
    for i in range(-14, 0):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i]  - closes[i - 1]))
        tr_values.append(tr)
    atr = float(np.mean(tr_values))
    atr_pct = atr / closes[-1] * 100   # ATR as % of price

    # ── Volume features ──────────────────────────────────────────────────────
    vol_trend  = float(np.mean(vols[-5:])) / float(np.mean(vols[-10:-5]) + 1e-9)
    vol_sma_20 = float(np.mean(vols[-20:])) if len(vols) >= 20 else float(np.mean(vols))
    vol_spike  = float(vols[-1]) / (vol_sma_20 + 1e-9)   # current candle vs average

    # ── Candle patterns (last candle) ────────────────────────────────────────
    body       = closes[-1] - opens[-1]
    candle_range = highs[-1] - lows[-1] + 1e-9
    body_pct   = body / candle_range                      # +1 = full green, -1 = full red
    upper_wick = (highs[-1] - max(opens[-1], closes[-1])) / candle_range
    lower_wick = (min(opens[-1], closes[-1]) - lows[-1])  / candle_range

    # ── Consecutive candle direction ─────────────────────────────────────────
    consec_green = 0
    consec_red   = 0
    for i in range(len(closes) - 1, max(0, len(closes) - 20), -1):
        if closes[i] >= opens[i]:
            if consec_red > 0:
                break
            consec_green += 1
        else:
            if consec_green > 0:
                break
            consec_red += 1

    # ── Multi-timeframe price changes ────────────────────────────────────────
    price_change_15m = (closes[-1] - closes[-2])  / closes[-2]  * 100 if len(closes) >= 2  else 0
    price_change_30m = (closes[-1] - closes[-3])  / closes[-3]  * 100 if len(closes) >= 3  else 0
    price_change_1h  = (closes[-1] - closes[-4])  / closes[-4]  * 100 if len(closes) >= 4  else 0
    price_change_2h  = (closes[-1] - closes[-8])  / closes[-8]  * 100 if len(closes) >= 8  else 0
    price_change_4h  = (closes[-1] - closes[-16]) / closes[-16] * 100 if len(closes) >= 16 else 0

    # ── Recent high-low range (spread) ───────────────────────────────────────
    range_4h   = (float(np.max(highs[-16:])) - float(np.min(lows[-16:]))) / closes[-1] * 100
    range_1h   = (float(np.max(highs[-4:]))  - float(np.min(lows[-4:])))  / closes[-1] * 100

    # ── Time features ────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    hour_of_day = now.hour + now.minute / 60.0     # 0.0 – 24.0
    day_of_week = now.weekday()                     # 0=Mon, 6=Sun

    # ── Raw OHLCV for last 30 candles (for LSTM training) ────────────────────
    n_raw = min(30, len(closes))
    raw_candles = []
    for i in range(-n_raw, 0):
        raw_candles.append({
            "o": round(float(opens[i]),  2),
            "h": round(float(highs[i]),  2),
            "l": round(float(lows[i]),   2),
            "c": round(float(closes[i]), 2),
            "v": round(float(vols[i]),   4),
        })

    return {
        # Core (used by display + rules)
        "current_price":    closes[-1],
        "ma_short":         ma_short,
        "ma_long":          ma_long,
        "rsi":              float(rsi),
        "bb_upper":         bb_upper,
        "bb_lower":         bb_lower,
        "bb_mid":           bb_mid,
        "bb_pct":           float(bb_pct),
        "macd":             macd_line,
        "vol_trend":        vol_trend,
        "price_change_1h":  price_change_1h,
        "price_change_4h":  price_change_4h,
        # Extended (for ML features)
        "bb_width":         bb_width,
        "macd_signal":      macd_signal,
        "macd_hist":        macd_hist,
        "stoch_k":          float(stoch_k),
        "stoch_d":          stoch_d,
        "atr":              atr,
        "atr_pct":          atr_pct,
        "vol_spike":        vol_spike,
        "body_pct":         float(body_pct),
        "upper_wick":       float(upper_wick),
        "lower_wick":       float(lower_wick),
        "consec_green":     consec_green,
        "consec_red":       consec_red,
        "price_change_15m": price_change_15m,
        "price_change_30m": price_change_30m,
        "price_change_2h":  price_change_2h,
        "range_4h":         range_4h,
        "range_1h":         range_1h,
        "hour_of_day":      round(hour_of_day, 2),
        "day_of_week":      day_of_week,
        "raw_candles":      raw_candles,
    }


# ── ML Prediction (fixed confidence) ──────────────────────────────────────────

def ml_predict(df: pd.DataFrame) -> dict:
    """
    Linear regression slope for trend direction.
    Confidence = signal-to-noise ratio: how large is the expected 1-candle move
    relative to the regression residuals.  Much more meaningful than 1-RMSE/price.
    """
    closes = df["close"].values[-30:]
    X = np.arange(len(closes)).reshape(-1, 1)

    model     = LinearRegression().fit(X, y := closes)
    predicted = float(model.predict([[len(closes)]])[0])

    residuals  = closes - model.predict(X).flatten()
    rmse       = float(np.sqrt(np.mean(residuals ** 2))) or 1e-9
    slope      = float(model.coef_[0])
    signal     = abs(slope)                        # expected $/candle
    confidence = signal / (signal + rmse)          # 0 → pure noise, 1 → perfect trend
    slope_pct  = slope / closes[-1] * 100

    return {
        "predicted_price": predicted,
        "confidence":      round(confidence, 4),
        "slope_pct":       slope_pct,
        "rmse":            rmse,
        "direction":       "UP" if slope > 0 else "DOWN",
    }


# ── Rule-Based Prediction ──────────────────────────────────────────────────────

def rule_based_predict(indicators: dict, ml: dict) -> dict:
    """
    Score indicators as bullish (+1) or bearish (-1).

    Signals are split into two independent groups to prevent correlated
    indicators from dominating the vote:

    TREND signals (4)  — is the current trend up or down?
      1. MA crossover      : MA9 vs MA21
      2. MACD direction    : positive / negative
      3. 1h momentum       : last 4 candles net direction
      4. ML slope          : linear regression slope

    MEAN-REVERSION signals (4)  — is price stretched and likely to snap back?
      5. RSI extreme       : <35 = oversold (bullish), >65 = overbought (bearish)
      6. Bollinger position: <25% = near lower band (bullish), >75% = near upper (bearish)
      7. Short-term drop   : price fell >0.4% in 1h → bounce candidate (+1)
         Short-term surge  : price rose >0.4% in 1h → fade candidate (−1)
      8. 4h exhaustion     : >1.5% 4h move in either direction → likely to mean-revert

    Keeping both groups the same size (4 each) prevents a sustained trend
    from always producing an 8-0 score.
    """
    p    = indicators["current_price"]
    rsi  = indicators["rsi"]
    bb   = indicators["bb_pct"]
    ch1h = indicators["price_change_1h"]
    ch4h = indicators["price_change_4h"]

    trend_signals = []
    rev_signals   = []

    # ── Trend group ──────────────────────────────────────────────────────────
    # 1. MA crossover (single consolidated MA signal — not price vs each MA separately)
    trend_signals.append(+1 if indicators["ma_short"] > indicators["ma_long"] else -1)
    # 2. MACD
    trend_signals.append(+1 if indicators["macd"] > 0 else -1)
    # 3. 1h momentum direction
    trend_signals.append(+1 if ch1h > 0 else -1)
    # 4. ML slope
    trend_signals.append(+1 if ml["direction"] == "UP" else -1)

    # ── Mean-reversion group ─────────────────────────────────────────────────
    # 5. RSI extremes — neutral zone counts as 0
    if   rsi < 35:  rev_signals.append(+1)   # oversold → expect bounce
    elif rsi > 65:  rev_signals.append(-1)   # overbought → expect fade
    else:           rev_signals.append(0)

    # 6. Bollinger Band position extremes
    if   bb < 0.25: rev_signals.append(+1)   # near lower band → mean reversion up
    elif bb > 0.75: rev_signals.append(-1)   # near upper band → mean reversion down
    else:           rev_signals.append(0)

    # 7. Short-term 1h overextension (contrarian)
    if   ch1h < -0.4: rev_signals.append(+1)   # big drop → bounce candidate
    elif ch1h >  0.4: rev_signals.append(-1)   # big surge → fade candidate
    else:             rev_signals.append(0)

    # 8. 4h exhaustion (contrarian)
    if   ch4h < -1.5: rev_signals.append(+1)   # large 4h drop → mean-revert up
    elif ch4h >  1.5: rev_signals.append(-1)   # large 4h rally → mean-revert down
    else:             rev_signals.append(0)

    all_signals = trend_signals + rev_signals
    score       = sum(all_signals)
    total       = len(all_signals)
    direction   = "UP" if score > 0 else "DOWN" if score < 0 else "SIDEWAYS"
    confidence  = round(abs(score) / total, 4)

    bull = sum(1 for s in all_signals if s > 0)
    bear = sum(1 for s in all_signals if s < 0)

    return {
        "direction":       direction,
        "confidence":      confidence,
        "score":           score,
        "bull_signals":    bull,
        "bear_signals":    bear,
        "total_signals":   total,
        "predicted_price": p * (1 + ml["slope_pct"] / 100),
        "reasoning":       (
            f"{bull} bullish vs {bear} bearish out of {total} signals "
            f"(trend: {sum(trend_signals):+d}, reversion: {sum(rev_signals):+d}). "
            f"Rule score: {score:+d}. ML slope: {ml['slope_pct']:+.4f}%/candle."
        ),
        "risk": "LOW" if confidence > 0.6 else "MEDIUM" if confidence > 0.35 else "HIGH",
    }


# ── Claude AI Analysis ────────────────────────────────────────────────────────

def claude_analyze(indicators: dict, ml: dict) -> dict:
    """Ask Claude to synthesize indicators and produce a balanced final prediction."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=api_key)

    p = indicators["current_price"]
    # Pre-tally signals so Claude can't ignore them
    bearish_flags = []
    bullish_flags = []
    if p < indicators["ma_short"]:  bearish_flags.append("price < MA9")
    else:                           bullish_flags.append("price > MA9")
    if p < indicators["ma_long"]:   bearish_flags.append("price < MA21")
    else:                           bullish_flags.append("price > MA21")
    if indicators["macd"] < 0:      bearish_flags.append("MACD negative")
    else:                           bullish_flags.append("MACD positive")
    if indicators["rsi"] < 40:      bullish_flags.append(f"RSI {indicators['rsi']:.0f} (near oversold)")
    elif indicators["rsi"] > 60:    bearish_flags.append(f"RSI {indicators['rsi']:.0f} (near overbought)")
    if indicators["price_change_1h"] < 0: bearish_flags.append(f"1h change {indicators['price_change_1h']:+.2f}%")
    else:                                 bullish_flags.append(f"1h change {indicators['price_change_1h']:+.2f}%")
    if indicators["price_change_4h"] < 0: bearish_flags.append(f"4h change {indicators['price_change_4h']:+.2f}%")
    else:                                 bullish_flags.append(f"4h change {indicators['price_change_4h']:+.2f}%")
    if indicators["bb_pct"] < 0.2:  bullish_flags.append("near Bollinger Lower Band")
    elif indicators["bb_pct"] > 0.8: bearish_flags.append("near Bollinger Upper Band")

    # Split signals into trend vs mean-reversion for balanced analysis
    trend_bull, trend_bear, rev_bull, rev_bear = [], [], [], []
    if indicators["ma_short"] > indicators["ma_long"]: trend_bull.append("MA9 > MA21")
    else:                                               trend_bear.append("MA9 < MA21")
    if indicators["macd"] > 0:                         trend_bull.append("MACD positive")
    else:                                               trend_bear.append("MACD negative")
    if indicators["price_change_1h"] > 0:              trend_bull.append(f"1h +{indicators['price_change_1h']:.2f}%")
    else:                                               trend_bear.append(f"1h {indicators['price_change_1h']:.2f}%")
    if ml["direction"] == "UP":                        trend_bull.append(f"ML slope {ml['slope_pct']:+.4f}%")
    else:                                               trend_bear.append(f"ML slope {ml['slope_pct']:+.4f}%")

    if   indicators["rsi"] < 35:   rev_bull.append(f"RSI {indicators['rsi']:.0f} oversold")
    elif indicators["rsi"] > 65:   rev_bear.append(f"RSI {indicators['rsi']:.0f} overbought")
    if   indicators["bb_pct"] < 0.25: rev_bull.append(f"BB {indicators['bb_pct']:.0%} near lower band")
    elif indicators["bb_pct"] > 0.75: rev_bear.append(f"BB {indicators['bb_pct']:.0%} near upper band")
    if   indicators["price_change_1h"] < -0.4: rev_bull.append(f"1h drop {indicators['price_change_1h']:.2f}% → bounce?")
    elif indicators["price_change_1h"] >  0.4: rev_bear.append(f"1h surge {indicators['price_change_1h']:.2f}% → fade?")
    if   indicators["price_change_4h"] < -1.5: rev_bull.append(f"4h drop {indicators['price_change_4h']:.2f}% → exhaustion?")
    elif indicators["price_change_4h"] >  1.5: rev_bear.append(f"4h surge {indicators['price_change_4h']:.2f}% → exhaustion?")

    prompt = f"""You are a strict quantitative crypto analyst making a 15-minute BTC directional call.

TREND SIGNALS (is the current trend up or down?):
  Bullish ({len(trend_bull)}): {', '.join(trend_bull) or 'none'}
  Bearish ({len(trend_bear)}): {', '.join(trend_bear) or 'none'}

MEAN-REVERSION SIGNALS (is price stretched and likely to snap back?):
  Bullish ({len(rev_bull)}): {', '.join(rev_bull) or 'none'}
  Bearish ({len(rev_bear)}): {', '.join(rev_bear) or 'none'}

FULL MARKET DATA:
  Current Price : ${p:,.2f}
  MA(9) / MA(21): ${indicators['ma_short']:,.2f} / ${indicators['ma_long']:,.2f}
  RSI(14)       : {indicators['rsi']:.1f}
  MACD Line     : {indicators['macd']:+.2f}
  BB Position   : {indicators['bb_pct']:.0%} (0%=lower band, 100%=upper band)
  Volume Trend  : {indicators['vol_trend']:.2f}x
  1h / 4h Change: {indicators['price_change_1h']:+.2f}% / {indicators['price_change_4h']:+.2f}%
  ML Slope      : {ml['slope_pct']:+.4f}%/candle ({ml['direction']})

RULES:
- Weigh BOTH groups. A strong mean-reversion signal can override a weak trend signal.
- Do NOT always follow trend — if price is oversold/at lower band after a big drop, UP is valid.
- Confidence = how one-sided the combined signal count is (0.0 = 50/50, 1.0 = all one way).
- Use "SIDEWAYS" only when trend and reversion signals directly cancel each other out.

Respond ONLY with a JSON object (no markdown):
{{
  "direction": "UP" or "DOWN" or "SIDEWAYS",
  "predicted_price": <number>,
  "confidence": <0.0-1.0>,
  "reasoning": "<2-3 sentences explicitly weighing trend vs mean-reversion>",
  "key_signals": ["signal1", "signal2", "signal3"],
  "risk": "LOW" or "MEDIUM" or "HIGH"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    text = re.sub(r"^```[a-z]*\n?", "", response.content[0].text.strip()).rstrip("` \n")
    return json.loads(text)


# ── Display ───────────────────────────────────────────────────────────────────

def render_prediction(indicators: dict, ml: dict, prediction: dict, strategy: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    direction_emoji = {"UP": "📈", "DOWN": "📉", "SIDEWAYS": "➡️"}.get(prediction["direction"], "❓")
    risk_color      = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(prediction["risk"], "white")
    conf_color      = "green" if prediction["confidence"] > 0.65 else "yellow" if prediction["confidence"] > 0.45 else "red"
    strat_label     = "[bold magenta]RULES[/]" if strategy == "rules" else "[bold blue]AI[/]"

    console.print(Panel(
        f"[bold cyan]BTC/USDT · 15m Predictor[/]  {strat_label}  [dim]{now}[/]",
        box=box.DOUBLE_EDGE
    ))

    console.print(f"\n  Current Price : [bold white]${indicators['current_price']:>12,.2f}[/]")
    console.print(f"  ML Forecast   : [bold yellow]${ml['predicted_price']:>12,.2f}[/]  "
                  f"[dim](slope {ml['slope_pct']:+.4f}%/candle  |  ML conf: {ml['confidence']:.1%}  |  ML: {ml['direction']})[/]")
    if "predicted_price" in prediction:
        console.print(f"  Pred Forecast : [bold yellow]${prediction['predicted_price']:>12,.2f}[/]")
    console.print(f"\n  Direction     : {direction_emoji}  [bold]{prediction['direction']}[/]")
    console.print(f"  Confidence    : [{conf_color}]{prediction['confidence']:.1%}[/]")
    console.print(f"  Risk Level    : [{risk_color}]{prediction['risk']}[/]")

    if strategy == "rules":
        console.print(f"  Signal Score  : {prediction['score']:+d}  "
                      f"([green]{prediction['bull_signals']} bull[/] / [red]{prediction['bear_signals']} bear[/] "
                      f"of {prediction['total_signals']})")

    # Indicator table
    table = Table(title="\nKey Indicators", box=box.SIMPLE_HEAVY)
    table.add_column("Indicator", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Signal", justify="center")

    p = indicators["current_price"]
    ma_sig  = lambda ma: "[green]Bullish[/]" if p > ma  else "[red]Bearish[/]"
    rsi_sig = lambda r:  "[red]Overbought[/]" if r > 70 else "[green]Oversold[/]" if r < 30 else \
                         "[yellow]Near OB[/]" if r > 60 else "[yellow]Near OS[/]" if r < 40 else "[white]Neutral[/]"
    bb_sig  = lambda bp: "[green]Near Lower[/]" if bp < 0.2 else "[red]Near Upper[/]" if bp > 0.8 else "[white]Middle[/]"

    table.add_row("MA(9)",      f"${indicators['ma_short']:,.2f}",   ma_sig(indicators["ma_short"]))
    table.add_row("MA(21)",     f"${indicators['ma_long']:,.2f}",    ma_sig(indicators["ma_long"]))
    table.add_row("RSI(14)",    f"{indicators['rsi']:.1f}",          rsi_sig(indicators["rsi"]))
    table.add_row("MACD",       f"{indicators['macd']:+.2f}",        "[green]Bullish[/]" if indicators["macd"] > 0 else "[red]Bearish[/]")
    table.add_row("BB Position",f"{indicators['bb_pct']:.0%}",       bb_sig(indicators["bb_pct"]))
    table.add_row("Volume ×",   f"{indicators['vol_trend']:.2f}x",   "[green]Rising[/]" if indicators["vol_trend"] > 1.1 else "[red]Falling[/]" if indicators["vol_trend"] < 0.9 else "[white]Stable[/]")
    table.add_row("1h Change",  f"{indicators['price_change_1h']:+.2f}%", "[green]▲[/]" if indicators["price_change_1h"] > 0 else "[red]▼[/]")
    table.add_row("4h Change",  f"{indicators['price_change_4h']:+.2f}%", "[green]▲[/]" if indicators["price_change_4h"] > 0 else "[red]▼[/]")
    console.print(table)

    # Reasoning panel
    console.print(Panel(
        f"[italic]{prediction['reasoning']}[/]\n\n"
        + "  ".join(f"[bold cyan]• {s}[/]" for s in prediction.get("key_signals", [])),
        title=f"[bold]{'Claude AI' if strategy == 'ai' else 'Rule Engine'} Analysis[/]",
        border_style="blue" if strategy == "ai" else "magenta"
    ))
    console.print()


# ── XGBoost Prediction ────────────────────────────────────────────────────────

def xgb_predict(indicators: dict, ml: dict) -> dict | None:
    """Load saved XGBoost model and predict direction + win probability."""
    import pickle
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        p            = indicators["current_price"]
        ma9          = indicators["ma_short"]
        ma21         = indicators["ma_long"]
        price_vs_ma9  = (p - ma9)  / ma9  * 100
        price_vs_ma21 = (p - ma21) / ma21 * 100
        ma_cross      = (ma9 - ma21) / ma21 * 100

        row = [
            indicators["rsi"],
            indicators["macd"],
            indicators["bb_pct"],
            price_vs_ma9,
            price_vs_ma21,
            ma_cross,
            indicators["vol_trend"],
            indicators["price_change_1h"],
            indicators["price_change_4h"],
            ml["slope_pct"],
            ml["confidence"],
            1 if ml["direction"] == "UP" else 0,
            0,   # strategy_ai = 0 (this is the XGB strategy itself)
        ]
        import numpy as _np
        X    = _np.array([row])
        prob = float(model.predict_proba(X)[0][1])   # prob of WIN
        direction = "UP" if prob >= 0.5 else "DOWN"

        return {
            "direction":       direction,
            "confidence":      round(prob, 4),
            "predicted_price": indicators["current_price"],
            "reasoning":       f"XGBoost win probability: {prob:.1%}",
            "key_signals":     [f"Win prob: {prob:.1%}", f"Direction: {direction}"],
            "risk":            "LOW" if abs(prob - 0.5) > 0.2 else "MEDIUM" if abs(prob - 0.5) > 0.1 else "HIGH",
        }
    except Exception as e:
        console.print(f"[dim yellow]XGB predict failed: {e}[/]")
        return None


# ── Win-Rate Tracking ─────────────────────────────────────────────────────────

def load_log() -> list:
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE) as f:
            return json.load(f)
    return []


def save_log(log: list):
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(log, f, indent=2)


def record_prediction(direction: str, entry_price: float, confidence: float,
                      strategy: str, indicators: dict, ml: dict):
    log = load_log()
    log.append({
        "id":           len(log) + 1,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "strategy":     strategy,          # "ai" or "rules"
        "direction":    direction,
        "entry_price":  entry_price,
        "confidence":   confidence,
        "exit_price":   None,
        "price_delta":  None,              # exit - entry ($)
        "price_delta_pct": None,           # exit - entry (%)
        "result":       None,              # "WIN" | "LOSS" | "PUSH"
        "kalshi_pnl":   None,             # simulated Kalshi P&L per $1 contract
        "features": {                      # raw snapshot for ML training
            # ── Core indicators ──
            "rsi":              round(indicators["rsi"], 4),
            "macd":             round(indicators["macd"], 4),
            "macd_signal":      round(indicators["macd_signal"], 4),
            "macd_hist":        round(indicators["macd_hist"], 4),
            "bb_pct":           round(indicators["bb_pct"], 4),
            "bb_width":         round(indicators["bb_width"], 4),
            "stoch_k":          round(indicators["stoch_k"], 4),
            "stoch_d":          round(indicators["stoch_d"], 4),
            # ── Moving averages (relative) ──
            "ma_short":         round(indicators["ma_short"], 2),
            "ma_long":          round(indicators["ma_long"], 2),
            "price_vs_ma9":     round((entry_price - indicators["ma_short"]) / indicators["ma_short"] * 100, 4),
            "price_vs_ma21":    round((entry_price - indicators["ma_long"])  / indicators["ma_long"]  * 100, 4),
            "ma_cross":         round((indicators["ma_short"] - indicators["ma_long"]) / indicators["ma_long"] * 100, 4),
            # ── Volatility ──
            "atr":              round(indicators["atr"], 2),
            "atr_pct":          round(indicators["atr_pct"], 4),
            "range_1h":         round(indicators["range_1h"], 4),
            "range_4h":         round(indicators["range_4h"], 4),
            # ── Volume ──
            "vol_trend":        round(indicators["vol_trend"], 4),
            "vol_spike":        round(indicators["vol_spike"], 4),
            # ── Multi-timeframe momentum ──
            "price_change_15m": round(indicators["price_change_15m"], 4),
            "price_change_30m": round(indicators["price_change_30m"], 4),
            "price_change_1h":  round(indicators["price_change_1h"], 4),
            "price_change_2h":  round(indicators["price_change_2h"], 4),
            "price_change_4h":  round(indicators["price_change_4h"], 4),
            # ── Candle patterns ──
            "body_pct":         round(indicators["body_pct"], 4),
            "upper_wick":       round(indicators["upper_wick"], 4),
            "lower_wick":       round(indicators["lower_wick"], 4),
            "consec_green":     indicators["consec_green"],
            "consec_red":       indicators["consec_red"],
            # ── Time features ──
            "hour_of_day":      indicators["hour_of_day"],
            "day_of_week":      indicators["day_of_week"],
            # ── ML regression ──
            "ml_slope_pct":     round(ml["slope_pct"], 6),
            "ml_confidence":    round(ml["confidence"], 4),
            "ml_direction":     1 if ml["direction"] == "UP" else -1,
            # ── Raw OHLCV (last 30 candles for LSTM) ──
            "raw_candles":      indicators["raw_candles"],
        },
    })
    save_log(log)


def resolve_pending(current_price: float):
    """Resolve predictions made >= 15 min ago. Returns newly resolved entries."""
    log = load_log()
    now = datetime.now(timezone.utc)
    resolved = []

    for entry in log:
        if entry["result"] is not None:
            continue
        made_at     = datetime.fromisoformat(entry["timestamp"])
        age_minutes = (now - made_at).total_seconds() / 60
        if age_minutes < 15:
            continue

        delta     = current_price - entry["entry_price"]
        delta_pct = delta / entry["entry_price"] * 100

        if abs(delta) < 0.01:
            entry["result"]     = "PUSH"
            entry["kalshi_pnl"] = 0.0
        elif entry["direction"] == "UP":
            entry["result"]     = "WIN" if delta > 0 else "LOSS"
        else:
            entry["result"]     = "WIN" if delta < 0 else "LOSS"

        # Simulated Kalshi P&L: buy YES at confidence price, $1 payout
        # WIN  → profit = $1 - confidence (e.g. paid 60¢, win 40¢)
        # LOSS → loss   = -confidence      (e.g. paid 60¢, lose 60¢)
        if entry["result"] == "WIN":
            entry["kalshi_pnl"] = round(KALSHI_CONTRACT - entry["confidence"], 4)
        elif entry["result"] == "LOSS":
            entry["kalshi_pnl"] = round(-entry["confidence"], 4)

        entry["exit_price"]     = current_price
        entry["price_delta"]    = round(delta, 2)
        entry["price_delta_pct"]= round(delta_pct, 4)
        resolved.append(entry)

    save_log(log)
    return resolved


def compute_stats(log: list) -> dict:
    resolved = [e for e in log if e["result"] in ("WIN", "LOSS", "PUSH")]
    wins     = [e for e in resolved if e["result"] == "WIN"]
    losses   = [e for e in resolved if e["result"] == "LOSS"]
    pushes   = [e for e in resolved if e["result"] == "PUSH"]
    pending  = [e for e in log      if e["result"] is None]

    decidable = len(wins) + len(losses)
    win_rate  = len(wins) / decidable if decidable > 0 else 0.0

    # Average move sizes (guard against old log entries missing these fields)
    win_deltas  = [abs(e["price_delta"]) for e in wins   if e.get("price_delta") is not None]
    loss_deltas = [abs(e["price_delta"]) for e in losses if e.get("price_delta") is not None]
    avg_win_delta  = float(np.mean(win_deltas))  if win_deltas  else 0.0
    avg_loss_delta = float(np.mean(loss_deltas)) if loss_deltas else 0.0

    # Simulated Kalshi P&L
    kalshi_pnl_total = sum(e["kalshi_pnl"] for e in resolved if e.get("kalshi_pnl") is not None)

    # Confidence calibration brackets
    brackets = [
        ("<50%",  0.00, 0.50),
        ("50-60%",0.50, 0.60),
        ("60-70%",0.60, 0.70),
        ("70-80%",0.70, 0.80),
        ("80%+",  0.80, 1.01),
    ]
    calibration = []
    for label, lo, hi in brackets:
        bucket = [e for e in resolved if lo <= (e.get("confidence") or 0) < hi and e["result"] != "PUSH"]
        if bucket:
            w = sum(1 for e in bucket if e["result"] == "WIN")
            calibration.append({"label": label, "n": len(bucket), "wins": w,
                                 "win_rate": w / len(bucket)})

    # Per-strategy breakdown
    strategies = {}
    for s in ("ai", "rules", "xgb"):
        sub = [e for e in resolved if e.get("strategy", "ai") == s and e["result"] != "PUSH"]
        if sub:
            w = sum(1 for e in sub if e["result"] == "WIN")
            strategies[s] = {"n": len(sub), "wins": w, "win_rate": w / len(sub)}

    return {
        "total": len(resolved), "wins": len(wins), "losses": len(losses),
        "pushes": len(pushes),  "pending": len(pending),
        "win_rate": win_rate,
        "avg_win_delta": avg_win_delta,
        "avg_loss_delta": avg_loss_delta,
        "kalshi_pnl": kalshi_pnl_total,
        "calibration": calibration,
        "strategies": strategies,
        "log": log,
    }


def render_stats(stats: dict):
    wr_color  = "green" if stats["win_rate"] >= 0.55 else "yellow" if stats["win_rate"] >= 0.45 else "red"
    pnl_color = "green" if stats["kalshi_pnl"] >= 0 else "red"

    console.print(Panel(
        f"[bold]Resolved:[/] {stats['total']}   "
        f"[green]Wins:[/] {stats['wins']}   "
        f"[red]Losses:[/] {stats['losses']}   "
        f"[dim]Pushes:[/] {stats['pushes']}   "
        f"[dim]Pending:[/] {stats['pending']}\n\n"
        f"[bold]Win Rate:[/] [{wr_color}]{stats['win_rate']:.1%}[/]  "
        f"  [bold]Kalshi P&L:[/] [{pnl_color}]{stats['kalshi_pnl']:+.2f}¢[/] per $1 contract\n"
        f"[dim]Avg win move: ${stats['avg_win_delta']:,.0f}  |  Avg loss move: ${stats['avg_loss_delta']:,.0f}[/]",
        title="[bold cyan]Prediction Win Rate[/]",
        border_style="cyan",
    ))

    # Confidence calibration table
    if stats["calibration"]:
        cal_table = Table(title="Confidence Calibration", box=box.SIMPLE_HEAVY)
        cal_table.add_column("Bucket",   style="cyan")
        cal_table.add_column("N",        justify="right")
        cal_table.add_column("Win Rate", justify="right")
        cal_table.add_column("Calibrated?", justify="center")
        for row in stats["calibration"]:
            wr = row["win_rate"]
            wrc = "green" if wr >= 0.55 else "yellow" if wr >= 0.45 else "red"
            mid_conf = {"<50%": 0.45, "50-60%": 0.55, "60-70%": 0.65,
                        "70-80%": 0.75, "80%+": 0.85}.get(row["label"], 0.5)
            cal_ok = "✅" if abs(wr - mid_conf) < 0.10 else "⚠️"
            cal_table.add_row(row["label"], str(row["n"]),
                              f"[{wrc}]{wr:.1%}[/]", cal_ok)
        console.print(cal_table)

    # Strategy comparison table
    if stats["strategies"]:
        st_table = Table(title="Strategy Comparison", box=box.SIMPLE_HEAVY)
        st_table.add_column("Strategy", style="cyan")
        st_table.add_column("N",        justify="right")
        st_table.add_column("Win Rate", justify="right")
        for s, d in stats["strategies"].items():
            wrc = "green" if d["win_rate"] >= 0.55 else "yellow" if d["win_rate"] >= 0.45 else "red"
            label = (
                "[bold blue]AI[/]"      if s == "ai"    else
                "[bold yellow]XGB[/]"   if s == "xgb"   else
                "[bold magenta]Rules[/]"
            )
            st_table.add_row(label, str(d["n"]), f"[{wrc}]{d['win_rate']:.1%}[/]")
        console.print(st_table)

    # Recent predictions log
    if not stats["log"]:
        return
    log_table = Table(title="Recent Predictions (last 20)", box=box.SIMPLE_HEAVY)
    log_table.add_column("#",        style="dim",  width=4)
    log_table.add_column("Time UTC", style="cyan", width=16)
    log_table.add_column("Strat",    justify="center", width=5)
    log_table.add_column("Dir",      justify="center")
    log_table.add_column("Conf",     justify="right")
    log_table.add_column("Entry $",  justify="right")
    log_table.add_column("Exit $",   justify="right")
    log_table.add_column("Δ$",       justify="right")
    log_table.add_column("Result",   justify="center")
    log_table.add_column("K P&L",    justify="right")

    for e in reversed(stats["log"][-20:]):
        res_str = (
            "[green]WIN[/]"  if e["result"] == "WIN"  else
            "[red]LOSS[/]"   if e["result"] == "LOSS" else
            "[dim]PUSH[/]"   if e["result"] == "PUSH" else
            "[yellow]...[/]"
        )
        pd_val    = e.get("price_delta")
        delta_str = (f"[green]+${pd_val:,.0f}[/]" if (pd_val or 0) > 0
                     else f"[red]-${abs(pd_val):,.0f}[/]" if (pd_val or 0) < 0
                     else "—") if pd_val is not None else "—"
        pnl_str   = (f"[green]+{e['kalshi_pnl']:.2f}¢[/]" if (e["kalshi_pnl"] or 0) > 0
                     else f"[red]{e['kalshi_pnl']:.2f}¢[/]" if (e["kalshi_pnl"] or 0) < 0
                     else "—") if e.get("kalshi_pnl") is not None else "—"
        strat_str = (
            "[blue]AI[/]"      if e.get("strategy") == "ai"    else
            "[yellow]XGB[/]"   if e.get("strategy") == "xgb"   else
            "[magenta]R[/]"
        )
        dir_str   = "[green]UP[/]" if e["direction"] == "UP" else "[red]DN[/]"

        log_table.add_row(
            str(e["id"]),
            e["timestamp"][5:16].replace("T", " "),
            strat_str, dir_str,
            f"{e['confidence']:.0%}",
            f"${e['entry_price']:,.0f}",
            f"${e['exit_price']:,.0f}" if e["exit_price"] else "—",
            delta_str,
            res_str,
            pnl_str,
        )
    console.print(log_table)


# ── Auto-Retraining ───────────────────────────────────────────────────────────

RETRAIN_EVERY   = 20          # retrain after this many new resolved predictions
RETRAIN_TRACKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_retrain_tracker.json")

def _retrain_count() -> int:
    """Return the resolved-count at the time of the last retrain (0 if never)."""
    if os.path.exists(RETRAIN_TRACKER):
        with open(RETRAIN_TRACKER) as f:
            return json.load(f).get("last_resolved_count", 0)
    return 0

def _save_retrain_count(n: int):
    with open(RETRAIN_TRACKER, "w") as f:
        json.dump({"last_resolved_count": n, "retrained_at": datetime.now(timezone.utc).isoformat()}, f)

def auto_retrain_if_needed(log: list):
    """Retrain XGBoost if RETRAIN_EVERY new predictions have resolved since last train."""
    resolved_with_features = [e for e in log if e["result"] in ("WIN", "LOSS") and e.get("features")]
    current_count = len(resolved_with_features)
    last_count    = _retrain_count()

    if current_count - last_count < RETRAIN_EVERY:
        return   # not enough new data yet

    console.print(f"\n[bold yellow]⟳ Auto-retraining XGBoost[/] "
                  f"[dim]({current_count - last_count} new samples since last train, "
                  f"{current_count} total)...[/]")
    try:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_model.py")],
            capture_output=True, text=True, timeout=60
        )
        # Print key lines only (skip warnings)
        for line in result.stdout.splitlines():
            if any(k in line for k in ["Loaded", "Train Accuracy", "Model saved", "Win Rate", "Pred Prob"]):
                console.print(f"  [dim]{line.strip()}[/]")
        if result.returncode == 0:
            _save_retrain_count(current_count)
            console.print("[green]✓ XGBoost retrained and saved.[/]\n")
        else:
            console.print(f"[red]Retrain failed:[/] {result.stderr[:200]}\n")
    except Exception as e:
        console.print(f"[red]Retrain error:[/] {e}\n")


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run(interval_minutes: int, once: bool, no_ai: bool):
    if not no_ai and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]Error:[/] Set ANTHROPIC_API_KEY or use --no-ai for rule-based mode.")
        sys.exit(1)

    mode_label = "[bold magenta]Rule-Based only[/]" if no_ai else "[bold blue]AI[/] + [bold magenta]Rules[/] (both)"
    console.print(f"[bold green]Bitcoin Predictor started.[/] Mode: {mode_label}  "
                  f"{'Single run' if once else f'Every {interval_minutes} min'}.\n")

    while True:
        try:
            console.print("[dim]Fetching market data...[/]")
            df            = fetch_klines()
            indicators    = compute_indicators(df)
            ml            = ml_predict(df)
            current_price = indicators["current_price"]

            # Resolve any predictions made >= 15 min ago
            resolved = resolve_pending(current_price)
            for r in resolved:
                color = "green" if r["result"] == "WIN" else "red" if r["result"] == "LOSS" else "dim"
                pnl   = f"  K P&L: [{color}]{r['kalshi_pnl']:+.2f}¢[/]" if r.get("kalshi_pnl") is not None else ""
                console.print(
                    f"  [dim]Resolved #{r['id']} ({r.get('strategy','?')}):[/] "
                    f"{'UP' if r['direction']=='UP' else 'DOWN'}  "
                    f"${r['entry_price']:,.0f} → ${r['exit_price']:,.0f}  "
                    f"([{color}]{r['price_delta']:+,.0f}[/])  [{color}]{r['result']}[/]{pnl}"
                )

            # Auto-retrain XGBoost if enough new data has accumulated
            auto_retrain_if_needed(load_log())

            # Always record rule-based prediction
            rule_pred = rule_based_predict(indicators, ml)
            if rule_pred["direction"] in ("UP", "DOWN"):
                record_prediction(rule_pred["direction"], current_price,
                                  rule_pred["confidence"], "rules", indicators, ml)

            # Always record XGBoost prediction (if model exists)
            xgb_pred = xgb_predict(indicators, ml)
            if xgb_pred and xgb_pred["direction"] in ("UP", "DOWN"):
                record_prediction(xgb_pred["direction"], current_price,
                                  xgb_pred["confidence"], "xgb", indicators, ml)

            # Also record AI prediction (unless --no-ai)
            if no_ai:
                prediction = xgb_pred or rule_pred
                strategy   = "xgb" if xgb_pred else "rules"
            else:
                console.print("[dim]Consulting Claude AI...[/]")
                prediction = claude_analyze(indicators, ml)
                strategy   = "ai"
                if prediction["direction"] in ("UP", "DOWN"):
                    record_prediction(prediction["direction"], current_price,
                                      prediction["confidence"], "ai", indicators, ml)

            console.clear()
            render_prediction(indicators, ml, prediction, strategy)

            stats = compute_stats(load_log())
            if stats["total"] > 0 or stats["pending"] > 0:
                render_stats(stats)

        except requests.exceptions.RequestException as e:
            console.print(f"[red]Network error:[/] {e}")
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            import traceback; traceback.print_exc()

        if once:
            break

        console.print(f"[dim]Next update in {interval_minutes} min... (Ctrl+C to stop)[/]")
        time.sleep(interval_minutes * 60)


def show_stats_only():
    stats = compute_stats(load_log())
    render_stats(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC 15m predictor with win-rate tracking")
    parser.add_argument("--interval", type=int, default=15,
                        help="Update interval in minutes (default: 15)")
    parser.add_argument("--once",   action="store_true", help="Run once and exit")
    parser.add_argument("--no-ai",  action="store_true", help="Rule-based only, no Claude API needed")
    parser.add_argument("--stats",  action="store_true", help="Show stats and exit (no API calls)")
    args = parser.parse_args()

    if args.stats:
        show_stats_only()
    else:
        run(args.interval, args.once, args.no_ai)
