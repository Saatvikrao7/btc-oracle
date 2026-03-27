# btc-oracle

Real-time Bitcoin price predictor that runs 3 competing strategies, tracks win rates, simulates Kalshi contract P&L, and auto-retrains an XGBoost model as it collects data.

## How It Works

Every 15 minutes, the system:

1. **Fetches** BTC/USD price data from Kraken (free public API, no key needed)
2. **Computes** 37 technical indicators (RSI, MACD, Bollinger Bands, ATR, Stochastic, volume trends, multi-timeframe momentum, candle patterns, and more)
3. **Runs 3 strategies** in parallel to predict the next 15-minute direction:
   - **Rules** — Signal-based scoring engine (4 trend + 4 mean-reversion signals, majority vote)
   - **AI** — Claude API analysis with explicit bullish/bearish signal tallying
   - **XGB** — XGBoost model trained on historical predictions with auto-retraining
4. **Logs** every prediction with full feature snapshots (37 indicators + 30 raw OHLCV candles)
5. **Resolves** predictions after 15 min, scoring WIN/LOSS and simulated Kalshi P&L
6. **Auto-retrains** XGBoost every 20 resolved predictions so the model learns from its mistakes

## Live Results (500+ predictions)

| Strategy | N | Win Rate | Kalshi P&L |
|----------|-----|----------|------------|
| AI | 201 | 55.2% | -0.36¢ |
| XGB | 90 | 54.4% | +3.62¢ |
| Rules | 212 | 51.4% | +55.21¢ |

> XGB recent 30 predictions: **63.3% win rate** (trending up after auto-retrain)

## Features

- **3 competing strategies** running side-by-side on every candle
- **Kalshi P&L simulation** — models contract pricing based on prediction confidence
- **Confidence calibration** — tracks whether 60% confidence predictions actually win 60% of the time
- **Auto-retraining** — XGBoost retrains every 20 resolved predictions
- **37 features per prediction** logged for ML training:
  - Momentum: RSI, MACD (line/signal/histogram), Stochastic %K/%D
  - Moving Averages: MA9, MA21, crossover gap, price distance
  - Volatility: ATR, BB width, BB position
  - Volume: trend ratio, spike detection
  - Multi-timeframe: 15m, 30m, 1h, 2h, 4h price changes
  - Candle patterns: body %, wick ratios, consecutive direction
  - Time: hour of day, day of week
  - Raw OHLCV: last 30 candles for LSTM training
- **Rich terminal UI** with colored tables, analysis boxes, and live stats

## Quick Start

```bash
# Clone
git clone https://github.com/Saatvikrao7/btc-oracle.git
cd btc-oracle

# Install dependencies
pip install requests numpy anthropic xgboost scikit-learn python-dotenv

# Rules + XGB only (no API key needed)
python BTC_Predictor.py --no-ai

# All 3 strategies (needs Anthropic API key)
echo 'ANTHROPIC_API_KEY=your-key-here' > .env
python BTC_Predictor.py
```

## Usage

```bash
# Run continuously (every 15 min)
python BTC_Predictor.py

# Single prediction
python BTC_Predictor.py --once

# Rules-only mode (no API calls)
python BTC_Predictor.py --no-ai

# View stats without making predictions
python BTC_Predictor.py --stats

# Retrain XGBoost manually
python btc_xgb_trainer.py
```

## Architecture

```
BTC_Predictor.py          # Main script — data fetching, indicators, strategies, display
btc_xgb_trainer.py        # Standalone XGBoost trainer with cross-validation
btc_predictions_log.json  # Prediction history with full feature snapshots (auto-created)
btc_xgb_model.pkl         # Trained XGBoost model (auto-created)
btc_retrain_tracker.json  # Auto-retrain state tracker (auto-created)
.env                      # API key (not committed)
```

## Key Insights

- **Rules beats AI on P&L** — lower confidence = cheaper Kalshi contracts = bigger payouts on wins
- **XGB confidence was inverted** — low confidence predictions won more often (fixed via retraining with calibration)
- **3 correlated MA signals** caused a persistent DOWN bias — fixed by deduplicating to 1 MA signal + adding mean-reversion signals
- **Auto-retraining matters** — XGB went from 50% to 63% win rate after learning from bear market data

## Roadmap

- [ ] LSTM neural network on raw candle sequences (needs 1000+ samples)
- [ ] Kalshi API integration for real trading
- [ ] Multi-asset support (ETH, SOL)
- [ ] Ensemble strategy (weighted vote across all 3)
- [ ] Web dashboard for monitoring

## Data Source

All price data comes from the [Kraken public API](https://docs.kraken.com/rest/) — free, no authentication required.

---

Built with Claude Code
