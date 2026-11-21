# Systematic Trading Strategies on Synthetic Indices: Core Components

Code for the BICT Honours research project *The Analysis of Systematic Trading Strategies in Synthetic Indices Markets for the Development of a Trading Decision Model* (University of Mpumalanga).

The study back-tests trend-following, breakout-and-retest and momentum strategies on Deriv's Volatility 10 and Volatility 75 indices (one-minute MT5 data, 2019 to 2025). One strategy per index is selected on in-sample data (2019 to 2023) and evaluated on unseen data (2024 to 2025). The selected strategy drives a rule-based trade guidance model.

## Components

| File | Purpose |
|---|---|
| `data_preparation.py` | Load the MT5 CSV export and build higher timeframes |
| `market_structure.py` | Swing points, order blocks, H4 bias (Dow Theory) and setup detection |
| `strategy_engine.py` | 15-minute confirmation, entry/stop/target, exit simulation, back-test runner |
| `evaluation_and_selection.py` | Profit, drawdown, return-to-drawdown, in-sample selection, random-entry benchmark |
| `trade_guidance.py` | Rule-based decision model (WAIT, SETUP, ENTER, NO TRADE, EXIT) |
| `guidance_validation.py` | Validation of the model against the back-test, including the look-ahead test |

## Notes
- Imports, project settings (split date, risk, grid, instrument paths), exploratory charts and the main run order are not included. The files above assume those are defined in the notebook.
- The price data is not included. It was exported from MetaTrader 5 using a Deriv demo account.
- This is a research artefact, not a live trading system and not financial advice.
