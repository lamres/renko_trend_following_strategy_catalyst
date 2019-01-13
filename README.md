# renko_trend_following_strategy_catalyst
Example of adaptive trend following strategy based on Renko. This article describes the strategy https://hackernoon.com/adaptive-trend-following-trading-strategy-based-on-renko-9248bf83554

Article about optimizer script
https://medium.freecodecamp.org/bayesian-optimization-in-trading-4fb918fc52a7

This strategy uses Catalyst framework for backtesting https://enigma.co/catalyst/beginner-tutorial.html

### Project contains:

renko_trend_following.py - main file. You should execute this file by python in Catalyst environment.

perf_TradingPair(452516 [eth_btc]).csv - you get this file when the main script is executed. The file contains basic stats of performance.

perf_analysis_pyfolio.ipynb - this ipython-notebook carries out an advanced analytics using csv-file.

pyrenko.py - necessary file to analysis. You can find the latest version here https://github.com/quantroom-pro/pyrenko
