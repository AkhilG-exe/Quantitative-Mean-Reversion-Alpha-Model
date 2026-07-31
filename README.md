# Quantitative Mean-Reversion Alpha Model (`quant-alpha-cpp`)

A C++ quantitative trading system built from scratch. Combines mathematical feature engineering (Z-Score, RSI, Log Returns) with L2-Regularized Logistic Regression to forecast price direction and evaluate trading performance via an event-driven backtesting engine.

## Key Features

- **Custom Quant Features**:
  - **Rolling Z-Score**: Quantifies statistical distance from historical moving average.
  - **Relative Strength Index (RSI)**: Normalizes momentum divergence.
  - **Log Returns**: Measures continuous asset return dynamics.
- **Machine Learning Engine**: L2-Regularized (Ridge) Logistic Regression optimizing cross-entropy loss via gradient descent.
- **Backtest Framework**: Simulates position executions, computing key portfolio analytics (**Sharpe Ratio**, **Cumulative Return**, and **Max Drawdown**).

## Mathematical Formulation

### 1. Feature Metrics
Rolling Z-score of price window $P$:
$$Z = \frac{P_t - \mu_w}{\sigma_w}$$

### 2. Regularized Objective Function
Minimizes binary cross-entropy loss with Ridge regularization:
$$L(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \Vert{}w\Vert{}^2$$

## Compilation & Execution

```bash
# Compile with C++17 standard and full O3 performance flags
g++ -O3 -std=c++17 main.cpp -o quant_alpha

# Execute backtest pipeline
./quant_alpha
