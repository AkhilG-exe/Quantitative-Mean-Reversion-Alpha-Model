#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <algorithm>

// --- QUANT UTILS & FEATURE ENGINEERING ---

struct MarketData {
    double price;
    double log_return;
    double z_score;
    double rsi;
};

// Compute Relative Strength Index (RSI)
double compute_rsi(const std::vector<double>& prices, int period = 14) {
    if (prices.size() < static_cast<size_t>(period + 1)) return 50.0;
    
    double gains = 0.0, losses = 0.0;
    for (size_t i = prices.size() - period; i < prices.size(); ++i) {
        double diff = prices[i] - prices[i - 1];
        if (diff >= 0) gains += diff;
        else losses -= diff;
    }
    
    if (losses == 0) return 100.0;
    double rs = (gains / period) / (losses / period);
    return 100.0 - (100.0 / (1.0 + rs));
}

// Compute Rolling Z-Score (Mean Reversion Metric)
double compute_zscore(const std::vector<double>& prices, int window = 20) {
    if (prices.size() < static_cast<size_t>(window)) return 0.0;
    
    auto start = prices.end() - window;
    double sum = std::accumulate(start, prices.end(), 0.0);
    double mean = sum / window;

    double sq_sum = 0.0;
    for (auto it = start; it != prices.end(); ++it) {
        sq_sum += (*it - mean) * (*it - mean);
    }
    double stdev = std::sqrt(sq_sum / window);
    
    return (stdev == 0) ? 0.0 : (prices.back() - mean) / stdev;
}

// --- MACHINE LEARNING: L2-REGULARIZED LOGISTIC REGRESSION ---

class QuantLogisticRegression {
private:
    std::vector<double> weights;
    double bias;
    double lr;
    double lambda; // L2 Regularization parameter

    double sigmoid(double z) const {
        return 1.0 / (1.0 + std::exp(-z));
    }

public:
    QuantLogisticRegression(int num_features, double learning_rate = 0.01, double l2_penalty = 0.01)
        : weights(num_features, 0.0), bias(0.0), lr(learning_rate), lambda(l2_penalty) {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int epochs = 2000) {
        size_t n_samples = X.size();
        size_t n_features = weights.size();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<double> dw(n_features, 0.0);
            double db = 0.0;

            for (size_t i = 0; i < n_samples; ++i) {
                double linear_model = bias;
                for (size_t j = 0; j < n_features; ++j) {
                    linear_model += X[i][j] * weights[j];
                }

                double y_pred = sigmoid(linear_model);
                double error = y_pred - y[i];

                for (size_t j = 0; j < n_features; ++j) {
                    dw[j] += error * X[i][j];
                }
                db += error;
            }

            // Update parameters with L2 Regularization (Ridge)
            for (size_t j = 0; j < n_features; ++j) {
                weights[j] -= lr * ((dw[j] / n_samples) + (lambda * weights[j]));
            }
            bias -= lr * (db / n_samples);
        }
    }

    double predict_proba(const std::vector<double>& x) const {
        double linear_model = bias;
        for (size_t j = 0; j < weights.size(); ++j) {
            linear_model += x[j] * weights[j];
        }
        return sigmoid(linear_model);
    }

    int predict(const std::vector<double>& x, double threshold = 0.5) const {
        return predict_proba(x) >= threshold ? 1 : 0;
    }

    const std::vector<double>& get_weights() const { return weights; }
};

// --- BACKTESTING ENGINE ---

void run_backtest(const std::vector<double>& prices, const std::vector<int>& signals, int offset) {
    double capital = 100000.0; // $100k starting capital
    double initial_capital = capital;
    int position = 0; // 1 = Long, -1 = Short, 0 = Cash
    
    std::vector<double> equity_curve;
    equity_curve.push_back(capital);

    for (size_t i = offset; i < signals.size(); ++i) {
        double return_pct = (prices[i] - prices[i - 1]) / prices[i - 1];
        
        // Strategy Return
        double strategy_return = position * return_pct;
        capital *= (1.0 + strategy_return);
        equity_curve.push_back(capital);

        // Update position for next bar based on ML signal
        // Signal 1 = Long (1), Signal 0 = Short (-1)
        position = (signals[i] == 1) ? 1 : -1;
    }

    // Performance Metrics
    double total_return = (capital - initial_capital) / initial_capital * 100.0;
    
    // Sharpe Ratio Calculation
    std::vector<double> daily_returns;
    for (size_t i = 1; i < equity_curve.size(); ++i) {
        daily_returns.push_back((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]);
    }

    double mean_ret = std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0) / daily_returns.size();
    double var = 0.0;
    for (double r : daily_returns) var += (r - mean_ret) * (r - mean_ret);
    double stdev = std::sqrt(var / daily_returns.size());
    double sharpe = (stdev > 0) ? (mean_ret / stdev) * std::sqrt(252) : 0.0; // Annualized

    // Max Drawdown
    double peak = equity_curve[0];
    double max_dd = 0.0;
    for (double p : equity_curve) {
        if (p > peak) peak = p;
        double dd = (peak - p) / peak;
        if (dd > max_dd) max_dd = dd;
    }

    std::cout << "\n=====================================================" << std::endl;
    std::cout << "                 BACKTEST RESULTS                    " << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "Initial Capital  : $" << std::fixed << std::setprecision(2) << initial_capital << std::endl;
    std::cout << "Final Capital    : $" << capital << std::endl;
    std::cout << "Total Return     : " << total_return << "%" << std::endl;
    std::cout << "Annualized Sharpe: " << std::setprecision(2) << sharpe << std::endl;
    std::cout << "Max Drawdown     : " << (max_dd * 100.0) << "%" << std::endl;
    std::cout << "=====================================================\n" << std::endl;
}

int main() {
    // Generate Synthetic Mean-Reverting Asset Price Series (Ornstein-Uhlenbeck style simulation)
    std::vector<double> prices = {100.0};
    double mean_price = 100.0;
    double theta = 0.15; // Mean reversion speed
    double sigma = 1.2;  // Volatility

    srand(42);
    for (int i = 1; i < 500; ++i) {
        double dW = ((double)rand() / RAND_MAX * 2.0 - 1.0);
        double prev = prices.back();
        double next_price = prev + theta * (mean_price - prev) + sigma * dW;
        prices.push_back(next_price);
    }

    // Build Feature Matrix (X) and Target Vector (y)
    std::vector<std::vector<double>> X;
    std::vector<int> y;

    int lookback = 20;
    for (size_t i = lookback; i < prices.size() - 1; ++i) {
        std::vector<double> current_prices(prices.begin(), prices.begin() + i + 1);
        
        double z_score = compute_zscore(current_prices, 20);
        double rsi = compute_rsi(current_prices, 14);
        double log_ret = std::log(prices[i] / prices[i - 1]);

        X.push_back({z_score, (rsi - 50.0) / 50.0, log_ret}); // Normalized features

        // Target: Direction of next bar (1 = Up, 0 = Down)
        int target = (prices[i + 1] > prices[i]) ? 1 : 0;
        y.push_back(target);
    }

    // Train/Test Split (80% Train, 20% Out-of-Sample Test)
    size_t train_size = static_cast<size_t>(X.size() * 0.8);
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);

    QuantLogisticRegression model(3, 0.05, 0.001);
    model.fit(X_train, y_train, 3000);

    // Generate Signal Vectors across full dataset
    std::vector<int> signals;
    for (size_t i = 0; i < X.size(); ++i) {
        signals.push_back(model.predict(X[i]));
    }

    // Backtest Strategy
    run_backtest(prices, signals, lookback);

    return 0;
}
