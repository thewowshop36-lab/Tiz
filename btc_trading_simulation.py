#!/usr/bin/env python3
"""
Bitcoin Price Simulation & Golden Cross Trading Algorithm.

Simulates 60 days of Bitcoin price data using Geometric Brownian Motion (GBM),
calculates 7-day and 30-day Moving Averages using pandas, and executes a
Golden Cross trading strategy.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def simulate_bitcoin_prices(
    days: int = 60,
    initial_price: float = 60000.0,
    annual_drift: float = 0.15,
    annual_volatility: float = 0.65,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Simulates daily Bitcoin price data using Geometric Brownian Motion (GBM).

    :param days: Number of days to simulate.
    :param initial_price: Starting price of BTC in USD.
    :param annual_drift: Expected annual return (drift).
    :param annual_volatility: Annualized volatility (standard deviation of log returns).
    :param seed: Random seed for reproducibility.
    :return: DataFrame with Date and Close price.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / 365.0  # Daily time step assuming 365 trading days/year for crypto
    daily_drift = (annual_drift - 0.5 * (annual_volatility**2)) * dt
    daily_vol = annual_volatility * np.sqrt(dt)

    # Generate daily random shocks
    shocks = np.random.normal(loc=0.0, scale=1.0, size=days)
    daily_log_returns = daily_drift + daily_vol * shocks

    # Calculate price series
    prices = [initial_price]
    for r in daily_log_returns:
        prices.append(prices[-1] * np.exp(r))

    # Exclude initial seed price to get exactly `days` data points
    prices = prices[1:]

    start_date = datetime.now().date() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    df = pd.DataFrame(
        {
            "Day": np.arange(1, days + 1),
            "Date": dates,
            "Price": np.round(prices, 2),
        }
    )
    return df


def calculate_moving_averages(
    df: pd.DataFrame, short_window: int = 7, long_window: int = 30
) -> pd.DataFrame:
    """
    Calculates short-term and long-term rolling moving averages using pandas.

    :param df: DataFrame containing 'Price' column.
    :param short_window: Window size for short-term MA (default 7 days).
    :param long_window: Window size for long-term MA (default 30 days).
    :return: DataFrame with added short and long MA columns.
    """
    df = df.copy()
    df[f"MA{short_window}"] = df["Price"].rolling(window=short_window).mean().round(2)
    df[f"MA{long_window}"] = df["Price"].rolling(window=long_window).mean().round(2)
    return df


def run_golden_cross_strategy(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
    short_col: str = "MA7",
    long_col: str = "MA30",
) -> tuple[pd.DataFrame, dict]:
    """
    Executes the Golden Cross trading strategy over the price series.

    Strategy Rules:
    - Initial Entry Condition: On the first day where both MAs are calculated:
      If MA_short > MA_long, buy BTC immediately.
    - Golden Cross: Short MA crosses above Long MA -> BUY (allocate all cash to BTC).
    - Death Cross: Short MA crosses below Long MA -> SELL (sell all BTC for cash).
    - Otherwise -> HOLD existing position.

    :param df: DataFrame with Price, short MA, and long MA columns.
    :param initial_cash: Starting cash balance in USD.
    :param short_col: Column name for short-term MA.
    :param long_col: Column name for long-term MA.
    :return: Tuple of (DataFrame with trade details, summary dict).
    """
    df = df.copy()

    signals = []
    actions = []
    cash_history = []
    btc_history = []
    portfolio_value_history = []

    cash = initial_cash
    btc_holdings = 0.0

    prev_short = None
    prev_long = None

    for idx, row in df.iterrows():
        price = row["Price"]
        short_ma = row[short_col]
        long_ma = row[long_col]

        signal = "NONE"
        action = "HOLD"

        # Trading signals require both moving averages to be available
        if pd.notna(short_ma) and pd.notna(long_ma):
            if prev_short is not None and prev_long is not None:
                # Golden Cross: Short MA crosses ABOVE Long MA
                if prev_short <= prev_long and short_ma > long_ma:
                    signal = "GOLDEN_CROSS"
                # Death Cross: Short MA crosses BELOW Long MA
                elif prev_short >= prev_long and short_ma < long_ma:
                    signal = "DEATH_CROSS"
            else:
                # Initial evaluation on day 30 when long MA first becomes available
                if short_ma > long_ma and cash > 0:
                    signal = "INITIAL_BUY"

            # Execute trade based on signal
            if signal in ("GOLDEN_CROSS", "INITIAL_BUY") and cash > 0:
                btc_holdings = cash / price
                cash = 0.0
                action = "BUY"
            elif signal == "DEATH_CROSS" and btc_holdings > 0:
                cash = btc_holdings * price
                btc_holdings = 0.0
                action = "SELL"

            prev_short = short_ma
            prev_long = long_ma

        portfolio_val = cash + (btc_holdings * price)

        signals.append(signal)
        actions.append(action)
        cash_history.append(round(cash, 2))
        btc_history.append(round(btc_holdings, 6))
        portfolio_value_history.append(round(portfolio_val, 2))

    df["Signal"] = signals
    df["Action"] = actions
    df["Cash"] = cash_history
    df["BTC_Holdings"] = btc_history
    df["Portfolio_Value"] = portfolio_value_history

    # Calculate summary metrics
    start_price = df.iloc[0]["Price"]
    end_price = df.iloc[-1]["Price"]
    final_portfolio_val = portfolio_value_history[-1]

    total_return_pct = ((final_portfolio_val - initial_cash) / initial_cash) * 100.0
    buy_hold_return_pct = ((end_price - start_price) / start_price) * 100.0
    total_buys = actions.count("BUY")
    total_sells = actions.count("SELL")

    summary = {
        "initial_capital": initial_cash,
        "final_portfolio_value": final_portfolio_val,
        "strategy_return_pct": total_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "start_btc_price": start_price,
        "end_btc_price": end_price,
    }

    return df, summary


def print_ledger_and_summary(df: pd.DataFrame, summary: dict) -> None:
    """
    Prints a formatted daily ledger and trading summary to standard output.
    """
    print("\n" + "=" * 100)
    print("                    BITCOIN 60-DAY GOLDEN CROSS TRADING LEDGER")
    print("=" * 100)
    header = f"{'Day':<4} | {'Date':<10} | {'BTC Price':<10} | {'7-Day MA':<10} | {'30-Day MA':<10} | {'Action':<6} | {'Cash ($)':<11} | {'BTC Balance':<11} | {'Total Val ($)':<12}"
    print(header)
    print("-" * 100)

    for _, row in df.iterrows():
        ma7_str = f"${row['MA7']:,.2f}" if pd.notna(row['MA7']) else "N/A"
        ma30_str = f"${row['MA30']:,.2f}" if pd.notna(row['MA30']) else "N/A"
        price_str = f"${row['Price']:,.2f}"
        cash_str = f"${row['Cash']:,.2f}"
        val_str = f"${row['Portfolio_Value']:,.2f}"

        print(
            f"{row['Day']:<4} | {str(row['Date']):<10} | {price_str:<10} | {ma7_str:<10} | {ma30_str:<10} | {row['Action']:<6} | {cash_str:<11} | {row['BTC_Holdings']:<11.6f} | {val_str:<12}"
        )

    print("-" * 100)
    print("\n" + "=" * 60)
    print("                 PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Initial Capital          : ${summary['initial_capital']:,.2f}")
    print(f"Final Portfolio Value    : ${summary['final_portfolio_value']:,.2f}")
    print(f"Strategy Total Return    : {summary['strategy_return_pct']:+.2f}%")
    print(f"Buy & Hold Return        : {summary['buy_hold_return_pct']:+.2f}%")
    print(f"Starting BTC Price       : ${summary['start_btc_price']:,.2f}")
    print(f"Ending BTC Price         : ${summary['end_btc_price']:,.2f}")
    print(f"Total Trades Executed    : {summary['total_buys']} Buys / {summary['total_sells']} Sells")
    print("=" * 60 + "\n")


def main():
    # 1. Simulate 60 days of BTC price data with a seed that shows cross dynamics
    # Seed 10 generates price movements that induce a Golden Cross transition
    df = simulate_bitcoin_prices(days=60, initial_price=60000.0, annual_drift=0.30, annual_volatility=0.70, seed=10)

    # 2. Calculate 7-day and 30-day Moving Averages using pandas
    df = calculate_moving_averages(df, short_window=7, long_window=30)

    # 3. Run Golden Cross Strategy
    df_ledger, summary = run_golden_cross_strategy(df, initial_cash=10000.0)

    # 4. Print Daily Ledger and Final Performance
    print_ledger_and_summary(df_ledger, summary)


if __name__ == "__main__":
    main()
