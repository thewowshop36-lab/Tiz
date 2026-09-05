"""
Bitcoin Price Simulation & Golden Cross Trading Algorithm
=========================================================
This script simulates 60 days of Bitcoin daily price data using Geometric Brownian Motion (GBM),
calculates 7-day and 30-day Simple Moving Averages (SMAs) with pandas, executes a Golden Cross
trading strategy (buying when 7-day MA crosses above 30-day MA, selling when 7-day MA crosses below),
and prints a daily ledger alongside final portfolio performance.
"""

import numpy as np
import pandas as pd


def simulate_btc_prices(
    days: int = 60,
    initial_price: float = 65000.0,
    mu: float = 0.15,
    sigma: float = 0.60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulates daily Bitcoin prices using Geometric Brownian Motion (GBM).

    Parameters:
    - days: Number of simulation days (default: 60)
    - initial_price: Starting BTC price in USD (default: $65,000)
    - mu: Annualized expected return / drift (default: 0.15)
    - sigma: Annualized volatility (default: 0.60, standard for BTC)
    - seed: Random seed for reproducibility

    Returns:
    - pd.DataFrame containing Day and Price
    """
    np.random.seed(seed)
    dt = 1.0 / 365.0  # Daily step in years

    daily_returns = np.random.normal(
        loc=(mu - 0.5 * sigma**2) * dt,
        scale=sigma * np.sqrt(dt),
        size=days,
    )

    prices = [initial_price]
    for r in daily_returns:
        prices.append(prices[-1] * np.exp(r))

    df = pd.DataFrame(
        {
            "Day": list(range(1, days + 1)),
            "Price": prices[1:],
        }
    )
    return df


def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 7-day (short-term) and 30-day (long-term) Simple Moving Averages using pandas.
    """
    df = df.copy()
    df["MA_7"] = df["Price"].rolling(window=7, min_periods=1).mean()
    df["MA_30"] = df["Price"].rolling(window=30, min_periods=1).mean()
    return df


def run_golden_cross_strategy(df: pd.DataFrame, initial_cash: float = 10000.0):
    """
    Simulates the Golden Cross trading algorithm:
    - Golden Cross (MA_7 crosses above MA_30): BUY signal -> Invest all available cash into BTC.
    - Death Cross (MA_7 crosses below MA_30): SELL signal -> Liquidate all BTC into cash.
    - Otherwise: HOLD.

    Tracks daily ledger and calculates final performance metrics.
    """
    cash = initial_cash
    btc_holdings = 0.0

    ledger = []

    for i in range(len(df)):
        day = df.loc[i, "Day"]
        price = df.loc[i, "Price"]
        ma_7 = df.loc[i, "MA_7"]
        ma_30 = df.loc[i, "MA_30"]

        signal = "HOLD"
        trade_detail = "No Action"

        if i > 0:
            prev_ma_7 = df.loc[i - 1, "MA_7"]
            prev_ma_30 = df.loc[i - 1, "MA_30"]

            # Golden Cross: Short MA crosses above Long MA
            if prev_ma_7 <= prev_ma_30 and ma_7 > ma_30:
                if cash > 0:
                    signal = "BUY"
                    btc_bought = cash / price
                    btc_holdings += btc_bought
                    trade_detail = f"BUY {btc_bought:.4f} BTC @ ${price:,.2f}"
                    cash = 0.0

            # Death Cross: Short MA crosses below Long MA
            elif prev_ma_7 >= prev_ma_30 and ma_7 < ma_30:
                if btc_holdings > 0:
                    signal = "SELL"
                    cash_received = btc_holdings * price
                    trade_detail = f"SELL {btc_holdings:.4f} BTC @ ${price:,.2f}"
                    cash += cash_received
                    btc_holdings = 0.0

        total_value = cash + (btc_holdings * price)

        ledger.append(
            {
                "Day": int(day),
                "Price": price,
                "MA_7": ma_7,
                "MA_30": ma_30,
                "Signal": signal,
                "Trade Detail": trade_detail,
                "Cash ($)": cash,
                "BTC Holdings": btc_holdings,
                "Total Value ($)": total_value,
            }
        )

    ledger_df = pd.DataFrame(ledger)

    # Initial and Final calculations
    start_price = df.loc[0, "Price"]
    end_price = df.iloc[-1]["Price"]
    final_value = ledger_df.iloc[-1]["Total Value ($)"]
    strategy_return = ((final_value - initial_cash) / initial_cash) * 100.0

    # Benchmark: Buy and Hold Strategy
    buy_hold_btc = initial_cash / start_price
    buy_hold_final_value = buy_hold_btc * end_price
    buy_hold_return = ((buy_hold_final_value - initial_cash) / initial_cash) * 100.0

    summary = {
        "Initial Capital": initial_cash,
        "Final Portfolio Value": final_value,
        "Strategy Total Return (%)": strategy_return,
        "Buy & Hold Final Value": buy_hold_final_value,
        "Buy & Hold Return (%)": buy_hold_return,
        "Initial BTC Price": start_price,
        "Final BTC Price": end_price,
        "Total Trades Executed": len(ledger_df[ledger_df["Signal"].isin(["BUY", "SELL"])]),
    }

    return ledger_df, summary


def print_simulation_results(ledger_df: pd.DataFrame, summary: dict):
    """
    Prints a formatted daily ledger and portfolio performance summary.
    """
    print("=" * 115)
    print(" " * 35 + "BITCOIN 60-DAY TRADING LEDGER (GOLDEN CROSS)")
    print("=" * 115)
    header = (
        f"{'Day':<5} | {'BTC Price ($)':<12} | {'7-Day MA ($)':<12} | {'30-Day MA ($)':<12} | "
        f"{'Signal':<6} | {'Cash ($)':<10} | {'BTC Position':<12} | {'Total Value ($)':<14}"
    )
    print(header)
    print("-" * 115)

    for _, row in ledger_df.iterrows():
        signal_str = row["Signal"]
        if signal_str == "BUY":
            signal_formatted = "\033[92mBUY \033[0m"  # Green
        elif signal_str == "SELL":
            signal_formatted = "\033[91mSELL\033[0m"  # Red
        else:
            signal_formatted = "HOLD"

        print(
            f"{int(row['Day']):<5} | "
            f"{row['Price']:>12,.2f} | "
            f"{row['MA_7']:>12,.2f} | "
            f"{row['MA_30']:>12,.2f} | "
            f"{signal_formatted:<6} | "
            f"{row['Cash ($)']:>10,.2f} | "
            f"{row['BTC Holdings']:>12.4f} | "
            f"{row['Total Value ($)']:>14,.2f}"
        )

    print("=" * 115)
    print("\n" + "=" * 60)
    print(" " * 15 + "PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Initial Investment       : ${summary['Initial Capital']:,.2f}")
    print(f"Final Portfolio Value    : ${summary['Final Portfolio Value']:,.2f}")
    print(f"Strategy Return          : {summary['Strategy Total Return (%)']:.2f}%")
    print("-" * 60)
    print(f"Buy & Hold Final Value   : ${summary['Buy & Hold Final Value']:,.2f}")
    print(f"Buy & Hold Return        : {summary['Buy & Hold Return (%)']:.2f}%")
    print("-" * 60)
    print(f"Starting BTC Price       : ${summary['Initial BTC Price']:,.2f}")
    print(f"Ending BTC Price         : ${summary['Final BTC Price']:,.2f}")
    print(f"Total Trades Executed    : {summary['Total Trades Executed']}")
    print("=" * 60)


if __name__ == "__main__":
    df_prices = simulate_btc_prices(days=60, seed=10)
    df_ma = calculate_moving_averages(df_prices)
    ledger_df, summary = run_golden_cross_strategy(df_ma, initial_cash=10000.0)
    print_simulation_results(ledger_df, summary)
