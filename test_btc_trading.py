#!/usr/bin/env python3
"""
Unit tests for Bitcoin Price Simulation & Golden Cross Trading Strategy.
"""

import unittest
import numpy as np
import pandas as pd

from btc_trading_simulation import (
    simulate_bitcoin_prices,
    calculate_moving_averages,
    run_golden_cross_strategy,
)


class TestBitcoinTradingSimulation(unittest.TestCase):

    def test_simulate_bitcoin_prices(self):
        days = 60
        initial_price = 50000.0
        df = simulate_bitcoin_prices(days=days, initial_price=initial_price, seed=42)

        self.assertEqual(len(df), days)
        self.assertIn("Day", df.columns)
        self.assertIn("Date", df.columns)
        self.assertIn("Price", df.columns)
        self.assertTrue((df["Price"] > 0).all())
        self.assertEqual(df["Day"].iloc[0], 1)
        self.assertEqual(df["Day"].iloc[-1], 60)

    def test_calculate_moving_averages(self):
        prices = [100.0] * 30
        df = pd.DataFrame({"Price": prices})
        df_ma = calculate_moving_averages(df, short_window=7, long_window=30)

        self.assertTrue(pd.isna(df_ma["MA7"].iloc[0]))
        self.assertEqual(df_ma["MA7"].iloc[6], 100.0)
        self.assertTrue(pd.isna(df_ma["MA30"].iloc[28]))
        self.assertEqual(df_ma["MA30"].iloc[29], 100.0)

    def test_golden_cross_and_death_cross_logic(self):
        # Construct synthetic price series with explicit cross conditions
        # Day 1-30: MA7 <= MA30
        # Day 31: Golden Cross (MA7 > MA30) -> BUY
        # Day 35: Death Cross (MA7 < MA30) -> SELL
        days = 40
        dates = pd.date_range("2026-01-01", periods=days)
        prices = [100.0] * days

        df = pd.DataFrame({"Day": range(1, days + 1), "Date": dates, "Price": prices})
        df["MA7"] = 100.0
        df["MA30"] = 100.0

        # MAs before day 30 are NaN
        df.loc[:28, "MA7"] = np.nan
        df.loc[:28, "MA30"] = np.nan

        # Day 29 (index 28): MA7 = 90, MA30 = 100 (MA7 <= MA30)
        df.loc[28, "MA7"] = 90.0
        df.loc[28, "MA30"] = 100.0

        # Day 30 (index 29): MA7 = 110, MA30 = 100 (Golden Cross!)
        df.loc[29, "MA7"] = 110.0
        df.loc[29, "MA30"] = 100.0

        # Day 31 (index 30): MA7 = 115, MA30 = 100
        df.loc[30, "MA7"] = 115.0
        df.loc[30, "MA30"] = 100.0

        # Day 32 (index 31): MA7 = 95, MA30 = 100 (Death Cross!)
        df.loc[31, "MA7"] = 95.0
        df.loc[31, "MA30"] = 100.0

        df_ledger, summary = run_golden_cross_strategy(df, initial_cash=1000.0)

        self.assertEqual(df_ledger.loc[29, "Signal"], "GOLDEN_CROSS")
        self.assertEqual(df_ledger.loc[29, "Action"], "BUY")
        self.assertEqual(df_ledger.loc[29, "Cash"], 0.0)
        self.assertEqual(df_ledger.loc[29, "BTC_Holdings"], 10.0)

        self.assertEqual(df_ledger.loc[31, "Signal"], "DEATH_CROSS")
        self.assertEqual(df_ledger.loc[31, "Action"], "SELL")
        self.assertEqual(df_ledger.loc[31, "Cash"], 1000.0)
        self.assertEqual(df_ledger.loc[31, "BTC_Holdings"], 0.0)

        self.assertEqual(summary["total_buys"], 1)
        self.assertEqual(summary["total_sells"], 1)


if __name__ == "__main__":
    unittest.main()
