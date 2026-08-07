import pandas as pd

CSV_FILE = "btc_data.csv"


def calculate_rsi(close, length=14):
    """Calculate RSI using Wilder's smoothing method."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (equivalent to EMA with alpha=1/length)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def main():
    # Read the CSV file
    df = pd.read_csv(CSV_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Calculate RSI (14-period)
    df["rsi"] = calculate_rsi(df["close"], length=14)

    # Calculate 20-day simple moving average
    df["sma_20"] = df["close"].rolling(window=20).mean()

    # Get the most recent values
    latest = df.iloc[-1]
    current_rsi = latest["rsi"]
    current_sma20 = latest["sma_20"]
    current_price = latest["close"]
    latest_date = latest["date"].strftime("%Y-%m-%d")

    print(f"Date: {latest_date}")
    print(f"BTC/USD Price: ${current_price:.2f}")
    print(f"RSI (14): {current_rsi:.2f}")
    print(f"20-day Moving Average: ${current_sma20:.2f}")


if __name__ == "__main__":
    main()