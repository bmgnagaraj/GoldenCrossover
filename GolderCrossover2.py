import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import os
from termcolor import colored

# Provide fallback list if tickers.csv is missing
def load_tickers(csv_path=r'Book1.csv'):
    try:
        df = pd.read_csv(csv_path)

        return df['Ticker'].dropna().unique().tolist()
    except:
        print(f"⚠️  File '{csv_path}' not found. Using default tickers.")
        return ['INTC', 'ASML']

tickers = load_tickers()
results = []

for ticker in tickers:
    print(f"\nProcessing: {ticker}")

    try:
        data = yf.download(ticker, start='2010-01-01', interval='1d', auto_adjust=True, progress=False)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]

        close_candidates = [col for col in data.columns if 'adj close' in col.lower() or 'close' in col.lower()]
        if not close_candidates:
            raise KeyError("No 'Close' or 'Adj Close' column found in data")

        close_col = close_candidates[0]
        data = data[[close_col]].rename(columns={close_col: 'Close'}).dropna()

        if data.empty or len(data) < 600:
            print(f"  Not enough data to compute indicators. Rows available: {len(data)}")
            continue

        data['50DMA'] = data['Close'].rolling(window=50).mean()
        data['200DMA'] = data['Close'].rolling(window=200).mean()
        data['500DMA'] = data['Close'].rolling(window=500).mean()

        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        sma20 = data['Close'].rolling(window=20).mean()
        std20 = data['Close'].rolling(window=20).std()
        data['UpperBB'] = sma20 + (2 * std20)
        data['LowerBB'] = sma20 - (2 * std20)

        low14 = data['Close'].rolling(window=14).min()
        high14 = data['Close'].rolling(window=14).max()
        data['%K'] = 100 * ((data['Close'] - low14) / (high14 - low14))

        data['ADX'] = 100 * abs(data['%K'].diff()).rolling(window=14).mean()

        expected_cols = ['Close', '50DMA', '200DMA', '500DMA']
        valid_data = data.dropna(subset=expected_cols)

        if valid_data.empty or len(valid_data) < 2:
            print("  Valid data after dropna is empty or too short.")
            continue

        last_row = valid_data.iloc[-1]
        prev_row = valid_data.iloc[-2]

        if last_row[expected_cols].isnull().any():
            print(f"  Final row still contains NaNs in {expected_cols}")
            continue

        def calc_change(days):
            if len(valid_data) > days:
                return round(((last_row['Close'] - valid_data.iloc[-days]['Close']) / valid_data.iloc[-days]['Close']) * 100, 2)
            return None

        changes = {
            '1W Change (%)': calc_change(5),
            '1M Change (%)': calc_change(21),
            '3M Change (%)': calc_change(63),
            '6M Change (%)': calc_change(126),
            '9M Change (%)': calc_change(189),
            '12M Change (%)': calc_change(252)
        }

        pct_diff_50_200 = ((last_row['50DMA'] - last_row['200DMA']) / last_row['200DMA']) * 100
        pct_diff_200_500 = ((last_row['200DMA'] - last_row['500DMA']) / last_row['500DMA']) * 100
        pct_diff_50_500 = ((last_row['50DMA'] - last_row['500DMA']) / last_row['500DMA']) * 100

        golden_crossover_pct = ((last_row['200DMA'] - last_row['50DMA']) / last_row['200DMA']) * 100 if last_row['50DMA'] < last_row['200DMA'] else 0
        death_crossover_pct = ((last_row['50DMA'] - last_row['200DMA']) / last_row['200DMA']) * 100 if last_row['50DMA'] > last_row['200DMA'] else 0

        crossover = ''
        if prev_row['50DMA'] < prev_row['200DMA'] and last_row['50DMA'] > last_row['200DMA']:
            crossover = 'Golden Cross'
        elif prev_row['50DMA'] > prev_row['200DMA'] and last_row['50DMA'] < last_row['200DMA']:
            crossover = 'Death Cross'

        result = {
            'Ticker': ticker,
            'Current Price': round(last_row['Close'], 2),
            '1W Change (%)': changes['1W Change (%)'],
            '1M Change (%)': changes['1M Change (%)'],
            '3M Change (%)': changes['3M Change (%)'],
            '6M Change (%)': changes['6M Change (%)'],
            '9M Change (%)': changes['9M Change (%)'],
            '12M Change (%)': changes['12M Change (%)'],
            '50DMA': round(last_row['50DMA'], 2),
            '200DMA': round(last_row['200DMA'], 2),
            '500DMA': round(last_row['500DMA'], 2),
            'PctDiff_50_200 (%)': round(pct_diff_50_200, 2),
            'PctDiff_200_500 (%)': round(pct_diff_200_500, 2),
            'PctDiff_50_500 (%)': round(pct_diff_50_500, 2),
            'GoldenCrossOver (%)': round(golden_crossover_pct, 2),
            'DeathCrossOver (%)': round(death_crossover_pct, 2),
            'Crossover': crossover
        }

        results.append(result)

    except Exception as e:
        print(f"  Error processing {ticker}: {e}")

def colorize(val, width=20):
    if isinstance(val, (int, float)) and val < 0:
        return colored(f"{val:<{width}.2f}", 'red')
    return f"{val:<{width}.2f}" if isinstance(val, (int, float)) else f"{val:<{width}}"

summary_df = pd.DataFrame(results)
if not summary_df.empty:
    print("\n📊 DMA Summary (including 50/200/500DMA crossovers and signals):")
    column_width = 20
    header = [f"{col:<{column_width}}" for col in summary_df.columns]
    print(" ".join(header))
    print("=" * (len(header) * column_width))

    for _, row in summary_df.iterrows():
        line = []
        for col in summary_df.columns:
            val = row[col]
            line.append(colorize(val, column_width))
        print(" ".join(line))
