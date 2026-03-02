"""Download Binance historical aggTrades data for calibration.

Usage (from project root):
    python -m data.download_binance --symbol BTCUSDT --dates 2025-01-01 2025-01-07

Downloads aggTrades CSVs from data.binance.vision → data/raw/{SYMBOL}/.
Each file: agg_trade_id, price, quantity, first_trade_id, last_trade_id,
           transact_time (ms), is_buyer_maker

Requirements: internet access + requests.
"""

import argparse
import io
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"

AGGTRADE_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]


def download_day(symbol: str, date_str: str, out_dir: Path) -> Path | None:
    """Download one day of aggTrades. Returns path to CSV or None."""
    fname = f"{symbol}-aggTrades-{date_str}"
    url = f"{BASE_URL}/{symbol}/{fname}.zip"
    csv_path = out_dir / f"{fname}.csv"

    if csv_path.exists():
        print(f"  [skip] {csv_path.name} already exists")
        return csv_path

    print(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"  [FAIL] HTTP {resp.status_code} for {date_str}")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        csv_name = [n for n in names if n.endswith(".csv")][0]
        zf.extract(csv_name, out_dir)
        extracted = out_dir / csv_name
        if extracted != csv_path:
            extracted.rename(csv_path)

    print(f"  [OK]   {csv_path.name}  ({csv_path.stat().st_size / 1e6:.1f} MB)")
    return csv_path


def download_range(symbol: str, start: str, end: str, out_dir: str = None) -> list[Path]:
    """Download aggTrades for a date range [start, end] inclusive."""
    if out_dir is None:
        out_dir = Path(__file__).parent / "raw" / symbol
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = datetime.strptime(start, "%Y-%m-%d")
    d_end = datetime.strptime(end, "%Y-%m-%d")
    paths = []

    while d <= d_end:
        date_str = d.strftime("%Y-%m-%d")
        p = download_day(symbol, date_str, out_dir)
        if p is not None:
            paths.append(p)
        d += timedelta(days=1)

    print(f"\nDownloaded {len(paths)} files to {out_dir}")
    return paths


def load_trades(symbol: str, start: str, end: str, data_dir: str = None) -> pd.DataFrame:
    """Load downloaded aggTrades into a single DataFrame.

    Returns columns: timestamp (datetime64), price, quantity, is_buyer_maker.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "raw" / symbol
    else:
        data_dir = Path(data_dir)

    d = datetime.strptime(start, "%Y-%m-%d")
    d_end = datetime.strptime(end, "%Y-%m-%d")
    frames = []

    while d <= d_end:
        date_str = d.strftime("%Y-%m-%d")
        csv_path = data_dir / f"{symbol}-aggTrades-{date_str}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, header=None, names=AGGTRADE_COLS)
            frames.append(df)
        d += timedelta(days=1)

    if not frames:
        raise FileNotFoundError(f"No data found in {data_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    df["price"] = df["price"].astype(float)
    df["quantity"] = df["quantity"].astype(float)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    return df.sort_values("timestamp").reset_index(drop=True)


def download_pair(sym1: str, sym2: str, start: str, end: str):
    """Download data for a correlated pair (e.g. BTCUSDT + ETHUSDT)."""
    print(f"=== Downloading {sym1} ===")
    download_range(sym1, start, end)
    print(f"\n=== Downloading {sym2} ===")
    download_range(sym2, start, end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance aggTrades")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--symbol2", default=None)
    parser.add_argument("--dates", nargs=2, metavar=("START", "END"),
                        default=["2025-01-01", "2025-01-07"])
    args = parser.parse_args()

    if args.symbol2:
        download_pair(args.symbol, args.symbol2, args.dates[0], args.dates[1])
    else:
        download_range(args.symbol, args.dates[0], args.dates[1])
