"""Price data loading, with on-disk caching and a retrieval manifest.

Every download is cached to CSV on first retrieval and re-read thereafter, so
a rerun is deterministic and offline. The manifest records what was fetched,
when, and over what span -- the same audit discipline the news layer uses.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .config import cache_dir, load_settings, study_end_date, study_start_date


def _manifest_path() -> Path:
    return cache_dir("prices") / "manifest.json"


def _read_manifest() -> dict:
    p = _manifest_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _write_manifest(entry_key: str, entry: dict) -> None:
    manifest = _read_manifest()
    manifest[entry_key] = entry
    _manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_prices(
    tickers: dict[str, str],
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool = False,
    cache_key: str = "universe",
) -> pd.DataFrame:
    """Daily close prices, columns named by the keys of ``tickers``.

    ``tickers`` maps a friendly name to a Yahoo symbol, e.g. ``{"spy": "SPY"}``.
    Rows with any missing value are dropped, matching the original procedure.
    """
    start = start or study_start_date()
    end = end or study_end_date()
    path = cache_dir("prices") / f"{cache_key}.csv"

    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        return df

    import yfinance as yf  # imported lazily so offline reruns need no network

    symbols = list(tickers.values())
    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    close = _extract_close(raw, symbols)
    # Rename Yahoo symbols to friendly names.
    inverse = {sym: name for name, sym in tickers.items()}
    close = close.rename(columns=inverse)
    close = close[[name for name in tickers if name in close.columns]]
    close = close.dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)

    close.to_csv(path)
    _write_manifest(
        cache_key,
        {
            "tickers": tickers,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "actual_start": str(close.index[0].date()) if len(close) else None,
            "actual_end": str(close.index[-1].date()) if len(close) else None,
            "n_rows": int(len(close)),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "yfinance",
            "cache_file": str(path.name),
        },
    )
    return close


def _extract_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Pull the Close block out of whatever shape yfinance returned.

    yfinance changes this layout between versions and between single- and
    multi-ticker requests, so normalise defensively rather than assume.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        if "Close" in level0:
            close = raw["Close"]
        else:  # grouped by ticker instead of by field
            close = raw.xs("Close", axis=1, level=1)
    else:
        if "Close" in raw.columns:
            close = raw[["Close"]]
            close.columns = symbols[:1]
        else:
            close = raw
    return pd.DataFrame(close)


def load_universe(*, refresh: bool = False) -> pd.DataFrame:
    """The original fitted universe: SPY, TLT, GLD, VIX, HYG."""
    cfg = load_settings()
    return load_prices(dict(cfg["universe"]), refresh=refresh, cache_key="universe")


def load_unseen_index(name: str, *, refresh: bool = False) -> pd.DataFrame:
    """One of the never-fitted-on indices, for the generalisation test."""
    cfg = load_settings()
    unseen = dict(cfg["unseen_indices"])
    if name not in unseen:
        raise KeyError(f"{name!r} not in settings.unseen_indices ({list(unseen)})")
    return load_prices(
        {name: unseen[name]}, refresh=refresh, cache_key=f"unseen_{name}"
    )


def load_vix(*, refresh: bool = False) -> pd.Series:
    """VIX alone -- the unseen-index feature sets reuse it as the fear gauge."""
    df = load_prices({"vix": "^VIX"}, refresh=refresh, cache_key="vix_only")
    return df["vix"]
