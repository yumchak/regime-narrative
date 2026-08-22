"""Feature construction.

Four features, unchanged from the original procedure:

    1. 20-day realised volatility of the target equity series
    2. 20-day momentum of the target equity series
    3. VIX level          (global fear gauge)
    4. 20-day return of HYG (credit stress proxy)

The generalisation test swaps only the equity leg. Feeding US VIX and HYG to a
Nikkei model is deliberate: the brief asks for the *identical procedure* run on
an unseen index, and changing the macro inputs would make it a different
procedure and a weaker comparison.
"""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["equity_vol", "equity_mom", "vix", "hyg_ret"]
LOOKBACK = 20


def build_features(
    equity: pd.Series,
    vix: pd.Series,
    hyg: pd.Series,
    *,
    lookback: int = LOOKBACK,
) -> pd.DataFrame:
    """Assemble the feature matrix on the intersection of the three indices.

    Returns unscaled features. Scaling happens inside each walk-forward fold,
    never globally -- fitting a scaler on the full history leaks the
    distribution of the test period into the training window.
    """
    equity = equity.dropna()
    idx = equity.index.intersection(vix.dropna().index).intersection(hyg.dropna().index)
    equity, vix, hyg = equity.loc[idx], vix.loc[idx], hyg.loc[idx]

    returns = equity.pct_change()
    features = pd.DataFrame(
        {
            "equity_vol": returns.rolling(lookback).std(),
            "equity_mom": equity.pct_change(lookback),
            "vix": vix,
            "hyg_ret": hyg.pct_change(lookback),
        },
        index=idx,
    )
    return features.dropna()


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change()


def forward_realised_vol(equity: pd.Series, horizon: int = 20) -> pd.Series:
    """Realised volatility over the ``horizon`` days *after* each date.

    This is the anti-circularity statistic. The HMM is fed *trailing* realised
    volatility, so separating states by trailing volatility is close to
    definitional. Separating them by volatility that has not happened yet is
    a genuine claim about the state's predictive content.

    The value at date t uses returns from t+1 to t+horizon inclusive, so it is
    strictly disjoint from anything the model saw when labelling t.
    """
    returns = equity.pct_change()
    fwd = returns.shift(-1).rolling(horizon).std().shift(-(horizon - 1))
    return fwd
