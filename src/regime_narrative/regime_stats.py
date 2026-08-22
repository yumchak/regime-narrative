"""Volatility-separation statistics.

Two versions of the headline number, reported side by side:

``contemporaneous``
    Standard deviation of same-day returns within each state. This is what the
    original notebook computed. It is partly definitional -- the HMM is fed
    trailing realised volatility, so it had better separate on volatility.

``forward``
    Standard deviation of returns over the days *after* each state assignment,
    with no overlap with anything the model saw. This is the honest version of
    the claim: the state says something about volatility that has not happened
    yet.

Both are computed on pooled out-of-sample test days only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass
class VolSeparation:
    label: str
    n_calm: int
    n_stressed: int
    vol_calm: float
    vol_stressed: float
    ratio: float
    levene_stat: float
    levene_p: float
    ttest_stat: float
    ttest_p: float
    mean_calm: float
    mean_stressed: float
    ratio_ci_low: float = float("nan")
    ratio_ci_high: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


def vol_separation(
    returns: pd.Series,
    states: pd.Series,
    *,
    label: str = "contemporaneous",
    n_boot: int = 2000,
    seed: int = 20260822,
) -> VolSeparation:
    """Volatility ratio between stressed (1) and calm (0) days, with a CI.

    The confidence interval is a stationary bootstrap over blocks, not an iid
    bootstrap: daily returns are serially dependent and volatility clusters, so
    an iid resample would give an interval that is far too tight.
    """
    idx = returns.index.intersection(states.index)
    r = returns.loc[idx].astype(float)
    s = states.loc[idx].astype(int)
    mask = r.notna()
    r, s = r[mask], s[mask]

    calm = r[s == 0]
    stressed = r[s == 1]

    if len(calm) < 2 or len(stressed) < 2:
        return VolSeparation(
            label=label,
            n_calm=len(calm),
            n_stressed=len(stressed),
            vol_calm=float(calm.std()) if len(calm) > 1 else float("nan"),
            vol_stressed=float(stressed.std()) if len(stressed) > 1 else float("nan"),
            ratio=float("nan"),
            levene_stat=float("nan"),
            levene_p=float("nan"),
            ttest_stat=float("nan"),
            ttest_p=float("nan"),
            mean_calm=float(calm.mean()) if len(calm) else float("nan"),
            mean_stressed=float(stressed.mean()) if len(stressed) else float("nan"),
        )

    vol_c, vol_s = float(calm.std()), float(stressed.std())
    lev_stat, lev_p = sps.levene(calm.values, stressed.values)
    t_stat, t_p = sps.ttest_ind(calm.values, stressed.values, equal_var=False)

    lo, hi = _block_bootstrap_ratio_ci(r.values, s.values, n_boot=n_boot, seed=seed)

    return VolSeparation(
        label=label,
        n_calm=len(calm),
        n_stressed=len(stressed),
        vol_calm=vol_c,
        vol_stressed=vol_s,
        ratio=vol_s / vol_c if vol_c else float("nan"),
        levene_stat=float(lev_stat),
        levene_p=float(lev_p),
        ttest_stat=float(t_stat),
        ttest_p=float(t_p),
        mean_calm=float(calm.mean()),
        mean_stressed=float(stressed.mean()),
        ratio_ci_low=lo,
        ratio_ci_high=hi,
    )


def _block_bootstrap_ratio_ci(
    returns: np.ndarray,
    states: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    block: int = 20,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for the volatility ratio under a moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < block * 2:
        return float("nan"), float("nan")
    n_blocks = int(np.ceil(n / block))
    ratios = np.empty(n_boot)

    for b in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        r_b, s_b = returns[idx], states[idx]
        calm, stressed = r_b[s_b == 0], r_b[s_b == 1]
        if len(calm) < 2 or len(stressed) < 2:
            ratios[b] = np.nan
            continue
        sd_c = calm.std(ddof=1)
        ratios[b] = stressed.std(ddof=1) / sd_c if sd_c > 0 else np.nan

    ratios = ratios[~np.isnan(ratios)]
    if len(ratios) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(ratios, 100 * alpha / 2)), float(
        np.percentile(ratios, 100 * (1 - alpha / 2))
    )


def forward_vol_separation(
    forward_vol: pd.Series, states: pd.Series, *, label: str = "forward-20d"
) -> dict:
    """Ratio of mean forward realised volatility between states.

    Forward volatility windows overlap heavily between adjacent days, so a
    p-value here would be badly overstated and none is reported. The ratio is
    descriptive, and it is the number that answers the circularity objection.
    """
    idx = forward_vol.index.intersection(states.index)
    f = forward_vol.loc[idx].astype(float)
    s = states.loc[idx].astype(int)
    mask = f.notna()
    f, s = f[mask], s[mask]

    calm, stressed = f[s == 0], f[s == 1]
    if not len(calm) or not len(stressed):
        return {"label": label, "ratio": float("nan"), "n_calm": len(calm), "n_stressed": len(stressed)}

    return {
        "label": label,
        "n_calm": int(len(calm)),
        "n_stressed": int(len(stressed)),
        "fwd_vol_calm": float(calm.mean()),
        "fwd_vol_stressed": float(stressed.mean()),
        "ratio": float(stressed.mean() / calm.mean()) if calm.mean() else float("nan"),
        "median_ratio": float(stressed.median() / calm.median()) if calm.median() else float("nan"),
    }


def per_fold_ratios(
    returns: pd.Series, states: pd.Series, fold_id: pd.Series
) -> pd.DataFrame:
    """Volatility ratio computed separately within each held-out fold.

    A single pooled number can be carried by two or three crisis folds. The
    per-fold distribution shows whether the separation is a general property
    or a handful of episodes, which is the question a judge will ask.
    """
    rows = []
    idx = returns.index.intersection(states.index).intersection(fold_id.index)
    r, s, f = returns.loc[idx], states.loc[idx], fold_id.loc[idx]

    for fold in sorted(f.unique()):
        m = f == fold
        calm, stressed = r[m][s[m] == 0].dropna(), r[m][s[m] == 1].dropna()
        rows.append(
            {
                "fold": int(fold),
                "start": str(r[m].index[0].date()),
                "end": str(r[m].index[-1].date()),
                "n_calm": len(calm),
                "n_stressed": len(stressed),
                "vol_calm": float(calm.std()) if len(calm) > 1 else np.nan,
                "vol_stressed": float(stressed.std()) if len(stressed) > 1 else np.nan,
                "ratio": (
                    float(stressed.std() / calm.std())
                    if len(calm) > 1 and len(stressed) > 1 and calm.std() > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
