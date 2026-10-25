"""Binning rules on synthetic data: monotonic, size floor, bin cap."""
import numpy as np
import pandas as pd

from csa import binning, config


def test_numeric_bins_follow_rules():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(700, 30, 50_000))
    p = 1 / (1 + np.exp((x - 680) / 25))           # bad rate falls as x rises
    y = pd.Series((rng.random(len(x)) < p).astype(int))
    spec = binning.fit("x", "numeric", x, y)
    t, m = binning.evaluate(spec, x, y, x, y)
    cfg = config.load("prep")["binning"]
    assert m["n_bins"] <= cfg["max_bins"]
    assert (t.vol_build_pct >= cfg["min_bin_share_pct"]).all()
    assert spec.direction == "down"
    assert (np.diff(t.bad_build_pct.to_numpy()) < 0).all()   # strictly falling
    assert m["iv_build"] > 0.1


def test_categorical_small_levels_merge():
    x = pd.Series(["a"] * 630 + ["b"] * 300 + ["c"] * 40 + ["d"] * 30)   # c, d < 5%
    y = pd.Series([0, 1] * 500)
    spec = binning.fit("x", "categorical", x, y)
    assert spec.groups["c"] == "Other" and spec.groups["d"] == "Other"
    assert spec.assign(pd.Series(["zzz"])).iloc[0] == "Other"   # unseen level
