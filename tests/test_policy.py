"""Current policy: every branch, missing values, determinism."""
import numpy as np
import pandas as pd

from csa import policy


def _rows():
    return pd.DataFrame([
        # grade, fico_mid, dti, loan_to_income, expected
        ("A", 662, 10, 0.10, True),     # A approves at any FICO
        ("B", 662, 34.9, 0.39, True),
        ("C", 677, 20, 0.20, True),     # C needs FICO >= 677
        ("C", 672, 20, 0.20, False),
        ("D", 697, 20, 0.20, True),     # D needs FICO >= 697
        ("D", 692, 20, 0.20, False),
        ("E", 800, 5, 0.05, False),     # E-G always decline
        ("A", 750, 35.1, 0.10, False),  # DTI knockout
        ("A", 750, 10, 0.41, False),    # loan-to-income knockout
        ("A", 750, np.nan, 0.10, False),  # missing DTI declines
        ("A", 750, 10, np.nan, False),    # missing income declines
    ], columns=["grade", "fico_mid", "dti", "loan_to_income", "expected"])


def test_each_branch():
    df = _rows()
    got = policy.approve(df)
    assert got.tolist() == df["expected"].tolist()


def test_deterministic():
    df = _rows()
    assert policy.approve(df).equals(policy.approve(df))
