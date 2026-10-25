"""Stage A: load the raw file, keep the analysis population, build bad flag and vintages.

Only role fields and application-time candidate fields are read. Blocked fields are
never loaded, so they cannot leak into anything downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

PCT_STRIP = {"revol_util", "sec_app_revol_util", "il_util", "all_util", "bc_util"}


def _columns_to_read() -> list[str]:
    c = config.candidate_fields()
    return list(config.role_fields()) + [*c["numeric"], *c["categorical"], *c["date"]]


def load_population(chunksize: int = 200_000) -> tuple[pd.DataFrame, dict]:
    """Return (loans, counts). counts records every exclusion step."""
    s = config.load("settings")["data"]
    d0, d1 = pd.Timestamp(s["issue_from"]), pd.Timestamp(s["issue_to"])
    good, bad = set(s["good_statuses"]), set(s["bad_statuses"])
    counts = {"raw_rows": 0, "non_loan_rows": 0, "other_term": 0,
              "outside_issue_window": 0, "still_open": 0, "kept": 0}
    parts = []
    reader = pd.read_csv(config.path("raw_file"), usecols=_columns_to_read(), dtype=str,
                         chunksize=chunksize, compression="gzip")
    for ch in reader:
        counts["raw_rows"] += len(ch)
        is_loan = ch["id"].str.fullmatch(r"\d+").fillna(False).astype(bool)
        counts["non_loan_rows"] += int((~is_loan).sum())
        ch = ch[is_loan]
        term_ok = ch["term"].str.strip() == s["term"]
        counts["other_term"] += int((~term_ok).sum())
        ch = ch[term_ok]
        issue = pd.to_datetime(ch["issue_d"], format="%b-%Y")
        yr_ok = issue.between(d0, d1)
        counts["outside_issue_window"] += int((~yr_ok).sum())
        ch, issue = ch[yr_ok], issue[yr_ok]
        resolved = ch["loan_status"].isin(good | bad)
        counts["still_open"] += int((~resolved).sum())
        ch, issue = ch[resolved].copy(), issue[resolved]
        ch["issue_date"] = issue
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    counts["kept"] = len(df)
    return _shape(df, bad), counts


def _shape(df: pd.DataFrame, bad_statuses: set) -> pd.DataFrame:
    s = config.load("settings")["data"]
    c = config.candidate_fields()
    out = pd.DataFrame({
        "id": df["id"].astype("int64"),
        "issue_date": df["issue_date"],
        "year": df["issue_date"].dt.year.astype("int16"),
        "vintage": df["issue_date"].dt.to_period("Q").astype(str),
        "term": df["term"].str.strip(),
        "grade": df["grade"],
        "bad": df["loan_status"].isin(bad_statuses).astype("int8"),
    })
    out["split"] = np.where(out["year"] <= s["build_years"][1], "build", "oot")
    for col in c["numeric"]:
        v = df[col]
        if col in PCT_STRIP:
            v = v.str.rstrip("%")
        out[col] = pd.to_numeric(v, errors="coerce").astype("float64")
    for col in c["categorical"]:
        out[col] = df[col].str.strip().replace("", np.nan)
    for col in c["date"]:
        out[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features listed in fields.yaml. Uses raw (untreated) values."""
    df = df.copy()
    df["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    inc = df["annual_inc"].where(df["annual_inc"] > 0)
    df["loan_to_income"] = df["loan_amnt"] / inc
    months = ((df["issue_date"].dt.year - df["earliest_cr_line"].dt.year) * 12
              + (df["issue_date"].dt.month - df["earliest_cr_line"].dt.month))
    df["credit_age_months"] = months.astype("float64")
    return df
