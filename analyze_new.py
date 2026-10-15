#!/usr/bin/env python3
"""
analyse.py -- Step 2 of 2: analyse the aggregated counts produced by aggregate.py.

RESEARCH QUESTION
    How does COVID-19 case fatality (deaths among cases with a KNOWN death_yn) differ by age group,
    sex and ethnicity, and how did it change per month?

DEFINITION
    case fatality (CFR) = deaths / cases with known outcome
    Cases whose death status is unknown are excluded from numerator AND denominator.

OUTPUTS (in --outdir)
    CSV tables, summary.txt (everything printed on screen), and PNG figures:
        fig1_missingness.png / fig1b_missingness_by_age.png   how complete is the outcome data?
        fig2_age_sex.png / fig2_age_sex_linear.png            CFR by age group and sex
        fig3_ethnicity.png                                     crude vs age-standardised CFR
        fig4_monthly_age.png / fig4_monthly_age_linear.png    monthly CFR per age band
        fig5_reporting_changes.png                             monthly case volume + reporting changes
    fig2 and fig4 are saved twice: log scale (handles the >100x range across groups) and linear
    scale (easier at a glance, but small groups are squeezed toward zero).

USAGE
    python analyse.py --input data/aggregated_counts.csv --outdir results
"""
import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no window needed: figures are only saved to files
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- configuration
MISSING_LABEL = "Missing/Unknown"   # same label as in aggregate.py
Z = 1.96                            # z-value for a 95% confidence interval
REQUIRED = ["month", "sex", "age_group", "race_eth", "death", "n"]  # columns aggregate.py must have written

# One colour palette, used in every figure so the report looks consistent.
PALETTE = {
    "blue": "#3B82F6",
    "orange": "#F59E0B",
    "red": "#EF4444",
    "green": "#10B981",
    "grey": "#6B7280",
}

# Everything passed to log() is printed AND stored here, then written to summary.txt at the end.
LOG_LINES = []

# Documented, dated changes to which jurisdictions report into this dataset (CDC's own notes on the
# public-use file) - NOT inferred from the data. The file has no geography column, so these cannot
# be excluded directly; they are only used to check whether they show up as a step in case volume.
REPORTING_CHANGES = [
    ("2021-11-08", "Iowa stops reporting"),
    ("2023-05-02", "Oklahoma stops reporting"),
    ("2023-05-12", "Kansas stops reporting"),
    ("2023-05-23", "New Hampshire stops reporting"),
    ("2023-07-13", "Oregon: pediatric deaths only"),
    ("2023-10-31", "Louisiana stops reporting"),
    ("2024-01-01", "Kentucky stops reporting"),
    ("2024-07-01", "CDC reporting ends"),
]


def log(msg=""):
    """Print a message and remember it for summary.txt."""
    print(msg)
    LOG_LINES.append(str(msg))


# --------------------------------------------------------------------------- plot style
def set_style():
    """One shared, modern look for every figure (clean sans-serif, dashed light gridlines,
    no top/right border). Called once from main() before any figure is drawn."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d6d6de",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#3a3a3a",
        "axes.titlesize": 14.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 13,
        "axes.labelsize": 10.5,
        "axes.labelweight": "medium",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "font.size": 10.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "grid.color": "#e8e8ee",
        "grid.linewidth": 0.8,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.markersize": 5.5,
        "lines.solid_capstyle": "round",
    })


def percent_formatter():
    """Axis formatter turning a fraction into a percentage: 0.0005 -> '0.05%', 0.2 -> '20%'.
    Used on every fatality axis (log or linear) instead of raw decimals or '10^-1'."""
    def fmt(x, _pos):
        if x <= 0:
            return ""
        pct = x * 100
        return f"{pct:g}%" if pct >= 1 else f"{pct:.2g}%"
    return FuncFormatter(fmt)


# --------------------------------------------------------------------------- statistics
def wilson_ci(k, n, z=Z):
    """
    Wilson 95% confidence interval for a proportion k/n (k deaths out of n known outcomes).

    Preferred over the simple 'p +- z*sqrt(p(1-p)/n)' interval because it stays inside [0, 1]
    and behaves well for very small or very large proportions (many of our groups have CFR near 0).
    Returns NaN where n == 0 (never divides by zero). Works on numbers or numpy arrays.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    safe_n = np.where(n > 0, n, np.nan)          # avoid division by zero
    p = k / safe_n
    denom = 1 + z**2 / safe_n
    centre = (p + z**2 / (2 * safe_n)) / denom
    half = z * np.sqrt(p * (1 - p) / safe_n + z**2 / (4 * safe_n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def odds_ratio(d1, s1, d0, s0, z=Z):
    """
    Odds ratio of death for group 1 versus reference group 0, from a 2x2 table:

                      died   survived
        group 1        d1       s1
        group 0 (ref)  d0       s0

    Returns (odds_ratio, ci_low, ci_high, p_value).
      * CI uses Woolf's method: log(OR) is approximately normal with
        standard error sqrt(1/d1 + 1/s1 + 1/d0 + 1/s0).
      * p_value is the two-sided Wald test of H0: OR = 1  (z = log(OR)/SE, normal distribution).
      * If any cell is 0, 0.5 is added to every cell (the usual continuity correction).
      * Returns four NaN for impossible input (negative/NaN cells or an empty group).
    """
    cells = np.array([d1, s1, d0, s0], dtype=float)
    if np.any(cells < 0) or np.any(np.isnan(cells)) or (cells[0] + cells[1] == 0) or (cells[2] + cells[3] == 0):
        return np.nan, np.nan, np.nan, np.nan
    if np.any(cells == 0):
        cells = cells + 0.5
    a, b, c, d = cells
    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    # Two-sided normal p-value without scipy: P(|Z| > z) = erfc(|z| / sqrt(2)).
    p_value = math.erfc(abs(log_or / se) / math.sqrt(2))
    return np.exp(log_or), np.exp(log_or - z * se), np.exp(log_or + z * se), p_value


def add_rate(df):
    """Add 'fatality' (deaths/known) and its Wilson CI to a table that has 'deaths' and 'known' columns."""
    df = df.copy()
    df["fatality"] = np.where(df["known"] > 0, df["deaths"] / df["known"].where(df["known"] > 0), np.nan)
    df["ci_low"], df["ci_high"] = wilson_ci(df["deaths"], df["known"])
    return df


# --------------------------------------------------------------------------- loading / helpers
def age_sort_key(label):
    """Numeric sort key for an age label: '0 - 9 Years' -> 0, '80+ Years' -> 80; None if no leading number."""
    m = re.match(r"^\s*(\d+)", str(label))
    return int(m.group(1)) if m else None


def load(path):
    """Read aggregated_counts.csv, validate it, and add 'year' and 'age_num' helper columns."""
    p = Path(path)
    if not p.is_file():
        sys.exit(f"ERROR: {p} not found. Run aggregate.py first.")
    df = pd.read_csv(p, dtype={"month": str, "sex": str, "age_group": str, "race_eth": str, "death": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {p} lacks column(s) {missing}; expected {REQUIRED}")
    # Keep only rows with a positive integer count.
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df = df[df["n"].notna() & (df["n"] > 0)].copy()
    df["n"] = df["n"].astype("int64")
    for c in ["sex", "age_group", "race_eth", "death", "month"]:
        df[c] = df[c].fillna(MISSING_LABEL).astype(str).str.strip()
    if df.empty:
        sys.exit("ERROR: the aggregated file contains no usable rows.")
    df["year"] = pd.to_numeric(df["month"].str[:4], errors="coerce")
    df.loc[df["month"] == "Invalid", "year"] = np.nan
    df["age_num"] = df["age_group"].map(age_sort_key)
    return df


def known_only(df):
    """Keep only rows with known death status and add 'deaths' (n if Yes, else 0) and 'known' (n) columns."""
    k = df[df["death"].isin(["Yes", "No"])].copy()
    k["deaths"] = np.where(k["death"] == "Yes", k["n"], 0)
    k["known"] = k["n"]
    return k


def summarise(k, by):
    """Sum deaths and known-outcome cases per group (columns in `by`), then add CFR + Wilson CI."""
    g = k.groupby(by, observed=True, dropna=False)[["deaths", "known"]].sum().reset_index()
    return add_rate(g)


def log_floor(*series):
    """Lower bound for a log-scale axis: smallest positive value / 3, so zeros never break a plot."""
    vals = np.concatenate([np.asarray(x, dtype=float).ravel() for x in series])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(vals.min() / 3) if vals.size else 1e-6


def short_eth(label):
    """Shorten an ethnicity label for the y-axis: 'White, Non-Hispanic' -> 'White '."""
    return str(label).replace(", Non-Hispanic", " ")


def _bar_chart(values, labels, ylabel, xlabel, title, path, figsize, rotate=False):
    """Shared bar chart (percent known) used by both missingness figures."""
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(values))
    bars = ax.bar(xs, values, color=PALETTE["blue"], width=0.6, zorder=3)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9.5, color="#333333")  # value on top of each bar
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 100)
    ax.set_title(title, color="#1a1a1a")
    ax.grid(axis="y", alpha=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------- analysis 1: missing outcomes
def missingness(df, out):
    """
    How many cases have a known death status, per year and per age group?
    Important because the CFR is only computed on known outcomes: if 'unknown' is common
    (and changes over time) the CFR may be biased.
    Uses ALL rows of `df` (not only known ones).
    """
    log("=" * 70)
    log("1. MISSING DEATH STATUS")
    total = df["n"].sum()
    known = df.loc[df["death"].isin(["Yes", "No"]), "n"].sum()
    log(f"Total cases in file:         {total:>14,}")
    log(f"Known death status:          {known:>14,}  ({100 * known / total:.1f}%)")
    log(f"Missing/Unknown death status:{total - known:>14,}  ({100 * (total - known) / total:.1f}%)")
    inv = df.loc[df["month"] == "Invalid", "n"].sum()
    log(f"Rows with invalid/absent date (excluded from time analysis): {inv:,} ({100 * inv / total:.2f}%)")

    # ---- per year
    d = df.assign(
        known_n=np.where(df["death"].isin(["Yes", "No"]), df["n"], 0),
        year_label=df["year"].map(lambda y: "Invalid" if pd.isna(y) else str(int(y))),
    )
    by_year = d.groupby("year_label").agg(cases=("n", "sum"), known=("known_n", "sum")).reset_index()
    by_year["pct_known"] = 100 * by_year["known"] / by_year["cases"]
    by_year.to_csv(out / "missingness_by_year.csv", index=False)
    log("\nPercentage with known death status per year:")
    log(by_year.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    shown = by_year[by_year["year_label"] != "Invalid"]   # 'Invalid' has no x-position on a year axis
    if not shown.empty:
        _bar_chart(shown["pct_known"], shown["year_label"], "Cases with known death status (%)",
                   "Year of earliest case date", "How complete is the outcome data?",
                   out / "fig1_missingness.png", (6.2, 3.9))

    # ---- per age group
    by_age = d.groupby("age_group", observed=True).agg(cases=("n", "sum"), known=("known_n", "sum")).reset_index()
    by_age["pct_known"] = 100 * by_age["known"] / by_age["cases"]
    by_age["age_num"] = by_age["age_group"].map(age_sort_key)
    by_age = by_age.sort_values("age_num", na_position="last")
    by_age.to_csv(out / "missingness_by_age.csv", index=False)
    log("\nPercentage with known death status per age group:")
    log(by_age.drop(columns="age_num").to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # 'Missing/Unknown' age has no numeric position: it is logged in the table but not drawn.
    shown_age = by_age[by_age["age_num"].notna()]
    if not shown_age.empty:
        _bar_chart(shown_age["pct_known"], [a.replace(" Years", "") for a in shown_age["age_group"]],
                   "Cases with known death status (%)", "Age group (years)",
                   "How complete is the outcome data, by age group?",
                   out / "fig1b_missingness_by_age.png", (7.4, 3.9), rotate=True)


# --------------------------------------------------------------------------- analysis 5: reporting changes
def _group_change_dates(changes, days=14):
    """Group dates that lie within `days` of each other, so labels of near-simultaneous changes
    (e.g. three states within three weeks in May 2023) are combined instead of overlapping."""
    changes = sorted(((pd.Timestamp(d), label) for d, label in changes), key=lambda t: t[0])
    groups = []
    for cutoff, label in changes:
        if groups and (cutoff - groups[-1][-1][0]).days <= days:
            groups[-1].append((cutoff, label))
        else:
            groups.append([(cutoff, label)])
    return groups


def _short_name(label):
    """'Oklahoma stops reporting' -> 'Oklahoma'; 'Oregon: pediatric deaths only' -> 'Oregon'."""
    return label.split(" stops")[0].split(":")[0]


def reporting_changes(df, out):
    """
    Plot monthly case VOLUME (not fatality) with the documented jurisdiction drop-out dates marked,
    so a real step in reporting can be told apart from ordinary month-to-month noise.
    Two outputs: a number (mean volume 2 months before vs after each date) and a plot,
    so the conclusion does not rely on eyeballing a line.
    """
    log("\n" + "=" * 70)
    log("5. REPORTING CHANGES OVER TIME (documented jurisdiction drop-outs)")
    d = df[df["month"] != "Invalid"].copy()
    d["date"] = pd.to_datetime(d["month"] + "-01", format="%Y-%m-%d", errors="coerce")
    d = d[d["date"].notna()]
    monthly = d.groupby("date")["n"].sum().reset_index().sort_values("date")
    monthly.to_csv(out / "monthly_case_volume.csv", index=False)
    if monthly.empty:
        log("No usable rows for this check.")
        return

    # ---- numeric check: mean monthly volume 2 months before vs after each documented change
    log("\nMonthly case volume around each documented change (mean of the 2 months before vs after):")
    s = monthly.set_index("date")["n"]
    for date_str, label in REPORTING_CHANGES:
        cutoff = pd.Timestamp(date_str)
        before = s[(s.index < cutoff) & (s.index >= cutoff - pd.DateOffset(months=2))]
        after = s[(s.index >= cutoff) & (s.index < cutoff + pd.DateOffset(months=2))]
        if len(before) and len(after):
            pct = 100 * (after.mean() - before.mean()) / before.mean()
            log(f"  {date_str}  {label:<32} before={before.mean():>10,.0f}  after={after.mean():>10,.0f}  change={pct:+.1f}%")
        else:
            log(f"  {date_str}  {label:<32} (not enough months on one side to compare)")

    # ---- figure: volume line + dashed vertical line per documented change
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(monthly["date"], monthly["n"], "-o", ms=3, lw=1.6, color=PALETTE["blue"], zorder=3)
    ax.fill_between(monthly["date"], 0, monthly["n"], color=PALETTE["blue"], alpha=0.08, zorder=1)
    ymax = monthly["n"].max()
    lo, hi = monthly["date"].min(), monthly["date"].max()
    for group in _group_change_dates(REPORTING_CHANGES):
        visible = [(c, l) for c, l in group if lo <= c <= hi]   # skip changes outside the plotted range
        if not visible:
            continue
        for cutoff, _ in visible:
            ax.axvline(cutoff, color=PALETTE["grey"], lw=1, ls="--", alpha=0.5, zorder=2)
        label_text = (" / ".join(_short_name(l) for _, l in visible) + " stop reporting"
                      if len(visible) > 1 else visible[0][1])
        ax.annotate(label_text, xy=(visible[-1][0], ymax), xytext=(3, 0), textcoords="offset points",
                    rotation=90, va="top", ha="left", fontsize=8, color="#555555")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Total cases reported per month")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Monthly case volume, with documented reporting changes", color="#1a1a1a")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "fig5_reporting_changes.png", dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------- analysis 2: age and sex
def _plot_age_sex(order, xpos, plot_tab, floor, log_scale, title, path):
    """Draw the age/sex figure in log or linear scale (shared by both calls in age_sex())."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # The two sexes are shifted slightly left/right so their error bars do not overlap.
    for sex, colour, shift in [("Female", PALETTE["red"], -0.08), ("Male", PALETTE["blue"], 0.08)]:
        s = plot_tab[plot_tab["sex"] == sex]
        if s.empty:
            continue
        x = np.array([xpos[a] for a in s["age_group"]]) + shift
        lo_bound = floor if log_scale else 0
        lower = s["fatality"] - s["ci_low"].clip(lower=lo_bound)   # error-bar length below the point
        ax.errorbar(x, s["fatality"], yerr=[lower, s["ci_high"] - s["fatality"]],
                    fmt="o-", lw=1.8, ms=6.5, color=colour, ecolor=colour, elinewidth=1.3,
                    capsize=3, markeredgecolor="white", markeredgewidth=1.1, label=sex, zorder=3)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        scale_note = "log scale"
    else:
        ax.set_ylim(bottom=0)
        scale_note = "linear scale"
    ax.yaxis.set_major_formatter(percent_formatter())
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o.replace(" Years", "") for o in order], rotation=45, ha="right")
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel(f"Case fatality (% of known outcomes), {scale_note}")
    ax.set_title(title, color="#1a1a1a")
    ax.grid(alpha=0.6 if log_scale else 0.5, which="major" if log_scale else "both", zorder=0)
    if log_scale:
        ax.grid(alpha=0.2, which="minor", zorder=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def age_sex(k, out, min_n):
    """
    CFR per age group and sex, plus two sets of odds ratios (with 95% CI and p-value):
      * each age group versus the youngest group (sexes pooled),
      * male versus female WITHIN each age group (so age differences cannot distort it).
    `k` = known-outcome rows, `min_n` = minimum known outcomes for a group to be reported.
    """
    log("\n" + "=" * 70)
    log("2. FATALITY BY AGE GROUP AND SEX (known death status only)")
    d = k[k["age_num"].notna() & k["sex"].isin(["Female", "Male"])]
    excluded = k["known"].sum() - d["known"].sum()
    log(f"Excluded (age or sex unknown / other): {excluded:,} of {k['known'].sum():,} known-outcome cases")
    if d.empty:
        log("No usable rows for this analysis.")
        return None
    tab = summarise(d, ["age_num", "age_group", "sex"]).sort_values(["age_num", "sex"])
    tab.to_csv(out / "fatality_by_age_sex.csv", index=False)
    log(tab[["age_group", "sex", "deaths", "known", "fatality", "ci_low", "ci_high"]].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"))

    # p-values are tiny, so show them in scientific notation (0.00 would be unhelpful).
    fmt = {"p_value": lambda x: f"{x:.1e}"}

    # ---- odds ratio per age group versus the youngest group with enough data (sexes pooled).
    # Pooled from `tab` (already aggregated) so the totals match tab exactly.
    pooled = tab.groupby(["age_num", "age_group"], observed=True)[["deaths", "known"]].sum().reset_index()
    pooled = add_rate(pooled).sort_values("age_num")
    pooled = pooled[pooled["known"] >= min_n]
    if len(pooled) >= 2:
        ref = pooled.iloc[0]   # reference group = youngest age group
        rows = []
        for _, r in pooled.iterrows():
            o, lo, hi, p = odds_ratio(r["deaths"], r["known"] - r["deaths"], ref["deaths"], ref["known"] - ref["deaths"])
            rows.append({"age_group": r["age_group"], "reference": ref["age_group"],
                         "odds_ratio": o, "ci_low": lo, "ci_high": hi, "p_value": p})
        orr = pd.DataFrame(rows)
        orr.to_csv(out / "odds_ratio_age.csv", index=False)
        log(f"\nOdds ratio of death vs {ref['age_group']} (crude, both sexes):")
        log(orr.to_string(index=False, float_format=lambda x: f"{x:,.2f}", formatters=fmt))

    # ---- odds ratio male vs female WITHIN each age group.
    # Pivot so every age group is one row with Female/Male columns. fillna(0): a missing combination counts as 0.
    piv = tab.pivot(index=["age_num", "age_group"], columns="sex", values=["deaths", "known"]).fillna(0)
    srows = []
    for (num, label), row in piv.iterrows():
        f_known, m_known = row.get(("known", "Female"), 0), row.get(("known", "Male"), 0)
        if f_known >= min_n and m_known >= min_n:   # need enough cases in BOTH sexes
            f_deaths, m_deaths = row[("deaths", "Female")], row[("deaths", "Male")]
            o, lo, hi, p = odds_ratio(m_deaths, m_known - m_deaths, f_deaths, f_known - f_deaths)
            srows.append({"age_group": label, "age_num": num, "odds_ratio_male_vs_female": o,
                          "ci_low": lo, "ci_high": hi, "p_value": p})
    if srows:
        sr = pd.DataFrame(srows).sort_values("age_num").drop(columns="age_num")
        sr.to_csv(out / "odds_ratio_sex_within_age.csv", index=False)
        log("\nOdds ratio of death, male vs female, within each age group (p = two-sided Wald test of OR = 1):")
        log(sr.to_string(index=False, float_format=lambda x: f"{x:,.2f}", formatters=fmt))

    # ---- figures: one series per sex, saved twice (log + linear)
    order = list(tab.drop_duplicates("age_num").sort_values("age_num")["age_group"])
    xpos = {a: i for i, a in enumerate(order)}          # age group -> x position
    base = tab[tab["known"] >= min_n]                    # hide groups with too few cases
    plot_log = base[base["deaths"] > 0]                  # zero deaths cannot be drawn on a log axis
    if len(plot_log) < len(base):
        log("(Groups with 0 deaths are excluded from the log-scale fig2 - a log axis cannot show "
            "zero. They ARE included in fig2_age_sex_linear.png.)")
    floor = log_floor(plot_log["fatality"])
    title = "Case fatality by age group and sex (95% CI)"
    _plot_age_sex(order, xpos, plot_log, floor, True, title, out / "fig2_age_sex.png")
    _plot_age_sex(order, xpos, base, floor, False, title, out / "fig2_age_sex_linear.png")
    return tab


# --------------------------------------------------------------------------- analysis 3: ethnicity
def age_standardised(d, group_col, min_cell):
    """
    Direct age standardisation, vectorised over groups (pivot tables + numpy, no per-group loop).

    IDEA: groups differ in age structure (e.g. White patients are older on average), and age is the
    strongest risk factor, so crude CFRs are not comparable. We ask instead: "what would this
    group's CFR be if it had the same age mix as everyone combined?"
        standardised CFR = sum over age bands of  weight(band) * CFR_group(band)
        weight(band)     = share of that age band among ALL known-outcome cases (the 'standard')

    Rules for sparse data:
      * an age band with fewer than `min_cell` known outcomes in a group is dropped and the
        remaining weights are renormalised to sum to 1;
      * if the remaining bands cover < 50% of the standard weight, the estimate is withheld (NaN).
    The standard error uses a smoothed variance ((deaths+0.5)/(known+1)) so zero-death bands do not
    give zero variance. Returns one row per group.
    """
    deaths = d.pivot_table(index=group_col, columns="age_num", values="deaths", aggfunc="sum", fill_value=0)
    known = d.pivot_table(index=group_col, columns="age_num", values="known", aggfunc="sum", fill_value=0)
    std = d.groupby("age_num")["known"].sum().reindex(known.columns).fillna(0)
    std = std / std.sum()                                   # standard age distribution (weights sum to 1)

    valid = known >= min_cell                               # which (group, age band) cells are usable
    known_v = known.where(valid)                            # NaN at cells excluded by min_cell
    deaths_v = deaths.where(valid, 0.0)
    coverage = valid.mul(std, axis=1).sum(axis=1)           # share of the standard covered per group
    w = valid.mul(std, axis=1).div(coverage.replace(0, np.nan), axis=0)   # renormalised weights

    p = (deaths_v / known_v).fillna(0)                      # age-specific CFR per group
    p_var = (deaths_v + 0.5) / (known_v.fillna(0) + 1)      # smoothed proportion for the variance
    rate = (w * p).sum(axis=1)                              # the standardised CFR
    se = np.sqrt((w**2 * p_var * (1 - p_var) / known_v).fillna(0).sum(axis=1))

    result = pd.DataFrame({
        "std_fatality": rate,
        "std_ci_low": (rate - Z * se).clip(lower=0),
        "std_ci_high": (rate + Z * se).clip(upper=1),
        "age_coverage": coverage,
    })
    result.loc[coverage < 0.5, ["std_fatality", "std_ci_low", "std_ci_high"]] = np.nan
    return result.reset_index()   # index name = group_col (set automatically by pivot_table)


def ethnicity(k, out, min_n):
    """Crude and age-standardised CFR per ethnic group, shown together in one dot-and-line figure."""
    log("\n" + "=" * 70)
    log("3. FATALITY BY ETHNICITY (known death status only)")
    total_known = k["known"].sum()
    d = k[k["age_num"].notna() & (k["race_eth"] != MISSING_LABEL)]
    log(f"Known-outcome cases with known ethnicity and age: {d['known'].sum():,} of {total_known:,} "
        f"({100 * d['known'].sum() / total_known:.1f}%)")
    if d.empty:
        log("No usable rows for this analysis.")
        return
    crude = summarise(d, ["race_eth"])
    crude = crude[crude["known"] >= min_n].rename(columns={
        "fatality": "crude_fatality", "ci_low": "crude_ci_low", "ci_high": "crude_ci_high"})
    std = age_standardised(d, "race_eth", min_cell=20)
    tab = crude.merge(std, on="race_eth", how="left").sort_values("crude_fatality", ascending=False)
    tab.to_csv(out / "fatality_by_ethnicity.csv", index=False)
    log("Crude vs age-standardised case fatality (standard = age mix of all known cases):")
    show = tab[["race_eth", "deaths", "known", "crude_fatality", "std_fatality", "std_ci_low", "std_ci_high"]]
    log(show.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    log("Note: crude differences mostly reflect different age structures; compare the standardised column.")

    # ---- figure: grey dot = crude, red square = age-standardised, thin line connects the two
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(tab))
    s = tab.dropna(subset=["std_fatality"])                 # groups whose standardised value was not withheld
    ys = np.array([list(tab["race_eth"]).index(r) for r in s["race_eth"]])
    # Connector first (zorder 1) so it sits behind the markers: shows how far each group moved.
    for yi, (_, r) in zip(ys, s.iterrows()):
        ax.plot([r["crude_fatality"], r["std_fatality"]], [yi, yi], color="#d6d6de", lw=1.3, zorder=1)
    ax.errorbar(tab["crude_fatality"], y,
                xerr=[tab["crude_fatality"] - tab["crude_ci_low"], tab["crude_ci_high"] - tab["crude_fatality"]],
                fmt="o", color=PALETTE["grey"], capsize=3, markeredgecolor="white", markeredgewidth=1.0,
                label="Crude", zorder=3)
    ax.errorbar(s["std_fatality"], ys,
                xerr=[s["std_fatality"] - s["std_ci_low"], s["std_ci_high"] - s["std_fatality"]],
                fmt="s", color=PALETTE["red"], capsize=3, markeredgecolor="white", markeredgewidth=1.0,
                label="Age-standardised", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([short_eth(r) for r in tab["race_eth"]])
    ax.invert_yaxis()                                       # highest crude CFR at the top
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.set_xlabel("Case fatality (% of known outcomes)")
    ax.set_title("Case fatality by ethnicity (95% CI)", color="#1a1a1a")
    ax.grid(alpha=0.6, axis="x")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig3_ethnicity.png", dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------- analysis 4: over time
def _plot_over_time(band_month, floor, log_scale, title, path):
    """Draw the monthly-CFR figure in log or linear scale (shared by both calls in over_time())."""
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    band_colours = {"0-39": PALETTE["green"], "40-59": PALETTE["blue"],
                    "60-79": PALETTE["orange"], "80+": PALETTE["red"]}
    for b, colour in band_colours.items():
        s = band_month[band_month["band"] == b].sort_values("date")
        if s.empty:
            continue
        if log_scale:
            rate = s["fatality"].where(s["fatality"] > 0)   # 0 deaths -> gap in the line (log axis)
            lo_clip = s["ci_low"].clip(lower=floor)
        else:
            rate = s["fatality"]                            # 0 is a valid point on a linear axis
            lo_clip = s["ci_low"].clip(lower=0)
        ax.plot(s["date"], rate, "-o", ms=3.2, lw=1.7, color=colour, label=f"{b} years",
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.fill_between(s["date"], lo_clip, s["ci_high"], color=colour, alpha=0.15, zorder=1)  # 95% CI band
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        scale_note = "log scale"
    else:
        ax.set_ylim(bottom=0)
        scale_note = "linear scale"
    ax.yaxis.set_major_formatter(percent_formatter())
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel(f"Monthly case fatality, {scale_note}")
    ax.set_title(title, color="#1a1a1a")
    ax.grid(alpha=0.6 if log_scale else 0.5, which="major" if log_scale else "both")
    if log_scale:
        ax.grid(alpha=0.2, which="minor")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))   # legend below the plot
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def age_band(age_num):
    """Group a 10-year age band (by its lower bound) into four broader bands so the time plot stays readable."""
    return "0-39" if age_num < 40 else "40-59" if age_num < 60 else "60-79" if age_num < 80 else "80+"


def over_time(k, out, min_n):
    """CFR per calendar year per age group, and per month for four broad age bands."""
    log("\n" + "=" * 70)
    log("4. FATALITY OVER TIME (known death status only)")
    d = k[k["age_num"].notna() & (k["month"] != "Invalid")].copy()
    if d.empty:
        log("No usable rows for this analysis.")
        return
    d["date"] = pd.to_datetime(d["month"] + "-01", format="%Y-%m-%d", errors="coerce")
    d = d[d["date"].notna()]

    # ---- yearly CFR of the oldest age group (printed) and of all age groups (CSV)
    yearly = summarise(d.assign(year=d["date"].dt.year), ["year", "age_num", "age_group"]).sort_values(["age_num", "year"])
    yearly.to_csv(out / "fatality_by_year_age.csv", index=False)
    old = yearly[yearly["age_num"] == yearly["age_num"].max()]
    log(f"Oldest age group ({old['age_group'].iloc[0] if len(old) else 'n/a'}), fatality per year:")
    log(old[["year", "deaths", "known", "fatality", "ci_low", "ci_high"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- monthly CFR per 10-year age group (CSV only)
    monthly = summarise(d, ["date", "age_num", "age_group"]).sort_values(["age_num", "date"])
    monthly.to_csv(out / "fatality_by_month_age.csv", index=False)

    # ---- monthly CFR per broad age band (this one is plotted)
    d["band"] = d["age_num"].map(age_band)
    band_month = summarise(d, ["date", "band"])
    band_month = band_month[band_month["known"] >= min_n]    # drop months with too few known outcomes
    band_month.to_csv(out / "fatality_by_month_ageband.csv", index=False)
    if band_month.empty:
        log(f"No month has >= {min_n} known outcomes; time plot skipped (lower --min-n).")
        return

    floor = log_floor(band_month["fatality"])
    title = f"Case fatality per month by age band (months with >= {min_n} known outcomes)"
    _plot_over_time(band_month, floor, True, title, out / "fig4_monthly_age.png")
    _plot_over_time(band_month, floor, False, title, out / "fig4_monthly_age_linear.png")
    log("\n(fig4 saved twice: log scale and linear scale. On the linear version the 0-39 and "
        "40-59 bands look close to flat against 80+ - that is the real gap in magnitude, "
        "not a rendering issue. Use the log version to compare across all four bands at once.)")


# --------------------------------------------------------------------------- main
def main():
    """Parse arguments, load the aggregated data, run the five analyses, write summary.txt."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/aggregated_counts.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--min-n", type=int, default=100, help="minimum known outcomes for a group/month to be shown")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    set_style()
    df = load(args.input)          # all cases (needed for the missingness analysis)
    k = known_only(df)             # only cases with a known outcome (needed for every CFR)
    if k.empty:
        sys.exit("ERROR: no case has a known death status (Yes/No); nothing to analyse.")

    # Order of the calls = order of the output. (Section numbers in the log match the figure numbers.)
    missingness(df, out)
    reporting_changes(df, out)
    age_sex(k, out, args.min_n)
    ethnicity(k, out, args.min_n)
    over_time(k, out, args.min_n)
    (out / "summary.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    print(f"\nDone. Tables, figures and summary.txt are in {out}/")


if __name__ == "__main__":
    main()