"""
test_project.py -- automated tests for aggregate.py and analyse.py.

Run with:   python -m pytest -v

The tests are grouped by the testing approaches from the lecture notes (Chapter 4):
    * black box   : check inputs -> expected outputs, without looking at the implementation
    * glass box   : check each branch inside a function (edge cases, zero cells, empty input)
    * random      : many random inputs, check properties that must ALWAYS hold
    * integration : run the whole pipeline on a tiny hand-made dataset
Re-running the whole file after every change is "regression testing".
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))   # so `import analyse` works from any folder
import aggregate
import analyse


# =========================================================================== black box tests
class TestBlackBox:
    def test_to_month_slash_and_dash_formats(self):
        s = pd.Series(["2022/10/13", "2021-03-05", "2020/1/9"])
        assert list(aggregate.to_month(s)) == ["2022-10", "2021-03", "2020-01"]

    def test_to_month_invalid_values(self):
        # typo years, impossible months, empty and garbage text must all become 'Invalid'
        s = pd.Series(["1999/01/01", "2022/13/01", "", "abc", None])
        assert list(aggregate.to_month(s)) == ["Invalid"] * 5

    def test_clean_death_only_returns_three_labels(self):
        s = pd.Series(["Yes", "yes ", "NO", "Unknown", "Missing", "", None, "maybe"])
        out = set(aggregate.clean_death(s))
        assert out <= {"Yes", "No", aggregate.MISSING_LABEL}
        assert list(aggregate.clean_death(pd.Series(["yes ", "NO"]))) == ["Yes", "No"]

    def test_clean_category_unifies_spelling(self):
        s = pd.Series(["  female", "FEMALE", "Female", "N/A", None])
        assert list(aggregate.clean_category(s)) == ["Female", "Female", "Female",
                                                     aggregate.MISSING_LABEL, aggregate.MISSING_LABEL]

    def test_odds_ratio_hand_calculation(self):
        # (10 * 95) / (90 * 5) = 2.111...
        o, lo, hi, p = analyse.odds_ratio(10, 90, 5, 95)
        assert o == pytest.approx(950 / 450)
        assert lo < o < hi

    def test_odds_ratio_equal_groups_gives_one(self):
        o, lo, hi, p = analyse.odds_ratio(20, 80, 20, 80)
        assert o == pytest.approx(1.0)
        assert p == pytest.approx(1.0)          # no difference -> p = 1
        assert lo < 1 < hi

    def test_odds_ratio_clear_difference_is_significant(self):
        _, lo, _, p = analyse.odds_ratio(500, 500, 100, 900)
        assert p < 1e-10
        assert lo > 1

    def test_wilson_ci_known_value(self):
        # 10 successes out of 100 -> Wilson 95% CI is about (0.055, 0.174)
        lo, hi = analyse.wilson_ci(10, 100)
        assert float(lo) == pytest.approx(0.0552, abs=1e-3)
        assert float(hi) == pytest.approx(0.1744, abs=1e-3)

    def test_age_sort_key(self):
        assert analyse.age_sort_key("0 - 9 Years") == 0
        assert analyse.age_sort_key("80+ Years") == 80
        assert analyse.age_sort_key("Missing/Unknown") is None

    def test_age_band(self):
        assert [analyse.age_band(a) for a in (0, 30, 40, 59, 60, 79, 80)] == \
               ["0-39", "0-39", "40-59", "40-59", "60-79", "60-79", "80+"]


# =========================================================================== glass box tests
class TestGlassBox:
    def test_wilson_ci_zero_n_gives_nan_not_error(self):
        lo, hi = analyse.wilson_ci(0, 0)                  # branch: n == 0
        assert np.isnan(lo) and np.isnan(hi)

    def test_wilson_ci_zero_deaths_lower_bound_is_zero(self):
        lo, hi = analyse.wilson_ci(0, 50)                 # branch: p == 0
        assert float(lo) == 0.0 and 0 < float(hi) < 1

    def test_odds_ratio_zero_cell_uses_continuity_correction(self):
        o, lo, hi, p = analyse.odds_ratio(0, 100, 5, 95)  # branch: a cell == 0 -> +0.5 to all
        assert np.isfinite(o) and o > 0

    def test_odds_ratio_invalid_input_gives_nan(self):
        assert all(np.isnan(x) for x in analyse.odds_ratio(-1, 10, 5, 5))     # negative cell
        assert all(np.isnan(x) for x in analyse.odds_ratio(0, 0, 5, 5))       # empty group 1
        assert all(np.isnan(x) for x in analyse.odds_ratio(1, 1, np.nan, 5))  # NaN cell

    def test_add_rate_handles_zero_known(self):
        df = pd.DataFrame({"deaths": [0, 5], "known": [0, 50]})
        out = analyse.add_rate(df)
        assert np.isnan(out.loc[0, "fatality"])           # no division by zero
        assert out.loc[1, "fatality"] == pytest.approx(0.1)

    def test_known_only_drops_missing(self):
        df = pd.DataFrame({"death": ["Yes", "No", "Missing/Unknown"], "n": [3, 7, 100]})
        k = analyse.known_only(df)
        assert len(k) == 2
        assert k["deaths"].sum() == 3 and k["known"].sum() == 10

    def test_age_standardised_equal_age_mix_equals_crude(self):
        # Two groups with the SAME age mix and same rates -> standardised == crude (here 10% and 50%).
        d = pd.DataFrame({
            "race_eth": ["A", "A", "B", "B"],
            "age_num": [20, 80, 20, 80],
            "deaths": [10, 50, 10, 50],
            "known": [100, 100, 100, 100],
        })
        out = analyse.age_standardised(d, "race_eth", min_cell=20).set_index("race_eth")
        assert out.loc["A", "std_fatality"] == pytest.approx(0.30)
        assert out.loc["B", "std_fatality"] == pytest.approx(0.30)

    def test_age_standardised_removes_age_confounding(self):
        # Group Y is much older, so its CRUDE rate is higher, but both groups have identical
        # age-specific rates -> the standardised rates must be equal.
        d = pd.DataFrame({
            "race_eth": ["X", "X", "Y", "Y"],
            "age_num": [20, 80, 20, 80],
            "deaths": [9, 5, 1, 40],                       # rates: young 1%, old 50% in both groups
            "known":  [900, 10, 100, 80],
        })
        # rate X young = 9/900 = 1%, old = 5/10 = 50%; Y young = 1/100 = 1%, old = 40/80 = 50%
        out = analyse.age_standardised(d, "race_eth", min_cell=5).set_index("race_eth")
        assert out.loc["X", "std_fatality"] == pytest.approx(out.loc["Y", "std_fatality"])

    def test_age_standardised_withholds_when_coverage_too_low(self):
        # Group C only has data for the youngest band (small share of the standard) -> NaN.
        d = pd.DataFrame({
            "race_eth": ["C", "D", "D"],
            "age_num": [20, 20, 80],
            "deaths": [1, 1, 50],
            "known": [100, 100, 900],
        })
        out = analyse.age_standardised(d, "race_eth", min_cell=20).set_index("race_eth")
        assert np.isnan(out.loc["C", "std_fatality"])
        assert not np.isnan(out.loc["D", "std_fatality"])


# =========================================================================== random tests
class TestRandom:
    def test_wilson_ci_always_valid(self):
        rng = np.random.default_rng(42)                   # fixed seed -> reproducible test
        for _ in range(500):
            n = int(rng.integers(1, 10_000))
            k = int(rng.integers(0, n + 1))
            lo, hi = analyse.wilson_ci(k, n)
            p = k / n
            assert 0.0 <= float(lo) <= p + 1e-12 <= float(hi) + 2e-12 <= 1.0 + 3e-12

    def test_odds_ratio_ci_contains_estimate_and_p_in_range(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            cells = rng.integers(1, 5_000, size=4)
            o, lo, hi, p = analyse.odds_ratio(*cells)
            assert lo <= o <= hi
            assert 0.0 <= p <= 1.0

    def test_odds_ratio_swapping_groups_inverts(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            a, b, c, d = rng.integers(1, 1_000, size=4)
            o1 = analyse.odds_ratio(a, b, c, d)[0]
            o2 = analyse.odds_ratio(c, d, a, b)[0]
            assert o1 * o2 == pytest.approx(1.0)


# =========================================================================== integration tests
RAW_HEADER = "cdc_case_earliest_dt,sex,age_group,race_ethnicity_combined,death_yn\n"
RAW_ROWS = [
    '2021/03/05,Female,20 - 29 Years,"White, Non-Hispanic",Yes',
    '2021/03/09,female ,20 - 29 Years,"White, Non-Hispanic",Yes',   # same cell after cleaning -> n = 2
    '2021/03/10,Male,80+ Years,"Black, Non-Hispanic",No',
    '2021/04/01,Male,80+ Years,"Black, Non-Hispanic",Yes',
    'abc,Male,Missing,Unknown,Missing',                              # bad date, everything unknown
    '2022/01/15,Female,0 - 9 Years,"Hispanic/Latino",No',
]


@pytest.fixture
def tiny_raw_csv(tmp_path):
    """A 6-row raw CSV in the CDC format."""
    p = tmp_path / "raw.csv"
    p.write_text(RAW_HEADER + "\n".join(RAW_ROWS) + "\n", encoding="utf-8")
    return p


class TestIntegration:
    def test_aggregate_counts_and_totals(self, tiny_raw_csv, tmp_path, monkeypatch):
        out = tmp_path / "agg.csv"
        monkeypatch.setattr(sys, "argv", ["aggregate.py", "--input", str(tiny_raw_csv), "--output", str(out)])
        assert aggregate.main() == 0                        # exit code 0 = success

        agg = pd.read_csv(out, keep_default_na=False)
        assert agg["n"].sum() == len(RAW_ROWS)              # every row counted exactly once
        # the two female / 2021-03 / 20-29 / White / Yes rows were merged into one cell with n = 2
        cell = agg[(agg["month"] == "2021-03") & (agg["sex"] == "Female") & (agg["death"] == "Yes")]
        assert len(cell) == 1 and int(cell["n"].iloc[0]) == 2
        # the row with the bad date is kept, labelled Invalid
        assert agg.loc[agg["month"] == "Invalid", "n"].sum() == 1

    def test_aggregate_missing_file_returns_error_code(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["aggregate.py", "--input", str(tmp_path / "nope.csv"),
                                          "--output", str(tmp_path / "o.csv")])
        assert aggregate.main() == 1

    def test_aggregate_missing_column_returns_error_code(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.csv"
        bad.write_text("a,b\n1,2\n")
        monkeypatch.setattr(sys, "argv", ["aggregate.py", "--input", str(bad), "--output", str(tmp_path / "o.csv")])
        assert aggregate.main() == 1

    def test_analyse_runs_end_to_end_on_synthetic_data(self, tmp_path, monkeypatch):
        """Build a synthetic aggregated table with a KNOWN structure and check the pipeline's outputs."""
        rows = []
        # CFR rises with age; males have double the odds-ish risk. Large n so every group passes --min-n.
        age_groups = ["0 - 9 Years", "20 - 29 Years", "50 - 59 Years", "80+ Years"]
        base = {"0 - 9 Years": 0.001, "20 - 29 Years": 0.005, "50 - 59 Years": 0.03, "80+ Years": 0.25}
        for month in ["2020-04", "2020-05", "2021-04", "2021-05"]:
            for sex, mult in [("Female", 1.0), ("Male", 1.5)]:
                for ag in age_groups:
                    for eth in ["White, Non-Hispanic", "Black, Non-Hispanic"]:
                        total = 20_000
                        deaths = int(total * base[ag] * mult)
                        rows.append((month, sex, ag, eth, "Yes", deaths))
                        rows.append((month, sex, ag, eth, "No", total - deaths))
                        rows.append((month, sex, ag, eth, "Missing/Unknown", 500))
        df = pd.DataFrame(rows, columns=["month", "sex", "age_group", "race_eth", "death", "n"])
        inp = tmp_path / "agg.csv"
        df.to_csv(inp, index=False)
        outdir = tmp_path / "results"

        analyse.LOG_LINES.clear()
        monkeypatch.setattr(sys, "argv", ["analyse.py", "--input", str(inp), "--outdir", str(outdir)])
        analyse.main()

        # all expected figures and tables exist
        for name in ["fig1_missingness.png", "fig1b_missingness_by_age.png", "fig2_age_sex.png",
                     "fig2_age_sex_linear.png", "fig3_ethnicity.png", "fig4_monthly_age.png",
                     "fig4_monthly_age_linear.png", "fig5_reporting_changes.png",
                     "fatality_by_age_sex.csv", "odds_ratio_sex_within_age.csv", "summary.txt"]:
            assert (outdir / name).is_file(), f"missing output: {name}"

        # numerical sanity: CFR of 80+ females must be close to the 25% we put in
        tab = pd.read_csv(outdir / "fatality_by_age_sex.csv")
        f80 = tab[(tab["age_group"] == "80+ Years") & (tab["sex"] == "Female")]["fatality"].iloc[0]
        assert f80 == pytest.approx(0.25, abs=0.01)
        # male-vs-female odds ratio must be > 1 and significant in every age group
        orr = pd.read_csv(outdir / "odds_ratio_sex_within_age.csv")
        assert (orr["odds_ratio_male_vs_female"] > 1).all()
        assert (orr["p_value"] < 0.05).all()
        # both ethnic groups have the same synthetic risk -> standardised CFRs must be equal
        eth = pd.read_csv(outdir / "fatality_by_ethnicity.csv")
        assert eth["std_fatality"].nunique() == 1 or np.ptp(eth["std_fatality"]) < 1e-9