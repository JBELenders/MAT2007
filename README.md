# COVID-19 case fatality rate by age, sex, ethnicity and time (CDC data)

MAT2007 Introduction to Programming by project by Julian Lenders (I6404793)

## Research question
How does the COVID-19 case fatality rate vary between patients (accross age, sex, and ethnicity), and how did
these patterns change over time?

**Definition used throughout**

```
case fatality rate = deaths / cases with a known outcome
```

Cases whose death status is unknown are left out of both the numerator and the denominator.
This is *not* the infection fatality rate: the denominator only contains diagnosed cases that have a
recorded outcome (only about 32% of all cases do).

## Dataset
* **CDC COVID-19 Case Surveillance Public Use Data**, published via HealthData.gov:
  https://healthdata.gov/CDC/COVID-19-Case-Surveillance-Public-Use-Data/knt4-7efa/about_data
* About 14 GB as CSV, ~106,219,500 rows. The downloaded file is called `export.csv`.
* Only five out of the twelve columns are used: `cdc_case_earliest_dt`, `sex`, `age_group`, `race_ethnicity_combined`, `death_yn`.
* The raw file is **not included** in this repository (too large). To reproduce step 1, download it as CSV
  and save it as `data/export.csv`.
* All cleaning of the data is done automatically by `aggregate.py`:
  all spellings of "unknown" become `Missing/Unknown`, text is trimmed and unified, and dates outside
  2019-2026 are marked `Invalid`.
* **Shortcut:** `data/aggregated_counts.csv` (30,768 rows) is the output of step 1. With this, you can skip
  the 14 GB download and run only step 2.

## Repository layout
```
aggregate.py     step 1: data/export.csv -> data/aggregated_counts.csv   (reads in chunks, low memory)
analyse.py       step 2: data/aggregated_counts.csv -> results/          (figures, CSV tables, summary.txt)
report/          short LaTeX report
data/            export.csv (not committed) and aggregated_counts.csv
results/         generated figures and tables
```

## Installation
Requires Python 3.9 or newer. Developed locally in Visual Studio Code.

```bash
pip install pandas numpy matplotlib
```

## How to run
Run these commands in a terminal (in VS Code: Ctrl+`) from the project folder.

```bash
# Step 1 - only if you have the raw CSV (takes some time)
python aggregate.py

# Step 2 - takes seconds
python analyse.py
```

Both scripts have defaults that match the layout above. Use `--input` / `--output` / `--outdir`
if your files are somewhere else, for example:

```bash
python aggregate.py --input "C:\path\to\export.csv"
python analyse.py --input data/aggregated_counts.csv --outdir results
```

Useful options:
* `aggregate.py --max-chunks 4` -- quick smoke test on the first 4 chunks (20 million rows with the default
  chunk size of 5,000,000). It writes a separate `.partial` file and never overwrites the real result.
  Do not draw conclusions from it: the raw file is not randomly ordered.
* `aggregate.py --resume` -- continue after an interruption (progress is saved regularly).
* `aggregate.py --chunksize 1000000` -- use less memory per chunk if the computer runs out of RAM.
* `analyse.py --min-n 100` -- minimum number of known outcomes a group or month needs to be shown.

## Output (`results/`)
| File | Content |
|---|---|
| `fig1_missingness.png`, `fig1b_missingness_by_age.png` | share of cases with a known outcome, per year and per age group |
| `fig2_age_sex.png`, `fig2_age_sex_linear.png` | case fatality rate by age group and sex (log and linear scale) |
| `fig3_ethnicity.png` | crude versus age-standardised case fatality rate by ethnicity |
| `fig4_monthly_age.png`, `fig4_monthly_age_linear.png` | monthly case fatality rate for four age bands |
| `fig5_reporting_changes.png` | monthly case volume with documented reporting changes |
| `*.csv` | all numbers behind the figures, including odds ratios with confidence intervals and p-values |
| `summary.txt` | everything printed during the run |

## Methods
1. **Aggregation (step 1).** The raw file is read in chunks. Each case is assigned to a cell defined by
   month x sex x age group x ethnicity x death status, and the cases per cell are counted. The 106.2 million
   rows become about 30,000 rows.
2. **Case fatality rate per group.** deaths / known outcomes, for each age group and sex, each ethnic group
   and each month.
3. **Sex comparison.** Odds ratio of death, male versus female, *within each age group*, so that age
   differences do not distort it.
4. **Ethnicity comparison.** Direct age standardisation: each group's age-specific rates are weighted with
   the age mix of all known-outcome cases combined. This answers "what would this group's rate be if it had
   the same age structure as all cases combined?" and removes the effect of groups simply being older or younger.
5. **Change over time.** Monthly case fatality rate for four age bands (0-39, 40-59, 60-79, 80+).
   Months with fewer than 100 known outcomes are not shown.

## Why confidence intervals are used, and how
Every rate in this project is an *estimate* from a finite number of cases. If we saw 30 deaths among
1,000 cases we would estimate 3%, but with a different group of 1,000 similar cases we might have seen 25
or 36 deaths. A **95% confidence interval (CI)** gives the range of true values that is compatible with
what we observed: if the whole study were repeated many times, about 95% of the intervals built this way
would contain the true value.

This matters for three reasons:

* **Is a difference real or just noise?** If the intervals of two groups do not overlap (for example,
  males and females aged 70-79), the difference is very unlikely to be a coincidence. If they overlap
  strongly, we should not claim a difference.
* **How much can we trust a number?** The width of the interval shrinks with the number of cases
  (roughly with 1 / sqrt(n)). Small groups, such as young age groups with very few deaths, or months late
  in 2024 with few reports, get wide intervals and should be interpreted carefully. A single number
  without an interval hides this.
* **Quantification instead of qualitative statements.** The project asks for quantified results; an
  estimate with an interval is a quantified statement, and it is the standard way to report
  uncertainty in science.

How they are computed in `analyse.py`:

| Where | Method | How to read it |
|---|---|---|
| Rate per group or month (figures 2, 3, 4 and the CSV files) | **Wilson interval** for a proportion: deaths *k* out of *n* known outcomes | Error bars (figures 2 and 3) and shaded bands (figure 4) |
| Male vs female, and age group vs youngest group | **Odds ratio** with a Woolf interval, plus a two-sided p-value | If the interval does not contain 1 (equivalently p < 0.05), the difference is statistically significant |
| Age-standardised rate (figure 3) | Normal approximation with the standard error of the weighted sum of the age-specific rates | Red squares with error bars |

The **Wilson interval** is used instead of the textbook formula `p +- 1.96 * sqrt(p(1-p)/n)` because the
textbook formula can go below 0% and is unreliable when the rate is very close to 0, which is the case for
many young age groups. The Wilson interval always stays between 0% and 100%.

The **odds ratio** compares the odds of death (deaths / survivors) between two groups. An odds ratio of 2
means the odds of death in the first group are twice those of the second group. For rare outcomes the
odds are close to the rate, so it is roughly "twice the risk". If a cell of the 2x2 table is 0, 0.5 is
added to every cell so that the ratio can be calculated.

**Limits of confidence intervals.** They only describe *random* sampling uncertainty. They do not
account for systematic problems such as the 68% of cases without a recorded outcome, jurisdictions that
stopped reporting, or differences in testing over time. With millions of cases the intervals are very
narrow, so almost every difference is "statistically significant": the size of a difference (the odds
ratio or the percentage-point gap) is more informative than the p-value alone.

## Validation
The scripts are designed to check themselves while they run:
* `aggregate.py` verifies after every chunk that each row was counted exactly once, verifies at the end
  that the output counts add up to the number of rows read, and compares the rows read with the number of
  lines in the file (a warning is printed if rows were skipped as malformed).
* Both scripts stop with a clear error message if the input file or the required columns are missing.
* `analyse.py` leaves out groups and months with too few known outcomes (`--min-n`) and reports in
  `results/summary.txt` how many cases are excluded in each analysis.

## Limitations
* Only age is adjusted for. Comorbidities, access to care and calendar time are not.
* About 68% of cases have no recorded outcome. This may bias the rate in either direction, and the %
  changes over time (39% known in 2020, 24% in 2024).
* Some jurisdictions stopped reporting over time and the data set ends in 2024. The file has no
  geography, so these jurisdictions cannot be excluded.
* Testing capacity changed over time, and early in the pandemic mainly severe or clearly symptomatic cases were tested, which inflates the early case fatality rate and can exaggerate the decline over time.
* Personal and geographic details are removed from the public file.