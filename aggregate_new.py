#!/usr/bin/env python3
"""
aggregate.py -- Step 1 of 2: reduce the huge CDC case-surveillance CSV to a small table of counts.

WHY THIS STEP EXISTS
    The raw file (~14 GB, ~106 million rows) does not fit in memory. Everything we need for the
    analysis is a COUNT of cases per combination of
        month x sex x age_group x race_ethnicity x death_yn
    so we read the file in chunks, clean + count each chunk, and add the counts together.
    The result is ~30,000 rows, which analyse.py can load instantly.

USAGE
    python aggregate.py --input data/export.csv
    python aggregate.py --input data/raw.csv --max-chunks 2     # quick smoke test (writes a separate .partial file)
    python aggregate.py --input data/raw.csv --resume           # continue after an interruption

Note
    The raw file is note randomly ordered (it starts with one demographic group), so never judge
    results from only the first chunks.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- configuration
# Mapping: column name in the raw CDC file -> short internal name used in this project.
WANTED = {
    "cdc_case_earliest_dt": "date",
    "sex": "sex",
    "age_group": "age_group",
    "race_ethnicity_combined": "race_eth",
    "death_yn": "death",
}
# The five columns that together define one "cell" of the aggregated table.
KEYS = ["month", "sex", "age_group", "race_eth", "death"]

# Every spelling of "we do not know" is mapped to this single label.
MISSING_LABEL = "Missing/Unknown"
MISSING_TOKENS = {"", "nan", "na", "n/a", "none", "null", "missing", "unknown",
                  "not available", "not reported"}

# Dates outside this window are treated as typos in the raw data and marked "Invalid".
YEAR_MIN, YEAR_MAX = 2019, 2026

COMPRESSED = {".gz", ".zip", ".bz2", ".xz", ".zst"}  # line counting is skipped for these
CHECKPOINT_EVERY = 4  # save progress every N chunks so a crash does not lose a lot of time


# --------------------------------------------------------------------------- cleaning
# These functions only ever run on SMALL tables (see count_chunk), so they can be slow-but-clear.
def clean_category(s: pd.Series) -> pd.Series:
    """Tidy a text column: strip whitespace, unify capitalisation, map all 'unknown' spellings to one label."""
    # 'string' dtype keeps real missing values (NA) instead of the text "nan".
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    is_missing = s.isna() | s.str.lower().isin(MISSING_TOKENS)
    # .title() makes 'FEMALE' / 'female' / 'Female' identical; missing cells get the common label.
    s = s.str.title().mask(is_missing, MISSING_LABEL)
    return s.astype(object)


def clean_death(s: pd.Series) -> pd.Series:
    """death_yn becomes exactly one of: Yes / No / Missing/Unknown."""
    s = clean_category(s)
    return s.where(s.isin(["Yes", "No"]), MISSING_LABEL)


def to_month(s: pd.Series) -> pd.Series:
    """Convert dates such as '2022/10/13' (or '2022-10-13') to '2022-10'; anything unparsable -> 'Invalid'."""
    # Regex pulls out year and month only; the day is not needed.
    ext = s.astype("string").str.strip().str.extract(r"^(\d{4})[/-](\d{1,2})")
    year = pd.to_numeric(ext[0], errors="coerce")
    mon = pd.to_numeric(ext[1], errors="coerce")
    # A date is valid only if year and month are inside the allowed ranges.
    ok = (year.between(YEAR_MIN, YEAR_MAX) & mon.between(1, 12)).fillna(False).to_numpy(bool)
    out = pd.Series("Invalid", index=s.index, dtype=object)
    good = year[ok].astype(int).astype(str) + "-" + mon[ok].astype(int).astype(str).str.zfill(2)
    out.loc[ok] = good.to_numpy()
    return out


def count_chunk(chunk: pd.DataFrame) -> pd.Series:
    """
    Count one chunk per cleaned (month, sex, age_group, race_eth, death) combination.

    SPEED TRICK: identical RAW rows are counted first (cheap, because the columns are categoricals,
    i.e. integer codes). The slow text cleaning then runs only on the few thousand UNIQUE
    combinations instead of on every one of the million rows in the chunk.
    dropna=False keeps rows with empty cells; they are cleaned to 'Missing/Unknown' afterwards.
    """
    raw_cols = list(WANTED)
    # Step A: count identical raw rows. Result has one row per unique raw combination plus a count 'n'.
    g = (chunk.groupby(raw_cols, observed=True, dropna=False, sort=False)
              .size().rename("n").reset_index()
              .rename(columns=WANTED))
    for c in WANTED.values():
        g[c] = g[c].astype(object)  # categoricals -> plain objects so the string cleaners work
    # Step B: clean the (small) table of unique combinations.
    cleaned = pd.DataFrame({
        "month": to_month(g["date"]),
        "sex": clean_category(g["sex"]),
        "age_group": clean_category(g["age_group"]),
        "race_eth": clean_category(g["race_eth"]),
        "death": clean_death(g["death"]),
        "n": g["n"].to_numpy(),
    })
    # Step C: different raw spellings may now be the same cleaned combination, so add their counts.
    return cleaned.groupby(KEYS, sort=False)["n"].sum()


# --------------------------------------------------------------------------- helpers
def count_lines(path: Path, block: int = 1 << 26) -> int:
    """Number of lines in a plain file, counted in 64 MB blocks (fast, constant memory)."""
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            n += b.count(b"\n")
            last = b[-1:]
    return n + (0 if last == b"\n" else 1)  # a final line without newline still counts


def fmt_time(seconds: float) -> str:
    """Seconds -> '1h05m' or '12m30s' for the progress messages."""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def save_checkpoint(total, chunks_done, rows_read, meta, ck_csv, ck_json):
    """
    Save progress so --resume can continue later. Best effort: a failed checkpoint must never stop the run.
    Files are written to *.tmp first and then renamed, so a crash cannot leave a half-written checkpoint.
    """
    try:
        tmp_csv = ck_csv.with_name(ck_csv.name + ".tmp")
        tmp_json = ck_json.with_name(ck_json.name + ".tmp")
        total.rename("n").astype("int64").reset_index().to_csv(tmp_csv, index=False)
        tmp_json.write_text(json.dumps({"meta": meta, "chunks_done": chunks_done, "rows_read": rows_read}))
        tmp_csv.replace(ck_csv)
        tmp_json.replace(ck_json)
    except OSError as exc:
        print(f"WARNING: could not write checkpoint ({exc}); the run continues without it.")


def load_checkpoint(meta, ck_csv, ck_json):
    """Return (total, chunks_done, rows_read), or None when there is no usable checkpoint."""
    if not (ck_csv.is_file() and ck_json.is_file()):
        print("No checkpoint found - starting from the beginning.")
        return None
    try:
        info = json.loads(ck_json.read_text())
        # A checkpoint is only valid for the same input file and chunk size.
        if info["meta"] != meta:
            print("Checkpoint is from a different input file or chunk size - starting from the beginning.")
            return None
        df = pd.read_csv(ck_csv, dtype={**{k: str for k in KEYS}, "n": "int64"}, keep_default_na=False)
        total = df.set_index(KEYS)["n"]
        # Integrity check: counts in the checkpoint must add up to the rows it claims to have read.
        if int(total.sum()) != int(info["rows_read"]):
            print("Checkpoint is inconsistent (interrupted while saving) - starting from the beginning.")
            return None
        return total, int(info["chunks_done"]), int(info["rows_read"])
    except Exception as exc:
        print(f"Could not use the checkpoint ({exc}) - starting from the beginning.")
        return None


# --------------------------------------------------------------------------- main
def main() -> int:
    """Run the aggregation. Returns a process exit code (0 = success)."""
    # ---- command-line arguments
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/export.csv", help="path to the raw CDC CSV (.csv, .csv.gz and .zip also work)")
    ap.add_argument("--output", default="data/aggregated_counts.csv")
    ap.add_argument("--chunksize", type=int, default=5_000_000, help="rows per chunk (lower it if memory is tight)")
    ap.add_argument("--max-chunks", type=int, default=0, help="stop after N chunks (0 = read everything; testing only)")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint of an interrupted run")
    ap.add_argument("--skip-line-check", action="store_true", help="do not count the file's lines first (saves a minute or two)")
    args = ap.parse_args()

    # ---- defensive checks on the inputs (fail fast with a clear message)
    src = Path(args.input)
    if not src.is_file():
        print(f"ERROR: input file not found: {src}\n       Use --input to give the path to the raw CSV.", file=sys.stderr)
        return 1
    if args.chunksize < 1:
        print("ERROR: --chunksize must be a positive number", file=sys.stderr)
        return 1
    testing = args.max_chunks > 0
    out_path = Path(args.output)
    if testing:  # a smoke test must never overwrite the real result
        out_path = out_path.with_name(out_path.stem + ".partial" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ck_csv = out_path.with_name(out_path.stem + ".checkpoint.csv")
    ck_json = out_path.with_name(out_path.stem + ".checkpoint.json")

    # ---- find the wanted columns in the header
    # Read the header with the SAME options as the main read, and match columns by their stripped
    # names (the raw header contains stray spaces, e.g. 'cdc_case_earliest_dt ').
    read_opts = dict(on_bad_lines="skip", encoding="utf-8-sig", encoding_errors="replace", skipinitialspace=True)
    try:
        header = pd.read_csv(src, nrows=0, dtype=str, **read_opts)
    except Exception as exc:
        print(f"ERROR: could not read the header of {src}: {exc}", file=sys.stderr)
        return 1
    actual = {c: str(c).strip() for c in header.columns}
    usecols = [c for c, stripped in actual.items() if stripped in WANTED]
    absent = sorted(set(WANTED) - {actual[c] for c in usecols})
    if absent:
        print(f"ERROR: column(s) not found in the file: {absent}\n"
              f"       columns in the file: {list(actual.values())}", file=sys.stderr)
        return 1
    if len(usecols) != len(WANTED):
        print(f"ERROR: some wanted columns appear twice in the header: {list(actual.values())}", file=sys.stderr)
        return 1
    # Categoricals: the reader de-duplicates the text for us (fast, low memory).
    dtype = {c: "category" for c in usecols}

    # ---- optional integrity check: compare rows parsed with lines in the file,
    # so silently skipped (malformed) rows show up at the end.
    expected_rows = None
    if not testing and not args.skip_line_check and src.suffix.lower() not in COMPRESSED:
        print("Counting the lines in the file first (one fast pass)...", flush=True)
        expected_rows = max(count_lines(src) - 1, 0)  # minus the header line
        print(f"  {expected_rows:,} data lines found", flush=True)

    # ---- optionally resume from a checkpoint
    meta = {"input": str(src.resolve()), "size": src.stat().st_size, "chunksize": args.chunksize}
    total, skip, rows_read = None, 0, 0
    if args.resume and not testing:
        loaded = load_checkpoint(meta, ck_csv, ck_json)
        if loaded:
            total, skip, rows_read = loaded
            print(f"Resuming after chunk {skip} ({rows_read:,} rows already counted). "
                  f"The file is still read from the start to get there, but not counted again.", flush=True)

    # ---- main loop: read chunk -> count -> add to running total
    start = time.time()
    t0, rows0 = start, rows_read  # t0/rows0 are used for the "time left" estimate
    chunks_done = skip
    status = "ok"
    try:
        with pd.read_csv(src, usecols=usecols, dtype=dtype, chunksize=args.chunksize, **read_opts) as reader:
            for i, chunk in enumerate(reader, start=1):
                if i <= skip:  # already counted before the interruption
                    if i == skip:
                        t0, rows0 = time.time(), rows_read
                    continue
                chunk.columns = [str(c).strip() for c in chunk.columns]
                counts = count_chunk(chunk)
                # Integrity check: every row must be counted exactly once.
                if int(counts.sum()) != len(chunk):
                    raise RuntimeError(f"row count mismatch in chunk {i} ({int(counts.sum()):,} counted vs "
                                       f"{len(chunk):,} read; pandas {pd.__version__}). Please report this.")
                # fill_value=0 so combinations that appear in only one of the two tables are kept.
                new_total = counts if total is None else total.add(counts, fill_value=0)
                total, rows_read, chunks_done = new_total, rows_read + len(chunk), i
                rate = (rows_read - rows0) / max(time.time() - t0, 1e-9)
                extra = ""
                if expected_rows:
                    extra = f"  {100 * rows_read / expected_rows:5.1f}%  ~{fmt_time((expected_rows - rows_read) / rate)} left"
                print(f"chunk {i:>5}: {rows_read:>13,} rows{extra}  ({fmt_time(time.time() - start)} elapsed)", flush=True)
                if testing and chunks_done >= args.max_chunks:
                    print("NOTE: stopped early because of --max-chunks; the output is a partial test file.")
                    break
                if not testing and chunks_done % CHECKPOINT_EVERY == 0:
                    save_checkpoint(total, chunks_done, rows_read, meta, ck_csv, ck_json)
    except KeyboardInterrupt:
        status = "Interrupted (Ctrl+C)"
    except Exception as exc:
        status = f"Stopped by an error: {type(exc).__name__}: {exc}"

    # ---- if the loop did not finish normally, save progress and tell the user how to continue
    if status != "ok":
        if total is not None and chunks_done > 0 and not testing:
            save_checkpoint(total, chunks_done, rows_read, meta, ck_csv, ck_json)
            print(f"\n{status}.\nProgress up to chunk {chunks_done} ({rows_read:,} rows) is saved. "
                  f"Run the same command again with --resume to continue.", file=sys.stderr)
        else:
            print(f"\n{status}.", file=sys.stderr)
        return 130 if status.startswith("Interrupted") else 1

    if total is None or rows_read == 0:
        print("ERROR: no data rows were read (empty file or every row was malformed).", file=sys.stderr)
        return 1

    # ---- write the result
    result = total.rename("n").astype("int64").reset_index()
    if int(result["n"].sum()) != rows_read:  # final internal consistency check
        print("ERROR: counts do not add up to the number of rows read - please report this bug.", file=sys.stderr)
        return 1

    tmp = out_path.with_name(out_path.name + ".tmp")
    result.to_csv(tmp, index=False)
    try:
        tmp.replace(out_path)  # a crash can never leave a half-written output file
    except OSError as exc:  # on Windows: the old file may be open in Excel / a VS Code preview
        print(f"ERROR: could not write {out_path} ({exc}).\n"
              f"       The finished result is safe in {tmp}. Close the program that has {out_path.name} open "
              f"and rename the .tmp file yourself.", file=sys.stderr)
        return 1

    # ---- write a small JSON log (useful for the report: row counts, runtime, pandas version)
    n_invalid = int(result.loc[result["month"] == "Invalid", "n"].sum())
    n_known = int(result.loc[result["death"].isin(["Yes", "No"]), "n"].sum())
    skipped = None if expected_rows is None else expected_rows - rows_read
    log = {"input": str(src), "finished": datetime.now().isoformat(timespec="seconds"),
           "partial_test_run": testing, "rows_read": rows_read, "lines_in_file": expected_rows,
           "rows_skipped_as_malformed": skipped, "chunks": chunks_done, "chunksize": args.chunksize,
           "output_rows": len(result), "invalid_date_rows": n_invalid, "known_death_rows": n_known,
           "pandas_version": pd.__version__, "seconds": round(time.time() - start, 1)}
    out_path.with_suffix(".log.json").write_text(json.dumps(log, indent=2))
    for p in (ck_csv, ck_json):  # the run finished, so the checkpoint is no longer needed
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    # ---- final summary on screen
    print(f"\nDone: {rows_read:,} rows -> {len(result):,} aggregated rows in {out_path}")
    print(f"Known death status:        {n_known:,} ({100 * n_known / rows_read:.1f}%)")
    print(f"Invalid/missing date:      {n_invalid:,} ({100 * n_invalid / rows_read:.2f}%)")
    if skipped is not None:
        if abs(skipped) > max(10, 0.001 * expected_rows):
            print(f"WARNING: {skipped:,} lines of the file were NOT parsed as rows ({100 * skipped / expected_rows:.2f}%). "
                  f"Check the file before trusting the results.")
        else:
            print(f"Lines not parsed as rows:  {skipped:,} (negligible)")
    return 0


if __name__ == "__main__":
    sys.exit(main())