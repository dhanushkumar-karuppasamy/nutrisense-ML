import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer

# ── DHS sentinel codes to replace with NaN ──────────────────────────────────
# These are standard DHS codes for "missing / don't know / not applicable"
SENTINEL_RANGES = {
    "anc_visits":       (90, 99),
    "birth_interval":   (990, 999),
    "mother_bmi_raw":   (9995, 9999),
    "stunting_haz":     (9990, 9999),
    "child_age_months": (97, 99),
    "mother_age":       (97, 99),
    "delivery_place":   (96, 99),
    "mother_anemia":    (8, 9),
    "caesarean":        (8, 9),   # NEW — 8=don't know, 9=missing
}

# ── Variables that CANNOT be imputed (must be present to keep the row) ───────
REQUIRED_COLS = ["stunting_haz", "child_age_months", "caseid"]

def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known DHS sentinel codes with NaN, then apply HAZ/BMI scaling."""
    df = df.copy()
    for col, (lo, hi) in SENTINEL_RANGES.items():
        if col in df.columns:
            mask = (df[col] >= lo) & (df[col] <= hi)
            replaced = mask.sum()
            df.loc[mask, col] = np.nan
            if replaced:
                print(f"  Replaced {replaced} sentinel values in '{col}'")

    # Scale stored integer codes to real values
    if "mother_bmi_raw" in df.columns:
        df["mother_bmi"] = df["mother_bmi_raw"] / 100
        df.drop(columns=["mother_bmi_raw"], inplace=True)

    if "stunting_haz" in df.columns:
        df["stunting_haz"] = df["stunting_haz"] / 100   # DHS stores HAZ × 100

    return df

def compute_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missing percentage table — document this in decisions.md."""
    report = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_pct":   df.isnull().mean() * 100
    }).sort_values("missing_pct", ascending=False)
    return report[report["missing_count"] > 0]

def apply_imputation(df: pd.DataFrame, miss_report: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation strategy (document each choice in decisions.md):
    - stunting_haz, child_age_months: drop row if missing (these are non-negotiable)
    - Continuous (BMI, ANC visits, birth_interval, mother_age): 
        > median if missingness ≤ 15%, MICE if > 15%
    - Ordinal/categorical (education, wealth, toilet, water_source, anemia, delivery_place):
        > mode imputation OR add 'missing' category if missingness > 10% 
          (missing-as-category can be predictive, e.g., no toilet = extreme poverty)
    - Binary (electricity, share_toilet, caesarean, sex):
        > mode imputation (only 2 values, mode is safe)
    """
    df = df.copy()

    # 1. Drop rows where target or key identifiers are missing
    before = len(df)
    df.dropna(subset=REQUIRED_COLS, inplace=True)
    print(f"Dropped {before - len(df)} rows with missing stunting_haz or child_age_months")

    # 2. Define variable groups
    continuous_cols = ["mother_bmi", "anc_visits", "birth_interval", "mother_age"]
    ordinal_cols    = ["mother_education", "wealth_index", "mother_age_group",
                       "mother_anemia", "delivery_place"]
    nominal_cols    = ["toilet_type", "water_source"]
    binary_cols = ["electricity", "share_toilet", "caesarean",
               "child_sex", "diarrhea_recent", "fever_recent"]

    # 3. Continuous: median or MICE based on missingness threshold
    high_miss_cont = []
    for col in continuous_cols:
        if col not in df.columns:
            continue
        pct = df[col].isnull().mean() * 100
        if pct == 0:
            continue
        if pct <= 15:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  [{col}] missingness={pct:.1f}% -> median imputation ({median_val:.2f})")
        else:
            high_miss_cont.append(col)
            print(f"  [{col}] missingness={pct:.1f}% -> flagged for MICE")

    if high_miss_cont:
        print(f"  Running MICE on: {high_miss_cont}")
        mice = IterativeImputer(max_iter=10, random_state=42)
        df[high_miss_cont] = mice.fit_transform(df[high_miss_cont])

    # 4. Ordinal: mode imputation (preserve ordering — do NOT one-hot)
    for col in ordinal_cols:
        if col not in df.columns:
            continue
        pct = df[col].isnull().mean() * 100
        if pct == 0:
            continue
        if pct > 10:
            # Add explicit 'missing' category — missingness may correlate with poverty
            df[col] = df[col].fillna(-1).astype(int)
            print(f"  [{col}] missingness={pct:.1f}% -> 'missing' category (-1)")
        else:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  [{col}] missingness={pct:.1f}% -> mode imputation ({mode_val})")

    # 5. Nominal: mode imputation
    for col in nominal_cols:
        if col not in df.columns:
            continue
        pct = df[col].isnull().mean() * 100
        if pct == 0:
            continue
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  [{col}] missingness={pct:.1f}% -> mode imputation ({mode_val})")

    # 6. Binary: mode imputation
    for col in binary_cols:
        if col not in df.columns:
            continue
        pct = df[col].isnull().mean() * 100
        if pct == 0:
            continue
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  [{col}] missingness={pct:.1f}% -> mode imputation ({mode_val})")

    return df

def run_quality_checks():
    proc_dir = os.path.join("data", "processed")
    in_path  = os.path.join(proc_dir, "tamilnadu_nfhs5_merged.csv")
    out_path = os.path.join(proc_dir, "tamilnadu_nfhs5_cleaned.csv")
    report_path = os.path.join("outputs", "missingness_report.csv")

    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run 03_merge_datasets.py first.")
        return

    df = pd.read_csv(in_path)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    print("\n-- Step 1: Replace DHS sentinel codes --")
    df = replace_sentinels(df)

    print("\n-- Step 2: Missingness report (before imputation) --")
    miss_report = compute_missingness_report(df)
    print(miss_report.to_string())
    os.makedirs("outputs", exist_ok=True)
    miss_report.to_csv(report_path)
    print(f"Saved missingness report -> {report_path}")

    print("\n-- Step 3: Apply imputation --")
    df = apply_imputation(df, miss_report)

    # Final sanity check
    remaining = df.isnull().sum().sum()
    print(f"\nRemaining NaNs after imputation: {remaining}")

    df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset -> {out_path}  ({df.shape[0]} rows x {df.shape[1]} cols)")

if __name__ == "__main__":
    run_quality_checks()