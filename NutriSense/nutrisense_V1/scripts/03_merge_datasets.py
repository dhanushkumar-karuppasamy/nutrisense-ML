import os
import pandas as pd
import pyreadstat

TN_STATE_CODE = 33

def read_dta_chunked(filepath, usecols, filter_col, filter_val, chunksize=50000):
    chunks = []
    total_raw_rows = 0
    for df, _ in pyreadstat.read_file_in_chunks(
        pyreadstat.read_dta, filepath, chunksize=chunksize, usecols=usecols
    ):
        total_raw_rows += len(df)
        filtered = df[df[filter_col] == filter_val].copy()
        chunks.append(filtered)
    merged_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    return merged_df, total_raw_rows

def merge_datasets():
    print("Starting dataset merge (Week 3 — expanded feature set)...")
    raw_dir  = os.path.join("data", "raw")
    proc_dir = os.path.join("data", "processed")

    kr_path = os.path.join(raw_dir, "IAKR7EFL.DTA")
    ir_path = os.path.join(raw_dir, "IAIR7EFL.DTA")
    hr_path = os.path.join(raw_dir, "IAHR7EFL.DTA")

    for path in [kr_path, ir_path, hr_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found → {path}")
            return

    # ── KR columns ──────────────────────────────────────────────────────────
# b4   = sex of child (1=male, 2=female)
# hw1  = child age in months
# b11  = preceding birth interval (months since previous birth)
# bord = birth order number
# h11  = had diarrhea recently (0=no, 1=yes)
# h22  = had fever recently (0=no, 1=yes)
# hw70 = height-for-age z-score (stunting_haz) — target
# m14  = number of antenatal visits (CONFIRMED actual name — no _1 suffix in this DTA)
# m15  = place of delivery (CONFIRMED actual name)
# m17  = delivered by caesarean section (CONFIRMED actual name)
    kr_cols = [
        "caseid", "midx", "v001", "v002", "v024",
        "b4", "hw1", "b11", "bord", "h11", "h22", "hw70",
        "m14", "m15", "m17"
]
    # NOTE: m14_1/m15_1/m17_1 are birth-history variables in KR (index _1 = most recent birth).
    # If your DTA version names them differently, check with:
    #   pyreadstat.read_dta(kr_path, row_limit=1)[1].column_names_to_labels

    # ── IR columns ──────────────────────────────────────────────────────────
    # v012 = mother's current age (continuous)
    # v013 = mother's age in 5-yr groups (1–7 categorical)
    # v106 = mother's education level (0=none … 3=higher)
    # v190 = wealth index (1–5 ordinal)
    # v445 = mother BMI × 100 (divide by 100 later)
    # v457 = mother's anemia level (1=severe … 4=not anemic)
    ir_cols = [
        "caseid", "v024",
        "v012", "v013", "v106", "v190", "v445", "v457"
    ]

    # ── HR columns ──────────────────────────────────────────────────────────
    # hv201 = drinking water source
    # hv205 = toilet type
    # hv206 = electricity (0/1)
    # hv225 = share toilet with other households (0/1)
    hr_cols = [
        "hhid", "hv001", "hv002", "hv024",
        "hv201", "hv205", "hv206", "hv225"
    ]

    print("Loading KR (children)...")
    df_kr, kr_raw = read_dta_chunked(kr_path, kr_cols, "v024", TN_STATE_CODE)
    print("Loading IR (mothers)...")
    df_ir, ir_raw = read_dta_chunked(ir_path, ir_cols, "v024", TN_STATE_CODE)
    print("Loading HR (households)...")
    df_hr, hr_raw = read_dta_chunked(hr_path, hr_cols, "hv024", TN_STATE_CODE)

    print(f"Raw counts  - KR: {kr_raw}, IR: {ir_raw}, HR: {hr_raw}")
    print(f"TN filtered - KR: {df_kr.shape[0]}, IR: {df_ir.shape[0]}, HR: {df_hr.shape[0]}")

    if df_kr.empty or df_ir.empty or df_hr.empty:
        print("WARNING: One or more TN-filtered datasets are empty. Stopping.")
        return

    # -- Merge KR + IR on caseid ----------------------------------------------
    df_ir_merge = df_ir.drop(columns=["v024"])
    merged = pd.merge(df_kr, df_ir_merge, on="caseid", how="inner")
    print(f"After KR x IR merge: {merged.shape[0]} rows")

    # -- Merge with HR on cluster+household ----------------------------------
    df_hr = df_hr.rename(columns={"hv001": "v001", "hv002": "v002"}).drop(columns=["hv024"])
    df_final = pd.merge(merged, df_hr, on=["v001", "v002"], how="inner")
    print(f"After HR merge: {df_final.shape[0]} rows, {df_final.shape[1]} columns")

    # -- Rename to human-readable names --------------------------------------
    rename_map = {
    "v024":  "state",
    "b4":    "child_sex",
    "hw1":   "child_age_months",
    "b11":   "birth_interval",
    "bord":  "birth_order",
    "h11":   "diarrhea_recent",
    "h22":   "fever_recent",
    "m14":   "anc_visits",       # fixed: was m14_1
    "m15":   "delivery_place",   # fixed: was m15_1
    "m17":   "caesarean",        # fixed: was m17_1
    "hw70":  "stunting_haz",
    "v012":  "mother_age",
    "v013":  "mother_age_group",
    "v106":  "mother_education",
    "v190":  "wealth_index",
    "v445":  "mother_bmi_raw",
    "v457":  "mother_anemia",
    "hv201": "water_source",
    "hv205": "toilet_type",
    "hv206": "electricity",
    "hv225": "share_toilet",
}
    df_final = df_final.rename(columns=rename_map)

    os.makedirs(proc_dir, exist_ok=True)
    out = os.path.join(proc_dir, "tamilnadu_nfhs5_merged.csv")
    df_final.to_csv(out, index=False)
    print(f"Saved -> {out}  ({df_final.shape[0]} rows x {df_final.shape[1]} cols)")
    print(f"Columns: {list(df_final.columns)}")
    
def validate_columns_exist(filepath, requested_cols, label):
    """Fail loudly if any requested column doesn't actually exist in the DTA file."""
    _, meta = pyreadstat.read_dta(filepath, row_limit=1)
    actual_cols = set(meta.column_names)
    missing = [c for c in requested_cols if c not in actual_cols]
    if missing:
        raise ValueError(
            f"[{label}] These requested columns do NOT exist in {filepath}: {missing}\n"
            f"pyreadstat silently drops unknown usecols instead of erroring — "
            f"this check exists to catch that. Inspect meta.column_names to find correct names."
        )
    print(f"  [{label}] All {len(requested_cols)} requested columns verified present.")

if __name__ == "__main__":
    merge_datasets()