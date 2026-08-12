# scripts/07c_data_diagnostics.py
import os
import pandas as pd
import numpy as np

PROC = os.path.join("data", "processed", "tamilnadu_nfhs5_encoded.csv")
RAW_CLEAN = os.path.join("data", "processed", "tamilnadu_nfhs5_cleaned.csv")

df = pd.read_csv(PROC)
print("=" * 70)
print("DATA DIAGNOSTICS")
print("=" * 70)

print("\n1) Dataset shape")
print(df.shape)

print("\n2) Household child counts (v001, v002)")
if {"v001", "v002"}.issubset(df.columns):
    hh_counts = df.groupby(["v001", "v002"]).size()
    print(hh_counts.value_counts().sort_index())
    print(f"Households with >1 child: {(hh_counts > 1).sum()} / {hh_counts.shape[0]}")
else:
    print("v001/v002 not present")

print("\n3) mother_bmi summary")
if "mother_bmi" in df.columns:
    print(df["mother_bmi"].describe())
    bad_bmi = df[(df["mother_bmi"] < 10) | (df["mother_bmi"] > 60)]
    print(f"Rows with implausible BMI (<10 or >60): {len(bad_bmi)}")
else:
    print("mother_bmi not present")

print("\n4) Binary feature prevalence")
for col in ["diarrhea_recent", "fever_recent", "electricity", "share_toilet", "child_sex"]:
    if col in df.columns:
        print(f"{col}:")
        print(df[col].value_counts(dropna=False).sort_index())

print("\n5) Current feature list")
print(sorted(df.columns.tolist()))

print("\n6) Features present in cleaned but maybe not encoded")
if os.path.exists(RAW_CLEAN):
    df_clean = pd.read_csv(RAW_CLEAN)
    missing_after_encoding = sorted(set(df_clean.columns) - set(df.columns))
    print(missing_after_encoding)
else:
    print("Cleaned file not found")