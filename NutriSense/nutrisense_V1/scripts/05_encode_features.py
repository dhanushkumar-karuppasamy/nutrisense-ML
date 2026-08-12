# scripts/05_encode_features.py
"""
Updated encoding pipeline:
  1. Collapse rare one-hot categories (< frequency threshold) into 'other'
     — eliminates the 9 sparse columns diagnosis identified
  2. Ordinal encoding for ranked variables
  3. One-hot encoding with rare-category collapsing
  4. Two interaction features (new ideas — Erda et al. did not have these)
  5. Binary recoding
  6. Stunting label from HAZ
"""
import os
import pandas as pd
import numpy as np

def encode_features():
    proc_dir = os.path.join("data", "processed")
    in_path  = os.path.join(proc_dir, "tamilnadu_nfhs5_cleaned.csv")
    out_path = os.path.join(proc_dir, "tamilnadu_nfhs5_encoded.csv")

    df = pd.read_csv(in_path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Columns: {df.columns.tolist()}")

    # ── 1. Ordinal encoding ──────────────────────────────────────────────────
    ordinal_cols = {
        "mother_education": {0: 0, 1: 1, 2: 2, 3: 3},
        "wealth_index":     {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        "mother_anemia":    {1: 3, 2: 2, 3: 1, 4: 0},
        "mother_age_group": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
    }
    for col, mapping in ordinal_cols.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
            print(f"  Ordinal kept: {col}")

    # ── 2. Binary recoding ────────────────────────────────────────────────────
    if "child_sex" in df.columns:
        df["child_sex"] = df["child_sex"].map({1: 0, 2: 1})
        print("  Recoded child_sex: 1=male->0, 2=female->1")

    # ── 3. Nominal: collapse rare categories, THEN one-hot encode ─────────────
    # WHY: One-hot columns with <1% of rows non-zero are pure noise for
    # tree splits — the algorithm tries them but never finds a useful threshold.
    # Collapsing rare categories before OHE directly fixes the 9 sparse columns
    # identified in the diagnosis.
    nominal_cols = [c for c in ["toilet_type", "water_source", "delivery_place"]
                    if c in df.columns]

    RARE_THRESHOLD = 0.01  # categories with < 1% frequency → merge into "other"

    for col in nominal_cols:
        freq = df[col].value_counts(normalize=True)
        rare_cats = freq[freq < RARE_THRESHOLD].index.tolist()
        if rare_cats:
            print(f"  Collapsing {len(rare_cats)} rare categories in '{col}' -> 'other': {rare_cats}")
            df[col] = df[col].apply(lambda x: "other" if x in rare_cats else x)
        else:
            print(f"  No rare categories in '{col}'")

    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)
        print(f"  One-hot encoded: {nominal_cols}")

    # ── 4. Interaction features ───────────────────────────────────────────────
    # WHY: These are engineered from domain knowledge, not just raw DHS variables.
    # Captures compounding risk pathways that Erda et al. did not model.
    added_interactions = []

    # Interaction 1: wealth × maternal education
    # Domain rationale: low wealth + low education compounds stunting risk
    # more than either factor alone (social determinants synergy)
    wealth_col = next((c for c in df.columns
                       if c in ["wealth_index", "v190", "wealth"]), None)
    edu_col    = next((c for c in df.columns
                       if c in ["mother_education", "v106", "m_educ"]), None)

    if wealth_col and edu_col:
        df["interact_wealth_x_education"] = df[wealth_col] * df[edu_col]
        added_interactions.append("interact_wealth_x_education")
        print(f"  [OK] Interaction created: wealth_x_education "
              f"({wealth_col} * {edu_col})")
    else:
        print(f"  [WARN] Skipped wealth x education "
              f"(wealth_col={wealth_col}, edu_col={edu_col})")
        print(f"    Available columns: {[c for c in df.columns if 'wealth' in c.lower() or 'educ' in c.lower()]}")

    # Interaction 2: unsafe water × diarrhea
    # Domain rationale: WASH-infection pathway — unsafe water AND active
    # diarrhea together indicate an ongoing infection-undernutrition cycle
    # that's more predictive than either feature alone
    water_cols = [c for c in df.columns if c.lower().startswith("water_source")]
    diarrhea_col = next((c for c in df.columns
                         if any(k in c.lower() for k in
                                ["diarrhea", "diarrhoea", "h11"])), None)

    if water_cols and diarrhea_col:
        # Identify which OHE water columns represent unsafe sources
        # DHS unsafe codes: 32 (unprotected well), 40-72 (surface/other)
        unsafe_cols = [c for c in water_cols
                       if any(str(code) in c for code in
                              ["31", "32", "40", "41", "42", "43", "51",
                               "61", "71", "72", "unprotect", "surface", "other"])]
        if unsafe_cols:
            df["unsafe_water_flag"] = df[unsafe_cols].sum(axis=1).clip(0, 1)
        elif "water_source" in df.columns:
            df["unsafe_water_flag"] = df["water_source"].between(30, 75).astype(int)
        else:
            df["unsafe_water_flag"] = 0

        df["interact_unsafe_water_x_diarrhea"] = (
            df["unsafe_water_flag"] * df[diarrhea_col]
        )
        added_interactions.append("interact_unsafe_water_x_diarrhea")
        print(f"  [OK] Interaction created: unsafe_water_x_diarrhea "
              f"(unsafe_flag * {diarrhea_col})")
        df.drop(columns=["unsafe_water_flag"], inplace=True, errors="ignore")
    else:
        print(f"  [WARN] Skipped water x diarrhea "
              f"(water_cols={len(water_cols)}, diarrhea_col={diarrhea_col})")
        print(f"    Available columns: {[c for c in df.columns if 'water' in c.lower() or 'diarr' in c.lower()]}")

    # ── 5. Stunting label from HAZ ────────────────────────────────────────────
    if "stunting_haz" in df.columns:
        df["stunting_label"] = (df["stunting_haz"] < -2.0).astype(int)
        prev = df["stunting_label"].mean() * 100
        print(f"\nStunting prevalence: {prev:.1f}%  "
              f"(stunted: {df['stunting_label'].sum()}, "
              f"not stunted: {(df['stunting_label']==0).sum()})")

    # ── 6. Post-encoding validation ───────────────────────────────────────────
    print(f"\nFinal shape: {df.shape[0]} rows × {df.shape[1]} cols")

    # Check remaining sparsity
    non_target = [c for c in df.columns if c not in ["stunting_label", "stunting_haz"]]
    numeric_cols = df[non_target].select_dtypes(include=[np.number]).columns
    col_sums   = df[numeric_cols].sum().sort_values()
    still_sparse = col_sums[col_sums < 20]
    if len(still_sparse) > 0:
        print(f"\n[WARN] Still-sparse columns after fix ({len(still_sparse)}):")
        print(still_sparse.to_string())
    else:
        print("\n[OK] No sparse columns remaining (< 20 non-zero entries).")

    print(f"\nInteraction features added: {added_interactions}")
    df.to_csv(out_path, index=False)
    print(f"\nSaved encoded dataset -> {out_path}")

if __name__ == "__main__":
    encode_features()