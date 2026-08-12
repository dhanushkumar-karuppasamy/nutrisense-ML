import os
import pandas as pd

def build_feature_dictionary():
    print("Building feature dictionary...")
    interim_dir = os.path.join("data", "interim")
    dict_path = os.path.join(interim_dir, "feature_dictionary.csv")
    
    # Custom structure with variable_code, feature_name, and source_file matching processed column names
    features = [
        {"variable_code": "v024", "feature_name": "state", "source_file": "KR"},
        {"variable_code": "v106", "feature_name": "mother_education", "source_file": "IR"},
        {"variable_code": "v190", "feature_name": "wealth_index", "source_file": "IR"},
        {"variable_code": "v445", "feature_name": "mother_bmi", "source_file": "IR"},
        {"variable_code": "hv206", "feature_name": "electricity", "source_file": "HR"},
        {"variable_code": "hv205", "feature_name": "toilet_type", "source_file": "HR"},
        {"variable_code": "hv225", "feature_name": "share_toilet", "source_file": "HR"},
        {"variable_code": "hw70", "feature_name": "stunting_haz", "source_file": "KR"}
    ]
    
    df = pd.DataFrame(features)
    os.makedirs(os.path.dirname(dict_path), exist_ok=True)
    df.to_csv(dict_path, index=False)
    print(f"Feature dictionary created at {dict_path}")

if __name__ == "__main__":
    build_feature_dictionary()
