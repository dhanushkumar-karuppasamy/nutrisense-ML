# NFHS-5 Tamil Nadu Child Stunting Analysis

This project analyzes factors associated with child stunting in Tamil Nadu using the National Family Health Survey (NFHS-5) data. Stunting (height-for-age z-score < -2.0 standard deviations) indicates chronic malnutrition and has significant long-term effects on child development.

---

## Directory Structure

```
├── .gitignore                   # Excludes virtual envs, pycache, and huge raw datasets
├── requirements.txt             # Project Python dependencies
├── README.md                    # Project documentation
│
├── data/
│   ├── raw/                     # Raw NFHS-5 datasets (Excluded from git due to size)
│   │   ├── IAKR7EFL.DTA         # Kids Recode (KR)
│   │   ├── IAHR7EFL.DTA         # Household Recode (HR)
│   │   └── IAIR7EFL.DTA         # Individual Recode (IR)
│   ├── interim/                 # Extracted metadata and feature mapping
│   │   ├── kr_metadata.csv
│   │   ├── hr_metadata.csv
│   │   ├── ir_metadata.csv
│   │   └── feature_dictionary.csv
│   └── processed/               # Cleaned and merged dataset
│       └── tamilnadu_nfhs5_merged.csv
│
├── notebooks/                   # Jupyter Notebooks for analysis
│   ├── 01_metadata_inspection.ipynb
│   ├── 02_feature_mapping.ipynb
│   ├── 03_merge_and_qc.ipynb
│   └── 04_eda_plots.ipynb
│
├── scripts/                     # Python scripts running the data pipeline
│   ├── 01_extract_metadata.py   # Step 1: Extract variables and labels
│   ├── 02_build_feature_dictionary.py  # Step 2: Build codebook of target variables
│   ├── 03_merge_datasets.py      # Step 3: Filter for Tamil Nadu, load relevant features, and merge
│   └── 04_quality_checks.py     # Step 4: Run quality checks and plot distributions
│
└── outputs/
    ├── tables/                  # Summary output tables
    └── figures/                 # Diagnostic/EDA plots
        └── stunting_distribution.png
```

---

## Installation & Setup

### 1. Create and Activate Virtual Environment
To keep dependencies isolated, it is recommended to use the Python virtual environment:

* **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 2. Select Interpreter in IDE
Ensure your editor (e.g., VS Code) is configured to use the workspace interpreter:
1. Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`).
2. Search for **`Python: Select Interpreter`**.
3. Choose the virtual environment python interpreter (`./venv/Scripts/python.exe`).

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Data Pipeline Execution

Run the scripts in sequential order to process the raw datasets:

> [!IMPORTANT]
> Before running the merge script, the state code for Tamil Nadu must be confirmed using the metadata inspection notebook (`notebooks/01_metadata_inspection.ipynb`) or by searching the extracted metadata CSVs (e.g., `data/interim/ir_metadata.csv`). If the state code differs from the default value of `33`, update the `TN_STATE_CODE` constant in `scripts/03_merge_datasets.py`.

```bash
# 1. Extract metadata and column descriptions
python scripts/01_extract_metadata.py

# 2. Build mapping dictionary for target features
python scripts/02_build_feature_dictionary.py

# 3. Merge Kids, Individual, and Household datasets for Tamil Nadu
python scripts/03_merge_datasets.py

# 4. Perform quality checks and generate stunting visualizations
python scripts/04_quality_checks.py
```
