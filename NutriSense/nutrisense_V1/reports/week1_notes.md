# Week 1 Progress Notes: Data Pipeline and Setup

## 1. Goal of Week 1
The main objective of this week was to set up the project repository, inspect the raw dataset schemas, map key nutritional and socioeconomic features, design a plan to filter the data for Tamil Nadu, and establish the merge logic and quality check pipelines for the recode datasets.

## 2. Datasets Used
We are working with the National Family Health Survey (NFHS-5) files:
* **Kids Recode (KR):** Contains information on children's health, immunization, and nutritional status.
* **Individual Recode (IR):** Contains women's questionnaire data, including mother's education and BMI.
* **Household Recode (HR):** Contains household characteristics, wealth index, and toilet/water facilities.

## 3. What Was Completed
* **Metadata Inspection:** Analyzed the structures, data formats, and schemas of the raw datasets to understand the encoding of variables.
* **Feature Mapping:** Created a mapping of key predictors (e.g., mother's education, household wealth, sanitation) against the target nutritional metrics (stunting and wasting Z-scores).
* **Tamil Nadu Filtering Plan:** Developed a targeted loading strategy using specific state codes to extract only the records relevant to Tamil Nadu, reducing memory usage.
* **Merge Logic:** Formulated a join strategy using household and individual identifiers to merge the KR, IR, and HR datasets correctly.
* **Quality Check Pipeline:** Implemented initial validation scripts to check for data types, range constraints, and missing values across the variables.

## 4. Current Pending Issues
* **Confirm raw file names:** Double-check and align on the exact filenames of the large raw datasets on the storage system.
* **Verify Tamil Nadu state code:** Ensure the state code used for filtering matches the official recode definitions for Tamil Nadu.
* **Run scripts on real raw data:** Execute the metadata extraction and merge scripts on the complete, non-dummy raw datasets.

## 5. Expected Week 2 Work
* **Data Cleaning:** Handle missing values, resolve flagged Z-score outliers, and encode categorical variables.
* **Correlation Matrix:** Generate a correlation heatmap to examine relationships between predictors and target variables.
* **Feature Selection:** Apply statistical tests to filter and select the most relevant features for the analysis.
* **Baseline Modeling:** Develop initial predictive models to set a performance benchmark for our nutritional outcomes.
