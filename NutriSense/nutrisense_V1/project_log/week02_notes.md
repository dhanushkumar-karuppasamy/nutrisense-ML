# Week 2 Notes

## Goal
Build the 3-way merge pipeline (KR + IR + HR) for Tamil Nadu NFHS5.

## Tasks Completed
- Implemented chunked reading via pyreadstat to avoid OOM on ~500k-row raw files
- DHS sentinel cleaning for HAZ and BMI columns
- 3-way inner merge: KR × IR (on caseid), then × HR (on v001+v002)
- Output: tamilnadu_nfhs5_merged.csv — 13 columns, ~[YOUR ROW COUNT] rows

## Key Decisions
- Inner join (not left): drops children without household match (~2-3% loss), 
  ensures all features are present for every row
- Chunked loading: 50k rows per chunk balances memory vs. speed

## Gap Identified (Week 3 action item)
- Only 6 socioeconomic columns pulled; no clinical variables
- No imputation logic — just sentinel→NaN; model cannot consume these yet

## Next Week
- Expand to 22-column clinical+socioeconomic feature set
- Add imputation strategy
- Begin EDA notebook
- Baseline stacking model