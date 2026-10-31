# DASE7506 MP2 — Final V8

Wang Junxin · 3036747408

This repository is the frozen V8 publication snapshot. The official complete public test result is **395/467 = 84.58244111349036%**.

- [Editable English report (8 pages)](report/MP2_Report_Wang_Junxin_V8_Final.docx)
- [Installation, prediction, scoring, training and nested CV instructions](README.txt)
- [Official score record](evidence/reproduction/test-score.json)
- [Selection chronology](evidence/final_selection.json)

The default inference entry is `code/submission.py`, using the matching `checkpoint/`. Inference requires NumPy only; no retraining is needed. Fixed training and nested cross-validation code are included.

Release tag: **`v8-final-20261031`**.

- [Complete package: MP2_V8_full_20261031.zip](https://github.com/Henrystation/dase7506-mp2/releases/download/v8-final-20261031/MP2_V8_full_20261031.zip)
- [Checkpoint: MP2_V8_checkpoint_20261031.zip](https://github.com/Henrystation/dase7506-mp2/releases/download/v8-final-20261031/MP2_V8_checkpoint_20261031.zip)

The complete archive extracts to `final_v8_20261031/`; the checkpoint archive extracts to `model_bundle/`. Each release asset is byte-for-byte identical to the corresponding frozen delivery ZIP. `PACKAGE_MANIFEST.json` and `SHA256SUMS.txt` cover the original release payload; this GitHub display page and Git configuration files are publishing additions.

Historical status fields describe the stage when each record was generated. The final designation occurred after retrospective test results were available; the preserved records explain that chronology and the earlier stability-gate outcome.

I reviewed relevant methods and discussed candidate approaches with Codex. Codex assisted with approach screening, coding, experiments, result checks and report preparation; I made the final method choice and am responsible for the submitted work.
