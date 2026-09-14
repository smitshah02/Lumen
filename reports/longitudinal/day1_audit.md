# Lumen — Longitudinal HF Readmission: Day-1 Audit

- Generated: 2026-09-14 19:29:04
- Config: `configs/longitudinal_hf.yaml` (version 1.0.1, frozen=True)
- Connection: src.storage.engine (project configuration)
- Database: `lumen` in container `lumen-pg`
- **MIMIC version: `unknown`**  _(not manually confirmed — must be resolved before results are reported)_
- Access mode: **READ-ONLY** (session read-only + SELECT/WITH-only statement guard)

- Result: **PASSED** (0 error(s), 8 warning(s))

## 1. Database & schema availability

| table | class | status | cols found | cols expected | missing |
|---|---|---|---|---|---|
| patients | required | OK | 6 | 6 | — |
| admissions | required | OK | 9 | 9 | — |
| diagnoses_icd | required | OK | 5 | 5 | — |
| labevents | expected | OK | 9 | 9 | — |
| d_labitems | expected | OK | 4 | 4 | — |
| prescriptions | expected | OK | 9 | 9 | — |
| clinical_notes | expected | OK | 4 | 4 | — |
| procedures_icd | expected | OK | 5 | 5 | — |


### Row counts

| table | rows |
|---|---|
| patients | 5,000 |
| admissions | 32,170 |
| diagnoses_icd | 448,096 |
| labevents | 995,988 |
| d_labitems | 1,650 |
| prescriptions | 1,344,746 |
| clinical_notes | 169,061 |
| procedures_icd | 48,779 |


## 2. Cohort definition (FROZEN)

- **Heart failure**: `icd_code` starting with I50 (ICD-10) or 428 (ICD-9), matched on `UPPER(TRIM(icd_code))`, any diagnosis position (`seq_num` unrestricted).
- **Index admission**: Any admission carrying a qualifying HF diagnosis code, with non-null admittime and dischtime, where the patient survived the hospitalization.
- **Exclusions**: `hospital_expire_flag = 1`; `deathtime IS NOT NULL`; `admittime IS NULL OR dischtime IS NULL`
- **Unit of analysis**: index_admission (a patient may contribute more than one index admission).

## 3. Outcome definition (FROZEN)

- **Primary** `readmit_90d`: next recorded hospital admission > 0 and <= 90 days after index `dischtime`.
- **Secondary** `readmit_30d`: same rule at <= 30 days.
- **Prediction cutoff**: `admissions.dischtime`.
- **Next admission scope**: any cause (the next admission need not carry an HF code).

## 4. Cohort flow and counts

**Cohort flow** — each stage is checked against its own frozen reference.

| stage | observed | reference | delta | drift |
|---|---|---|---|---|
| Raw HF admissions (before exclusions) | 7,442 | 7,442 | +0 | 0.00% PASS |
| Excluded — in-hospital death | 238 | 238 | +0 | 0.00% PASS |
| Excluded — missing discharge time | 0 | 0 | +0 | n/a PASS |
| Final eligible readmission cohort | 7,204 | 7,204 | +0 | 0.00% PASS |


Reconciliation: 7,442 raw − 238 in-hospital death − 0 missing discharge time = **7,204** (reconciles with the eligible cohort of 7,204).


**Supporting counts**

| metric | value |
|---|---|
| Patients in DB | 5,000 |
| Admissions in DB | 32,170 |
| HF patients (any qualifying code) | 1,881 |
| Distinct patients contributing index admissions | 1,848 |
| Index admissions per contributing patient (mean) | 3.90 |


**Longitudinal depth of HF patients** (descriptive; not an eligibility filter)

| metric | value |
|---|---|
| HF patients | 1,881 |
| … with >= 2 admissions | 1,881 (100.00%) |
| … with >= 3 admissions | 1,881 (100.00%) |
| … with >= 5 admissions | 1,132 (60.18%) |
| Mean admissions per HF patient | 7.46 |
| Max admissions for one HF patient | 89 |


## 5. Outcome counts and rates

Outcome = next recorded admission in **(0, W] days** after index discharge, computed with `LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id)` over all admissions, then restricted to eligible index admissions.

| outcome | n | denominator | rate |
|---|---|---|---|
| readmit_30d (secondary) | 2,041 | 7,204 | 28.33% |
| readmit_90d (primary) | 3,383 | 7,204 | 46.96% |
| any later admission (unbounded) | 6,006 | 7,204 | 83.37% |
| no readmission within 90d | 3,821 | 7,204 | 53.04% |


Median days to readmission among 90-day readmits: **22.0**


Competing risk — died within 90d of discharge with no recorded readmission: **333** (4.62% of index admissions).


**Outcome counts vs frozen reference** (cohort size is reconciled in section 4)

| metric | observed | reference | delta | drift | status |
|---|---|---|---|---|---|
| readmit_30d n | 2,041 | 2,041 | +0 | 0.00% | PASS |
| readmit_90d n | 3,383 | 3,383 | +0 | 0.00% | PASS |


- `readmit_30d` rate: observed 28.33% vs reference 28.33% (denominator = eligible cohort, not raw HF count).

- `readmit_90d` rate: observed 46.96% vs reference 46.96% (denominator = eligible cohort, not raw HF count).



## 6. Data availability checklist

### 6.1 Date integrity

| check | rows affected | expectation | status |
|---|---|---|---|
| dischtime < admittime | 8 | must be 0 | FAIL |
| dischtime = admittime (zero-length stay) | 1 | review | WARN |
| deathtime < admittime | 2 | must be 0 | FAIL |
| deathtime > dischtime | 46 | review | WARN |
| hospital_expire_flag=1 but deathtime NULL | 0 | review | PASS |
| deathtime present but expire_flag=0 | 0 | review | PASS |
| length of stay > 365 days | 1 | review | WARN |
| next admittime < current dischtime (overlap) | 2 | review | WARN |
| dod earlier than last discharge (>1d) | 1 | review | WARN |


### 6.2 Missingness in key outcome/time fields

| table | column | rows | nulls | % missing |
|---|---|---|---|---|
| admissions | admittime | 32,170 | 0 | 0.00% |
| admissions | dischtime | 32,170 | 0 | 0.00% |
| admissions | deathtime | 32,170 | 31,623 | 98.30% |
| admissions | admission_type | 32,170 | 0 | 0.00% |
| admissions | admission_location | 32,170 | 0 | 0.00% |
| admissions | discharge_location | 32,170 | 0 | 0.00% |
| admissions | hospital_expire_flag | 32,170 | 0 | 0.00% |
| patients | gender | 5,000 | 0 | 0.00% |
| patients | anchor_age | 5,000 | 0 | 0.00% |
| patients | anchor_year | 5,000 | 0 | 0.00% |
| patients | anchor_year_group | 5,000 | 0 | 0.00% |
| patients | dod | 5,000 | 3,181 | 63.62% |
| diagnoses_icd | hadm_id | 448,096 | 0 | 0.00% |
| diagnoses_icd | seq_num | 448,096 | 0 | 0.00% |
| diagnoses_icd | icd_code | 448,096 | 0 | 0.00% |
| diagnoses_icd | icd_version | 448,096 | 0 | 0.00% |


### 6.3 Mortality fields

| field | source | n | coverage |
|---|---|---|---|
| dod (date of death) | patients.dod | 1,819 | 36.38% of patients |
| deathtime | admissions.deathtime | 547 | admissions with in-hospital death time |
| hospital_expire_flag = 1 | admissions | 547 | admissions flagged as died |
| hospital_expire_flag IS NULL | admissions | 0 | must be 0 for a clean exclusion |
| HF admissions excluded for in-hospital death | derived | 238 | — |


### 6.4 Clinical notes

| metric | value |
|---|---|
| Clinical notes | 169,061 |
| Notes with NULL charttime (unusable under the cutoff rule) | 0 (0.00%) |
| Index admissions with >=1 pre-cutoff note | 7,198 (99.92%) |



| note_type | rows |
|---|---|
| radiology | 145,089 |
| discharge | 23,972 |


## 7. Candidate-variable availability

### 7.1 Laboratory concepts

| concept | itemids matched | lab rows | % numeric | index admissions with >=1 pre-cutoff value | status |
|---|---|---|---|---|---|
| creatinine | 20 | 32,530 | 99.90% | 7,185 (99.74%) | OK |
| bun_urea | 10 | 27,585 | 99.99% | 7,184 (99.72%) | OK |
| sodium | 13 | 28,500 | 99.79% | 7,133 (99.01%) | OK |
| potassium | 13 | 30,071 | 99.93% | 7,150 (99.25%) | OK |
| hemoglobin | 25 | 33,483 | 99.75% | 7,199 (99.93%) | OK |
| bnp | 2 | 817 | 96.94% | 2,124 (29.48%) | OK |


**Matched itemids (first 6 per concept)**

| concept | itemid | label | fluid | category | rows | numeric rows |
|---|---|---|---|---|---|---|
| creatinine | 51067 | 24 hr Creatinine | Urine | Chemistry | 77 | 77 |
| creatinine | 51070 | Albumin/Creatinine, Urine | Urine | Chemistry | 758 | 733 |
| creatinine | 51963 | Amylase/Creatinine Clearance | Urine | Chemistry | 0 | 0 |
| creatinine | 51073 | Amylase/Creatinine Ratio, Urine | Urine | Chemistry | 1 | 1 |
| creatinine | 50912 | Creatinine | Blood | Chemistry | 29,407 | 29,405 |
| creatinine | 52546 | Creatinine | Blood | Chemistry | 0 | 0 |
| bun_urea | 51842 | Bun | Other Body Fluid | Chemistry | 0 | 0 |
| bun_urea | 51006 | Urea Nitrogen | Blood | Chemistry | 27,305 | 27,303 |
| bun_urea | 52647 | Urea Nitrogen | Blood | Chemistry | 0 | 0 |
| bun_urea | 50851 | Urea Nitrogen, Ascites | Ascites | Chemistry | 0 | 0 |
| bun_urea | 51045 | Urea Nitrogen, Body Fluid | Other Body Fluid | Chemistry | 1 | 1 |
| bun_urea | 51804 | Urea Nitrogen, CSF | Cerebrospinal Fluid | Chemistry | 0 | 0 |
| sodium | 52623 | Sodium | Blood | Chemistry | 0 | 0 |
| sodium | 50983 | Sodium | Blood | Chemistry | 26,609 | 26,604 |
| sodium | 50848 | Sodium, Ascites | Ascites | Chemistry | 0 | 0 |
| sodium | 50834 | Sodium, Body Fluid | Other Body Fluid | Blood Gas | 0 | 0 |
| sodium | 51042 | Sodium, Body Fluid | Other Body Fluid | Chemistry | 0 | 0 |
| sodium | 51801 | Sodium, CSF | Cerebrospinal Fluid | Chemistry | 0 | 0 |
| potassium | 52610 | Potassium | Blood | Chemistry | 0 | 0 |
| potassium | 50833 | Potassium | Other Body Fluid | Blood Gas | 0 | 0 |
| potassium | 50971 | Potassium | Blood | Chemistry | 27,334 | 27,323 |
| potassium | 50847 | Potassium, Ascites | Ascites | Chemistry | 0 | 0 |
| potassium | 51041 | Potassium, Body Fluid | Other Body Fluid | Chemistry | 0 | 0 |
| potassium | 51800 | Potassium, CSF | Cerebrospinal Fluid | Chemistry | 0 | 0 |
| hemoglobin | 50855 | Absolute Hemoglobin | Blood | Chemistry | 45 | 0 |
| hemoglobin | 50805 | Carboxyhemoglobin | Blood | Blood Gas | 60 | 60 |
| hemoglobin | 51212 | Fetal Hemoglobin | Blood | Hematology | 1 | 1 |
| hemoglobin | 51631 | Glycated Hemoglobin | Blood | Chemistry | 0 | 0 |
| hemoglobin | 51640 | Hemoglobin | Blood | Chemistry | 0 | 0 |
| hemoglobin | 50811 | Hemoglobin | Blood | Blood Gas | 1,661 | 1,661 |
| bnp | 50963 | NTproBNP | Blood | Chemistry | 812 | 787 |
| bnp | 51921 | proBNP, Pleural | Pleural | Chemistry | 5 | 5 |


### 7.2 Medications (names only — no features engineered at Day 1)

| metric | value |
|---|---|
| Prescription rows | 1,344,746 |
| Distinct drug names | 3,238 |
| Missing drug name | 0 (0.00%) |
| Missing starttime | 1,798 (0.13%) |
| Missing route | 0 (0.00%) |
| Missing dose_val_rx | 0 (0.00%) |
| Index admissions with >=1 pre-cutoff prescription | 6,973 (96.79%) |


**HF-relevant drug-name probes** (name availability only — no features built)

| name pattern | rows matched | status |
|---|---|---|
| furosemide% | 33,915 | present |
| torsemide% | 5,000 | present |
| bumetanide% | 1,179 | present |
| lisinopril% | 8,657 | present |
| losartan% | 2,819 | present |
| sacubitril% | 153 | present |
| carvedilol% | 3,655 | present |
| metoprolol% | 27,919 | present |
| spironolactone% | 3,055 | present |
| eplerenone% | 126 | present |
| dapagliflozin% | 11 | present |
| empagliflozin% | 24 | present |
| digoxin% | 1,529 | present |


**Top 20 drug names within eligible index admissions**

| drug | rows |
|---|---|
| insulin | 26,943 |
| furosemide | 21,327 |
| potassium chloride | 17,933 |
| sodium chloride 0.9%  flush | 11,632 |
| 0.9% sodium chloride | 10,222 |
| acetaminophen | 9,211 |
| 5% dextrose | 8,729 |
| metoprolol tartrate | 8,379 |
| warfarin | 8,027 |
| heparin | 7,484 |
| magnesium sulfate | 6,968 |
| bag | 6,760 |
| senna | 6,681 |
| iso-osmotic dextrose | 5,888 |
| docusate sodium | 5,634 |
| aspirin | 5,247 |
| vancomycin | 4,974 |
| sodium chloride 0.9% | 4,486 |
| torsemide | 4,475 |
| oxycodone (immediate release) | 4,254 |


Full candidate list with roles, timestamps and leakage rules: `reports/longitudinal/data_dictionary.csv`.


## 8. Leakage rules (FROZEN)

| id | rule |
|---|---|
| LR1 | No predictor may use data timestamped after the index dischtime. |
| LR2 | No future labevents (charttime > cutoff). |
| LR3 | No future prescriptions (starttime > cutoff). |
| LR4 | No diagnoses from admissions later than the index admission. |
| LR5 | No future admissions except as the outcome itself. |
| LR6 | No clinical notes with charttime > cutoff. |
| LR7 | Discharge-time fields that encode the outcome (discharge_location, deathtime, dod, hospital_expire_flag) are outcome/QC fields, not predictors. |
| LR8 | Never use absolute calendar year, anchor_year or anchor_year_group as a population-level temporal feature — dates are patient-shifted. |
| LR9 | Index-admission diagnoses are billing codes finalized at discharge; they are allowed as comorbidity predictors but must be flagged as discharge-coded in the data dictionary. |
| LR10 | Any aggregate (mean, imputation value, scaler, encoder) must be fit on training patients only, never on the full cohort. |


**Split rule**: patient-level on `subject_id` (0.7/0.15/0.15, seed 20260912, stratified on readmit_90d). Admission-level splitting is prohibited.


## 9. Warnings and limitations

### Warnings

- 7 index admissions have a next admittime <= dischtime (overlapping/erroneous admissions). These are correctly scored as non-events by the strict `> dischtime` bound but should be inspected on Day 2.
- Impossible dates: dischtime < admittime affects 8 admissions.
- Date anomaly: dischtime = admittime (zero-length stay) affects 1 admissions.
- Impossible dates: deathtime < admittime affects 2 admissions.
- Date anomaly: deathtime > dischtime affects 46 admissions.
- Date anomaly: length of stay > 365 days affects 1 admissions.
- Date anomaly: next admittime < current dischtime (overlap) affects 2 admissions.
- Date anomaly: dod earlier than last discharge (>1d) affects 1 admissions.

### Standing limitations

- **Date shifting.** MIMIC shifts each patient's timeline by a random per-patient
  offset. Within-patient intervals are valid; absolute calendar time, `anchor_year`
  and `anchor_year_group` must never be used as population-level temporal features.
- **Observability.** Readmissions to hospitals outside this database are invisible,
  so the observed rates are lower bounds and the negative class is contaminated.
- **Competing risk.** Post-discharge death censors the readmission outcome; the
  count is reported in section 5.
- **Discharge-coded diagnoses.** ICD codes are finalized at discharge, so index-
  admission comorbidities are only cutoff-safe under the convention in LR9.
- **Cohort size.** This is a 5,000-patient development subset, not full MIMIC-IV;
  rates here should not be read as population estimates.
- **Repeated measures.** Patients contribute multiple index admissions, so the
  patient-level split in section 8 is mandatory, not a preference.
- **Not clinical decision support.** Retrospective research/portfolio work only.


## 10. MIMIC version

`unknown`  — **unconfirmed.** Confirm from the PhysioNet download directory name or the dataset CHANGELOG before publishing any results.

---
_Generated by `scripts/audit_longitudinal_day1.py` (read-only)._
