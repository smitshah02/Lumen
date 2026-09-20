"""
Synthetic demo corpus generator
===============================
Writes the fictional Lumen demo dataset (JSON) into this directory. Every
patient, admission, lab, medication, diagnosis and note below is AUTHORED here
from invented scenarios — nothing is read from, sampled from, or modelled on
MIMIC or any other patient record. See README.md.

Deterministic: fixed SEED, no wall-clock input, stable JSON output. Notes are
rendered from the same structured facts that populate the lab / prescription /
diagnosis tables, so the golden QA facts are consistent by construction (the
generator asserts every expected fact appears in that patient's notes).

    python -m src.demo_data.generate             # rewrite the JSON files
    python -m src.demo_data.generate --check     # verify files are up to date
    python -m src.demo_data.generate --validate  # integrity checks only, writes nothing
"""

from __future__ import annotations

import re
import sys
import json
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20260918
OUT = Path(__file__).resolve().parent
VERSION = "lumen-demo-v1"

# Clearly artificial ID ranges (MIMIC subject_ids are 8-digit 1xxxxxxx).
SUBJECT_BASE, HADM_BASE, NOTE_BASE, LAB_BASE, RX_BASE = 90000000, 91000000, 92000000, 93000000, 94000000

# label -> (itemid, unit, fluid, category, decimals). Generic lab names; itemids are synthetic.
LAB_ITEMS = {
    "Creatinine":        (990001, "mg/dL",  "Blood", "Chemistry",  1),
    "Urea Nitrogen":     (990002, "mg/dL",  "Blood", "Chemistry",  0),
    "Potassium":         (990003, "mEq/L",  "Blood", "Chemistry",  1),
    "Sodium":            (990004, "mEq/L",  "Blood", "Chemistry",  0),
    "Glucose":           (990005, "mg/dL",  "Blood", "Chemistry",  0),
    "Hemoglobin":        (990006, "g/dL",   "Blood", "Hematology", 1),
    "White Blood Cells": (990007, "K/uL",   "Blood", "Hematology", 1),
    "Platelet Count":    (990008, "K/uL",   "Blood", "Hematology", 0),
    "Hemoglobin A1c":    (990009, "%",      "Blood", "Chemistry",  1),
    "NT-proBNP":         (990010, "pg/mL",  "Blood", "Chemistry",  0),
    "Lactate":           (990011, "mmol/L", "Blood", "Chemistry",  1),
    "Ferritin":          (990012, "ng/mL",  "Blood", "Chemistry",  0),
    "Troponin T":        (990013, "ng/mL",  "Blood", "Chemistry",  2),
    "Bicarbonate":       (990014, "mEq/L",  "Blood", "Chemistry",  0),
    "Albumin":           (990015, "g/dL",   "Blood", "Chemistry",  1),
    "Total Bilirubin":   (990016, "mg/dL",  "Blood", "Chemistry",  1),
    "INR":               (990017, "",       "Blood", "Coagulation", 1),
    "Alanine Aminotransferase": (990018, "IU/L", "Blood", "Chemistry", 0),
    "Lipase":            (990019, "IU/L",   "Blood", "Chemistry",  0),
    "C-Reactive Protein": (990020, "mg/L",  "Blood", "Chemistry",  0),
    "Magnesium":         (990021, "mg/dL",  "Blood", "Chemistry",  1),
    "Calcium":           (990022, "mg/dL",  "Blood", "Chemistry",  1),
    "Thyroid Stimulating Hormone": (990023, "uIU/mL", "Blood", "Chemistry", 2),
    "Absolute Neutrophil Count": (990024, "K/uL", "Blood", "Hematology", 1),
    "Vitamin B12":       (990025, "pg/mL",  "Blood", "Chemistry",  0),
}
SUPPORT_LABS = ["Sodium", "Potassium", "Urea Nitrogen", "Hemoglobin", "White Blood Cells", "Glucose", "Platelet Count"]
SUPPORT_SPREAD = {"Sodium": 2, "Potassium": 0.3, "Urea Nitrogen": 4, "Hemoglobin": 0.5,
                  "White Blood Cells": 1.0, "Glucose": 15, "Platelet Count": 25}


def M(drug, dose, route, freq, status="continued", note=""):
    return {"drug": drug, "dose": dose, "route": route, "freq": freq, "status": status, "note": note}


def IMG(kind, time, indication, comparison, findings, impression):
    return {"exam": kind, "time": time, "indication": indication, "comparison": comparison,
            "findings": findings, "impression": impression}


def PROG(time, day, interval, exam, ap, labs=None):
    return {"time": time, "day": day, "interval": interval, "exam": exam, "ap": ap, "labs": labs or []}


# ===========================================================================
# Fictional patients
# ===========================================================================
PATIENTS = [
    # ---- A: HFrEF, three admissions, diuretic + GDMT changes ----------------
    {"n": 1, "sex": "M", "age": 67, "pmh_short": "ischemic cardiomyopathy (LVEF 30%), coronary artery disease, and hypertension",
     "pmh": ["Heart failure with reduced ejection fraction due to ischemic cardiomyopathy (LVEF 30%)",
             "Coronary artery disease, percutaneous coronary intervention to the LAD in 2019",
             "Hypertension", "Hyperlipidemia"],
     "social": "Retired machinist. Former smoker, quit in 2015. Lives with his spouse. No alcohol use.",
     "base": {"Sodium": 137, "Potassium": 4.2, "Urea Nitrogen": 26, "Hemoglobin": 13.2, "White Blood Cells": 7.1, "Glucose": 118, "Platelet Count": 214},
     "admissions": [
        {"admit": "2023-01-10 14:20", "disch": "2023-01-15 11:05", "service": "Cardiology",
         "cc": "Progressive shortness of breath, orthopnea, and bilateral leg swelling.",
         "hpi": "He reports three weeks of worsening dyspnea on exertion, two-pillow orthopnea, and a 4 kg weight gain. He had not previously been prescribed a loop diuretic. He ate more salty food than usual over the holidays.",
         "vitals": "temperature 36.7 C, blood pressure 148/92, heart rate 98, respiratory rate 22, oxygen saturation 91% on room air",
         "exam": "Jugular venous pressure elevated to 12 cm. Bibasilar crackles. 2+ pitting edema of both lower extremities to the knees.",
         "course": "Admitted with acute decompensated heart failure and volume overload. Furosemide was started as 40 mg IV twice daily, with net negative 5.2 liters and a 4.6 kg weight loss over the admission. Creatinine rose from 1.3 to a peak of 1.8 mg/dL on hospital day 3 during diuresis and improved to 1.5 mg/dL by discharge. Lisinopril was held for one day and resumed at 5 mg daily. Metoprolol succinate was continued. He was transitioned to oral furosemide 40 mg daily on day 4 and remained euvolemic.",
         "labs": {"Creatinine": [("2023-01-10 15:02", 1.3), ("2023-01-12 06:10", 1.8), ("2023-01-15 06:05", 1.5)],
                  "NT-proBNP": [("2023-01-10 15:02", 6850)]},
         "dx": [("I50.23", "Acute on chronic systolic heart failure"), ("I11.0", "Hypertensive heart disease with heart failure"),
                ("I25.10", "Coronary artery disease"), ("E78.5", "Hyperlipidemia")],
         "inpatient": [M("Furosemide", "40 mg", "IV", "twice daily")],
         "meds": [M("Furosemide", "40 mg", "PO", "daily", "new", "first loop diuretic"),
                  M("Metoprolol succinate", "50 mg", "PO", "daily"), M("Lisinopril", "5 mg", "PO", "daily"),
                  M("Atorvastatin", "80 mg", "PO", "nightly"), M("Aspirin", "81 mg", "PO", "daily")],
         "imaging": [IMG("Chest radiograph, PA and lateral", "2023-01-10 16:40", "Dyspnea and leg swelling, evaluate for pulmonary edema.",
                         "Chest radiograph from 2021.",
                         "The cardiac silhouette is enlarged. There is cephalization of the pulmonary vasculature with bilateral interstitial opacities and Kerley B lines. Small bilateral pleural effusions are present. No focal consolidation or pneumothorax.",
                         "Cardiomegaly with moderate pulmonary edema and small bilateral pleural effusions, consistent with volume overload.")],
         "followup": "Heart failure clinic in 1 week with a basic metabolic panel. Weigh daily and call for a gain of more than 1 kg in a day or 2 kg in a week. Limit sodium to 2 g per day."},
        {"admit": "2023-08-02 09:40", "disch": "2023-08-07 12:30", "service": "Cardiology",
         "cc": "Shortness of breath and abdominal fullness.",
         "hpi": "He reports ten days of recurrent dyspnea, orthopnea, and a 3 kg weight gain despite taking furosemide 40 mg daily, which was started during his January admission. He ate restaurant meals frequently while traveling.",
         "vitals": "temperature 36.6 C, blood pressure 138/86, heart rate 92, respiratory rate 20, oxygen saturation 93% on room air",
         "exam": "Jugular venous pressure 14 cm. Crackles at both lung bases. Mild ascites. 2+ lower extremity edema.",
         "course": "Second admission for acute decompensated heart failure. Diuresed with furosemide 80 mg IV twice daily. Creatinine increased from 1.7 to 2.1 mg/dL on hospital day 3, consistent with cardiorenal physiology, and improved to 1.6 mg/dL at discharge as congestion resolved. Home furosemide was increased from 40 mg to 80 mg daily and spironolactone 12.5 mg daily was added. Potassium remained normal.",
         "labs": {"Creatinine": [("2023-08-02 10:15", 1.7), ("2023-08-04 05:50", 2.1), ("2023-08-07 06:00", 1.6)],
                  "NT-proBNP": [("2023-08-02 10:15", 5120)],
                  "Potassium": [("2023-08-02 10:15", 4.4), ("2023-08-07 06:00", 4.6)]},
         "dx": [("I50.23", "Acute on chronic systolic heart failure"), ("N17.9", "Acute kidney injury"),
                ("I11.0", "Hypertensive heart disease with heart failure"), ("I25.10", "Coronary artery disease")],
         "inpatient": [M("Furosemide", "80 mg", "IV", "twice daily")],
         "meds": [M("Furosemide", "80 mg", "PO", "daily", "changed", "increased from 40 mg daily"),
                  M("Spironolactone", "12.5 mg", "PO", "daily", "new"),
                  M("Metoprolol succinate", "50 mg", "PO", "daily"), M("Lisinopril", "5 mg", "PO", "daily"),
                  M("Atorvastatin", "80 mg", "PO", "nightly"), M("Aspirin", "81 mg", "PO", "daily")],
         "imaging": [IMG("Chest radiograph, PA and lateral", "2023-08-02 11:05", "Recurrent dyspnea in a patient with heart failure.",
                         "Chest radiograph from 2023-01-10.",
                         "Stable cardiomegaly. Pulmonary vascular congestion with mild interstitial edema, less pronounced than on the January study. Trace left pleural effusion. No consolidation.",
                         "Mild pulmonary edema, improved compared with 2023-01-10. Stable cardiomegaly.")],
         "progress": [PROG("2023-08-04 10:30", 3,
                           "Breathing is better and he slept flat for the first time. Urine output 3.1 liters over 24 hours.",
                           "Jugular venous pressure 10 cm, fewer crackles, 1+ edema.",
                           "1. Acute decompensated heart failure with volume overload: improving on IV furosemide 80 mg twice daily, net negative 4.0 liters so far. Continue diuresis and daily weights.\n2. Acute kidney injury, creatinine 2.1 mg/dL from 1.7 mg/dL: likely cardiorenal; continue decongestion and recheck in the morning.\n3. Heart failure with reduced ejection fraction: continue metoprolol succinate; hold lisinopril today.",
                           labs=["Creatinine"])],
         "followup": "Heart failure clinic in 7 days with a basic metabolic panel to check potassium and creatinine on spironolactone."},
        {"admit": "2024-03-12 13:15", "disch": "2024-03-18 10:40", "service": "Cardiology",
         "cc": "Dyspnea on exertion and leg swelling.",
         "hpi": "This is his third heart failure hospitalization. He reports two weeks of increasing dyspnea and edema despite furosemide 80 mg daily and describes a variable response to oral furosemide.",
         "vitals": "temperature 36.8 C, blood pressure 132/80, heart rate 88, respiratory rate 18, oxygen saturation 94% on room air",
         "exam": "Jugular venous pressure 11 cm. Scattered bibasilar crackles. 1+ edema to the mid shins.",
         "course": "Third admission for acute decompensated heart failure, milder than prior episodes. Diuresed with IV furosemide, then switched from furosemide to torsemide 20 mg daily for more reliable oral absorption. Guideline-directed medical therapy was optimized: lisinopril was stopped and, after a 36-hour washout, sacubitril-valsartan 24-26 mg twice daily was started. Metoprolol succinate was increased from 50 mg to 100 mg daily. Creatinine was 1.6 mg/dL on admission, 1.7 mg/dL on day 3, and improved to 1.4 mg/dL at discharge. Transthoracic echocardiogram showed a left ventricular ejection fraction of 30%.",
         "labs": {"Creatinine": [("2024-03-12 14:00", 1.6), ("2024-03-14 06:15", 1.7), ("2024-03-18 06:00", 1.4)],
                  "NT-proBNP": [("2024-03-12 14:00", 3980)],
                  "Potassium": [("2024-03-12 14:00", 4.3), ("2024-03-18 06:00", 4.5)]},
         "dx": [("I50.23", "Acute on chronic systolic heart failure"), ("I25.5", "Ischemic cardiomyopathy"),
                ("I11.0", "Hypertensive heart disease with heart failure"), ("I34.0", "Nonrheumatic mitral valve insufficiency")],
         "inpatient": [M("Furosemide", "80 mg", "IV", "twice daily")],
         "meds": [M("Torsemide", "20 mg", "PO", "daily", "new", "replaces furosemide"),
                  M("Sacubitril-valsartan", "24-26 mg", "PO", "twice daily", "new", "replaces lisinopril"),
                  M("Metoprolol succinate", "100 mg", "PO", "daily", "changed", "increased from 50 mg daily"),
                  M("Spironolactone", "12.5 mg", "PO", "daily"),
                  M("Atorvastatin", "80 mg", "PO", "nightly"), M("Aspirin", "81 mg", "PO", "daily")],
         "stopped": [("Furosemide", "replaced by torsemide"), ("Lisinopril", "replaced by sacubitril-valsartan")],
         "imaging": [IMG("Transthoracic echocardiogram", "2024-03-13 11:20", "Heart failure with reduced ejection fraction, reassess left ventricular function.",
                         "Echocardiogram from 2022.",
                         "The left ventricle is mildly dilated with moderate global hypokinesis and akinesis of the anterior wall. Left ventricular ejection fraction is estimated at 30%. Moderate functional mitral regurgitation. Right ventricular function is mildly reduced. Estimated pulmonary artery systolic pressure is 44 mmHg.",
                         "Ischemic cardiomyopathy with left ventricular ejection fraction 30%, unchanged from 2022. Moderate functional mitral regurgitation.")],
         "followup": "Heart failure clinic in 1 week with a basic metabolic panel after starting sacubitril-valsartan. Continue daily weights."},
     ]},

    # ---- B: diabetic CKD with rising creatinine across admissions ---------
    {"n": 2, "sex": "F", "age": 72, "pmh_short": "type 2 diabetes mellitus with diabetic kidney disease and hypertension",
     "pmh": ["Type 2 diabetes mellitus for 15 years with diabetic retinopathy",
             "Chronic kidney disease due to diabetic nephropathy", "Hypertension", "Obesity"],
     "social": "Retired teacher. Never smoker. Lives alone; her daughter helps organize her medications.",
     "base": {"Sodium": 139, "Potassium": 4.6, "Urea Nitrogen": 38, "Hemoglobin": 11.4, "White Blood Cells": 7.6, "Glucose": 164, "Platelet Count": 245},
     "admissions": [
        {"admit": "2022-11-03 18:30", "disch": "2022-11-07 11:00", "service": "Medicine",
         "cc": "Polyuria, thirst, and fatigue.",
         "hpi": "She reports one week of polyuria, polydipsia, and fatigue with home glucose readings above 350 mg/dL. She was taking metformin 1000 mg twice daily and glipizide 5 mg daily.",
         "exam": "Dry mucous membranes. Mild bilateral ankle edema. Decreased monofilament sensation in both feet.",
         "course": "Admitted with hyperglycemia and volume depletion without ketoacidosis. Glucose was 412 mg/dL on arrival with a normal anion gap. Hemoglobin A1c was 8.9%. Creatinine was 2.0 mg/dL on admission and 1.9 mg/dL at discharge after IV fluids, with an estimated GFR of 27, consistent with chronic kidney disease stage 4. Metformin was discontinued because of reduced kidney function. Insulin glargine 14 units nightly was started and glipizide was continued. Diabetes education was completed.",
         "labs": {"Glucose": [("2022-11-03 19:05", 412), ("2022-11-07 06:00", 168)],
                  "Creatinine": [("2022-11-03 19:05", 2.0), ("2022-11-07 06:00", 1.9)],
                  "Hemoglobin A1c": [("2022-11-04 06:00", 8.9)]},
         "dx": [("E11.65", "Type 2 diabetes mellitus with hyperglycemia"), ("E11.22", "Type 2 diabetes mellitus with diabetic chronic kidney disease"),
                ("N18.4", "Chronic kidney disease, stage 4"), ("E86.0", "Dehydration"), ("I12.9", "Hypertensive chronic kidney disease")],
         "meds": [M("Insulin glargine", "14 units", "SC", "nightly", "new"), M("Glipizide", "5 mg", "PO", "daily"),
                  M("Lisinopril", "10 mg", "PO", "daily"), M("Amlodipine", "5 mg", "PO", "daily"), M("Atorvastatin", "40 mg", "PO", "nightly")],
         "stopped": [("Metformin", "discontinued for estimated GFR 27")],
         "followup": "Primary care in 1 week with fingerstick glucose log. Nephrology referral for chronic kidney disease stage 4."},
        {"admit": "2023-09-14 15:10", "disch": "2023-09-18 13:20", "service": "Medicine",
         "cc": "Redness and swelling of the right foot.",
         "hpi": "She reports three days of right foot redness, warmth, and swelling around a small plantar callus. No fever at home.",
         "exam": "Erythema and warmth over the dorsum of the right foot extending to the ankle, without fluctuance. Pedal pulses intact. Decreased monofilament sensation in both feet.",
         "course": "Admitted with right foot cellulitis in the setting of diabetic neuropathy. Foot radiographs showed no osteomyelitis. Treated with IV cefazolin and transitioned to oral cephalexin. Creatinine was 2.5 mg/dL on admission and 2.4 mg/dL at discharge, higher than her prior baseline of 1.9 mg/dL, consistent with progression of diabetic kidney disease. Hemoglobin A1c was 7.8%. Insulin glargine was increased from 14 to 18 units nightly. Dapagliflozin 10 mg daily was started for kidney protection.",
         "labs": {"Creatinine": [("2023-09-14 15:50", 2.5), ("2023-09-18 06:00", 2.4)],
                  "Hemoglobin A1c": [("2023-09-15 06:00", 7.8)],
                  "White Blood Cells": [("2023-09-14 15:50", 13.4), ("2023-09-18 06:00", 8.2)],
                  "Glucose": [("2023-09-14 15:50", 236), ("2023-09-18 06:00", 142)]},
         "dx": [("L03.115", "Cellulitis of right lower limb"), ("E11.40", "Type 2 diabetes mellitus with diabetic neuropathy"),
                ("E11.22", "Type 2 diabetes mellitus with diabetic chronic kidney disease"), ("N18.4", "Chronic kidney disease, stage 4")],
         "inpatient": [M("Cefazolin", "2 g", "IV", "every 8 hours")],
         "meds": [M("Cephalexin", "500 mg", "PO", "four times daily for 5 more days", "new"),
                  M("Dapagliflozin", "10 mg", "PO", "daily", "new"),
                  M("Insulin glargine", "18 units", "SC", "nightly", "changed", "increased from 14 units"),
                  M("Glipizide", "5 mg", "PO", "daily"), M("Lisinopril", "10 mg", "PO", "daily"),
                  M("Amlodipine", "5 mg", "PO", "daily"), M("Atorvastatin", "40 mg", "PO", "nightly")],
         "imaging": [IMG("Radiograph of the right foot, three views", "2023-09-14 17:20", "Right foot cellulitis in a diabetic patient, evaluate for osteomyelitis.",
                         "None.",
                         "Soft tissue swelling over the dorsum of the foot. No cortical erosion, periosteal reaction, or soft tissue gas. Mild degenerative change at the first metatarsophalangeal joint.",
                         "Soft tissue swelling without radiographic evidence of osteomyelitis.")],
         "followup": "Podiatry in 2 weeks. Primary care in 1 week. Nephrology as scheduled."},
        {"admit": "2024-10-21 11:45", "disch": "2024-10-25 10:15", "service": "Medicine",
         "cc": "Weakness and an abnormal potassium level.",
         "hpi": "She reports generalized weakness and nausea. Outpatient labs showed potassium 5.9 mEq/L and creatinine 3.1 mg/dL.",
         "exam": "Alert and mildly fatigued. Trace ankle edema. No asterixis.",
         "course": "Admitted with hyperkalemia and progressive chronic kidney disease. Electrocardiogram showed peaked T waves, treated with IV calcium gluconate and insulin with dextrose. Potassium improved from 5.9 to 4.8 mEq/L. Creatinine was 3.1 mg/dL on admission and at discharge, with an estimated GFR of 16. Lisinopril was stopped because of hyperkalemia and patiromer 8.4 g daily was started. Hemoglobin A1c was 7.1%. Glipizide was stopped because of hypoglycemia risk in advanced kidney disease. Renal ultrasound showed small echogenic kidneys. Nephrology recommended planning for kidney replacement therapy.",
         "labs": {"Potassium": [("2024-10-21 12:20", 5.9), ("2024-10-22 06:00", 5.2), ("2024-10-25 06:00", 4.8)],
                  "Creatinine": [("2024-10-21 12:20", 3.1), ("2024-10-25 06:00", 3.1)],
                  "Hemoglobin A1c": [("2024-10-22 06:00", 7.1)],
                  "Bicarbonate": [("2024-10-21 12:20", 18)]},
         "dx": [("E87.5", "Hyperkalemia"), ("N18.4", "Chronic kidney disease, stage 4"),
                ("E11.22", "Type 2 diabetes mellitus with diabetic chronic kidney disease"), ("I12.9", "Hypertensive chronic kidney disease")],
         "meds": [M("Patiromer", "8.4 g", "PO", "daily", "new"), M("Sodium bicarbonate", "650 mg", "PO", "twice daily", "new"),
                  M("Amlodipine", "10 mg", "PO", "daily", "changed", "increased from 5 mg daily"),
                  M("Insulin glargine", "18 units", "SC", "nightly"), M("Dapagliflozin", "10 mg", "PO", "daily"),
                  M("Atorvastatin", "40 mg", "PO", "nightly")],
         "stopped": [("Lisinopril", "stopped for hyperkalemia"), ("Glipizide", "stopped for hypoglycemia risk")],
         "imaging": [IMG("Renal ultrasound", "2024-10-22 14:00", "Progressive chronic kidney disease.", "None.",
                         "The right kidney measures 9.1 cm and the left kidney 9.4 cm. Both kidneys show increased cortical echogenicity with cortical thinning. No hydronephrosis, stones, or masses. The bladder is decompressed.",
                         "Small echogenic kidneys consistent with chronic medical renal disease. No hydronephrosis.")],
         "followup": "Nephrology in 1 week to discuss dialysis access planning. Repeat potassium in 3 days."},
     ]},

    # ---- C: COPD with community-acquired pneumonia, then improvement ------
    {"n": 3, "sex": "M", "age": 64, "pmh_short": "severe COPD and hypertension",
     "pmh": ["Chronic obstructive pulmonary disease, GOLD stage 3, on tiotropium", "Former smoker, 45 pack-years",
             "Hypertension", "Gastroesophageal reflux disease"],
     "social": "Former warehouse supervisor. Quit smoking in 2020. Lives with his partner.",
     "base": {"Sodium": 138, "Potassium": 4.0, "Urea Nitrogen": 16, "Hemoglobin": 14.6, "White Blood Cells": 8.4, "Glucose": 104, "Platelet Count": 268},
     "admissions": [
        {"admit": "2023-12-04 20:10", "disch": "2023-12-09 12:00", "service": "Medicine",
         "cc": "Fever, productive cough, and shortness of breath.",
         "hpi": "He reports four days of fever to 38.9 C, cough productive of yellow-green sputum, right-sided pleuritic chest pain, and worsening shortness of breath. Oxygen saturation was 86% on room air on arrival.",
         "vitals": "temperature 38.6 C, blood pressure 126/74, heart rate 112, respiratory rate 28, oxygen saturation 86% on room air",
         "exam": "Tachypneic with accessory muscle use. Bronchial breath sounds and crackles over the right lower lung field. Diffuse expiratory wheezes.",
         "course": "Admitted with community-acquired pneumonia of the right lower lobe complicated by acute hypoxemic respiratory failure and a COPD exacerbation. He required supplemental oxygen at 4 liters per minute by nasal cannula to keep oxygen saturation 88-92%. Treated with ceftriaxone and azithromycin, then oral amoxicillin-clavulanate to complete 7 days, plus prednisone 40 mg daily for 5 days and scheduled nebulized albuterol-ipratropium. White blood cell count fell from 16.2 to 9.8 K/uL. Oxygen was weaned to 2 liters on day 3 and to room air on day 4, with an ambulatory oxygen saturation of 91% on room air at discharge.",
         "labs": {"White Blood Cells": [("2023-12-04 20:40", 16.2), ("2023-12-06 06:00", 12.1), ("2023-12-09 06:00", 9.8)],
                  "Lactate": [("2023-12-04 20:40", 1.9)]},
         "dx": [("J18.1", "Lobar pneumonia, right lower lobe"), ("J96.01", "Acute respiratory failure with hypoxia"),
                ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"), ("I10", "Essential hypertension")],
         "inpatient": [M("Ceftriaxone", "1 g", "IV", "daily"), M("Azithromycin", "500 mg", "IV", "daily")],
         "meds": [M("Amoxicillin-clavulanate", "875-125 mg", "PO", "twice daily for 3 more days", "new"),
                  M("Prednisone", "40 mg", "PO", "daily for 1 more day", "new"),
                  M("Tiotropium", "18 mcg", "inhaled", "daily"), M("Albuterol inhaler", "2 puffs", "inhaled", "every 4 hours as needed"),
                  M("Lisinopril", "10 mg", "PO", "daily"), M("Omeprazole", "20 mg", "PO", "daily")],
         "imaging": [IMG("Chest radiograph, PA and lateral", "2023-12-04 21:15", "Fever, cough, and hypoxemia.", "Chest radiograph from 2022.",
                         "Dense consolidation in the right lower lobe with air bronchograms. Hyperinflated lungs with flattened hemidiaphragms. No pleural effusion or pneumothorax. Normal heart size.",
                         "Right lower lobe consolidation consistent with community-acquired pneumonia. Background emphysematous hyperinflation.")],
         "progress": [PROG("2023-12-05 09:15", 2,
                           "Still short of breath with minimal exertion. Febrile overnight to 38.4 C. Sputum culture pending.",
                           "Crackles at the right base, diffuse wheezes, speaking in full sentences.",
                           "1. Community-acquired pneumonia, right lower lobe: continue ceftriaxone and azithromycin, day 2 of therapy.\n2. Acute hypoxemic respiratory failure: requiring 4 liters of oxygen by nasal cannula, target saturation 88-92%; wean as tolerated.\n3. COPD exacerbation: prednisone 40 mg daily and scheduled nebulizers.",
                           labs=["White Blood Cells"])],
         "followup": "Primary care in 1 week. Repeat chest radiograph in 6 to 8 weeks to confirm resolution."},
        {"admit": "2024-02-20 07:50", "disch": "2024-02-23 11:30", "service": "Medicine",
         "cc": "Increased cough and wheezing.",
         "hpi": "He reports three days of increased cough with clear sputum and wheezing after a viral upper respiratory infection. No fever.",
         "exam": "Diffuse expiratory wheezes and prolonged expiration. No focal crackles.",
         "course": "Admitted with a COPD exacerbation triggered by a viral infection. Follow-up chest radiograph showed interval resolution of the previously seen right lower lobe consolidation. He required 2 liters of oxygen for one night and was then on room air. Treated with prednisone 40 mg daily for 5 days and nebulized bronchodilators; antibiotics were not needed. Maintenance inhaler therapy was escalated by adding budesonide-formoterol. Pulmonary rehabilitation referral placed.",
         "labs": {"White Blood Cells": [("2024-02-20 08:20", 8.9), ("2024-02-23 06:00", 9.4)]},
         "dx": [("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"), ("J06.9", "Acute upper respiratory infection"),
                ("I10", "Essential hypertension")],
         "meds": [M("Budesonide-formoterol", "160-4.5 mcg, 2 puffs", "inhaled", "twice daily", "new"),
                  M("Prednisone", "40 mg", "PO", "daily for 2 more days", "new"),
                  M("Tiotropium", "18 mcg", "inhaled", "daily"), M("Albuterol inhaler", "2 puffs", "inhaled", "every 4 hours as needed"),
                  M("Lisinopril", "10 mg", "PO", "daily"), M("Omeprazole", "20 mg", "PO", "daily")],
         "imaging": [IMG("Chest radiograph, PA and lateral", "2024-02-20 08:40", "COPD exacerbation; prior right lower lobe pneumonia.",
                         "Chest radiograph from 2023-12-04.",
                         "Interval resolution of the previously seen right lower lobe consolidation. Persistent hyperinflation with flattened hemidiaphragms. No new focal opacity, effusion, or pneumothorax.",
                         "Interval resolution of right lower lobe pneumonia. Chronic hyperinflation from emphysema without an acute process.")],
         "followup": "Pulmonology in 4 weeks. Pulmonary rehabilitation intake."},
     ]},

    # ---- D: atrial fibrillation, anticoagulation + rate/rhythm changes ----
    {"n": 4, "sex": "F", "age": 71, "pmh_short": "hypertension and hypothyroidism",
     "pmh": ["Hypertension", "Hypothyroidism", "Osteoarthritis of both knees"],
     "social": "Retired librarian. Never smoker. Drinks one glass of wine per week. Lives with her husband.",
     "base": {"Sodium": 140, "Potassium": 4.1, "Urea Nitrogen": 18, "Hemoglobin": 12.6, "White Blood Cells": 6.8, "Glucose": 98, "Platelet Count": 231},
     "admissions": [
        {"admit": "2022-06-08 16:05", "disch": "2022-06-11 12:10", "service": "Cardiology",
         "cc": "Palpitations and lightheadedness.",
         "hpi": "She reports two days of rapid, irregular palpitations and lightheadedness. The emergency department electrocardiogram showed atrial fibrillation with rapid ventricular response at 142 beats per minute.",
         "vitals": "temperature 36.9 C, blood pressure 118/76, heart rate 142 and irregular, respiratory rate 18, oxygen saturation 97% on room air",
         "exam": "Irregularly irregular tachycardia. Lungs clear. No edema.",
         "course": "New diagnosis of atrial fibrillation with rapid ventricular response. Rate controlled with IV diltiazem, then oral diltiazem extended-release 180 mg daily with heart rate 70-85. CHA2DS2-VASc score is 3 (age, female sex, hypertension), and anticoagulation was started with apixaban 5 mg twice daily after shared decision-making. Thyroid function was normal. Transthoracic echocardiogram showed normal left ventricular ejection fraction and a moderately dilated left atrium.",
         "labs": {"Hemoglobin": [("2022-06-08 16:40", 12.6), ("2022-06-11 06:00", 12.4)]},
         "dx": [("I48.91", "Atrial fibrillation"), ("I10", "Essential hypertension"), ("E03.9", "Hypothyroidism")],
         "inpatient": [M("Diltiazem", "10 mg/h", "IV", "continuous infusion")],
         "meds": [M("Diltiazem extended-release", "180 mg", "PO", "daily", "new"), M("Apixaban", "5 mg", "PO", "twice daily", "new"),
                  M("Levothyroxine", "75 mcg", "PO", "daily"), M("Losartan", "50 mg", "PO", "daily")],
         "imaging": [IMG("Transthoracic echocardiogram", "2022-06-09 10:00", "New atrial fibrillation.", "None.",
                         "Normal left ventricular size with ejection fraction 60%. The left atrium is moderately dilated. No significant valvular disease. No pericardial effusion.",
                         "Normal left ventricular systolic function, ejection fraction 60%. Moderate left atrial enlargement.")],
         "followup": "Cardiology clinic in 2 weeks. Ambulatory heart rhythm monitor for 7 days."},
        {"admit": "2023-05-16 03:40", "disch": "2023-05-20 14:00", "service": "Medicine",
         "cc": "Black stools and fatigue.",
         "hpi": "She reports two days of black, tarry stools and exertional fatigue while taking apixaban 5 mg twice daily and ibuprofen for knee pain.",
         "vitals": "temperature 36.8 C, blood pressure 104/62, heart rate 104 and irregular, respiratory rate 18, oxygen saturation 97% on room air",
         "exam": "Pale conjunctivae. Abdomen soft and non-tender. Melena on rectal examination.",
         "course": "Admitted with upper gastrointestinal bleeding and acute blood loss anemia. Hemoglobin was 7.9 g/dL, down from a baseline of 12.6 g/dL. Apixaban was held and she received 2 units of packed red blood cells. Upper endoscopy showed a clean-based gastric ulcer; ibuprofen was stopped and pantoprazole 40 mg twice daily was started. Hemoglobin stabilized at 9.4 g/dL. After 72 hours without further bleeding, and in discussion with gastroenterology, apixaban 5 mg twice daily was resumed before discharge because of her stroke risk.",
         "labs": {"Hemoglobin": [("2023-05-16 04:10", 7.9), ("2023-05-17 06:00", 8.8), ("2023-05-18 06:00", 9.1), ("2023-05-20 06:00", 9.4)],
                  "Urea Nitrogen": [("2023-05-16 04:10", 48), ("2023-05-20 06:00", 19)]},
         "dx": [("K25.4", "Chronic gastric ulcer with hemorrhage"), ("D62", "Acute posthemorrhagic anemia"),
                ("I48.91", "Atrial fibrillation"), ("Z79.01", "Long term use of anticoagulants")],
         "meds": [M("Pantoprazole", "40 mg", "PO", "twice daily", "new"), M("Apixaban", "5 mg", "PO", "twice daily", "continued", "resumed after 72 hours"),
                  M("Diltiazem extended-release", "180 mg", "PO", "daily"),
                  M("Levothyroxine", "75 mcg", "PO", "daily"), M("Losartan", "50 mg", "PO", "daily")],
         "stopped": [("Ibuprofen", "gastric ulcer; avoid NSAIDs")],
         "progress": [PROG("2023-05-18 11:00", 3,
                           "No further melena for 36 hours. Tolerating a regular diet.",
                           "Heart rate 88 and irregular. Abdomen benign.",
                           "1. Upper GI bleed from gastric ulcer: hemoglobin stable at 9.1 g/dL after 2 units; continue pantoprazole 40 mg twice daily and avoid NSAIDs.\n2. Atrial fibrillation: rate controlled on diltiazem; apixaban on hold, plan to resume in 24 hours if hemoglobin remains stable per gastroenterology.\n3. Acute blood loss anemia: transfuse for hemoglobin below 7 g/dL.",
                           labs=["Hemoglobin"])],
         "followup": "Gastroenterology in 8 weeks. Complete blood count in 1 week."},
        {"admit": "2024-04-09 07:00", "disch": "2024-04-12 11:30", "service": "Cardiology",
         "cc": "Fatigue and exercise intolerance with persistent atrial fibrillation.",
         "hpi": "She reports three months of fatigue and reduced exercise tolerance with persistent atrial fibrillation despite diltiazem. Admitted electively for rhythm control.",
         "vitals": "temperature 36.6 C, blood pressure 128/78, heart rate 96 and irregular, respiratory rate 16, oxygen saturation 98% on room air",
         "exam": "Irregularly irregular rhythm. Lungs clear. No edema.",
         "course": "Admitted for rhythm control of persistent atrial fibrillation. Transesophageal echocardiogram showed no left atrial appendage thrombus, followed by successful direct-current cardioversion to sinus rhythm. Amiodarone 200 mg daily was started to maintain sinus rhythm. Diltiazem was discontinued and metoprolol succinate 50 mg daily was started for rate control. Apixaban 5 mg twice daily was continued without interruption and must not be stopped for at least 4 weeks after cardioversion. Pantoprazole was reduced to 40 mg daily.",
         "labs": {"Hemoglobin": [("2024-04-09 07:30", 12.1)], "Potassium": [("2024-04-09 07:30", 4.3)]},
         "dx": [("I48.19", "Other persistent atrial fibrillation"), ("I10", "Essential hypertension"), ("Z79.01", "Long term use of anticoagulants")],
         "meds": [M("Amiodarone", "200 mg", "PO", "daily", "new"),
                  M("Metoprolol succinate", "50 mg", "PO", "daily", "new", "replaces diltiazem"),
                  M("Apixaban", "5 mg", "PO", "twice daily"),
                  M("Pantoprazole", "40 mg", "PO", "daily", "changed", "reduced from twice daily"),
                  M("Levothyroxine", "75 mcg", "PO", "daily"), M("Losartan", "50 mg", "PO", "daily")],
         "stopped": [("Diltiazem extended-release", "replaced by metoprolol succinate")],
         "followup": "Cardiology in 4 weeks with electrocardiogram, thyroid and liver tests on amiodarone."},
     ]},

    # ---- E: iron deficiency anemia with recovery ---------------------------
    {"n": 5, "sex": "F", "age": 46, "pmh_short": "uterine fibroids with heavy menstrual bleeding",
     "pmh": ["Uterine fibroids with heavy menstrual bleeding", "Iron deficiency anemia", "Migraine without aura"],
     "social": "Works as an accountant. Never smoker. Exercises twice weekly.",
     "base": {"Sodium": 139, "Potassium": 4.0, "Urea Nitrogen": 12, "Hemoglobin": 12.8, "White Blood Cells": 6.2, "Glucose": 92, "Platelet Count": 356},
     "admissions": [
        {"admit": "2023-03-02 10:30", "disch": "2023-03-04 15:00", "service": "Medicine",
         "cc": "Fatigue and shortness of breath on exertion.",
         "hpi": "She reports two months of progressive fatigue, shortness of breath climbing one flight of stairs, and heavy menstrual periods lasting 8 days.",
         "exam": "Conjunctival pallor. Soft systolic flow murmur. No lymphadenopathy.",
         "course": "Admitted with symptomatic iron deficiency anemia. Hemoglobin was 7.2 g/dL with a mean corpuscular volume of 71 fL and ferritin of 6 ng/mL. She received 1 unit of packed red blood cells and IV iron sucrose 200 mg on two consecutive days. Hemoglobin was 8.1 g/dL at discharge. Pelvic ultrasound confirmed multiple uterine fibroids, and gynecology planned an outpatient myomectomy.",
         "labs": {"Hemoglobin": [("2023-03-02 11:00", 7.2), ("2023-03-04 06:00", 8.1)], "Ferritin": [("2023-03-02 11:00", 6)]},
         "dx": [("D50.0", "Iron deficiency anemia secondary to chronic blood loss"), ("N92.0", "Excessive and frequent menstruation"),
                ("D25.1", "Intramural leiomyoma of uterus")],
         "inpatient": [M("Iron sucrose", "200 mg", "IV", "daily for 2 doses")],
         "meds": [M("Ferrous sulfate", "325 mg", "PO", "every other day", "new"),
                  M("Tranexamic acid", "1300 mg", "PO", "three times daily during menses", "new"),
                  M("Sumatriptan", "50 mg", "PO", "as needed for migraine")],
         "imaging": [IMG("Pelvic ultrasound, transabdominal and transvaginal", "2023-03-03 13:00", "Heavy menstrual bleeding and anemia.", "None.",
                         "The uterus is enlarged, measuring 12.4 cm, with three intramural fibroids, the largest 4.8 cm in the posterior body. The endometrium measures 9 mm. Both ovaries are normal. No free fluid.",
                         "Enlarged fibroid uterus with multiple intramural leiomyomas, the largest 4.8 cm.")],
         "followup": "Hematology clinic in 6 weeks. Gynecology for myomectomy planning."},
        {"admit": "2023-07-10 06:30", "disch": "2023-07-12 12:00", "service": "Gynecology",
         "cc": "Scheduled laparoscopic myomectomy.",
         "hpi": "She has symptomatic uterine fibroids and iron deficiency anemia and was admitted for an elective laparoscopic myomectomy after iron repletion.",
         "exam": "Well appearing. Abdomen soft and non-tender.",
         "course": "Underwent an uncomplicated laparoscopic myomectomy with removal of three intramural fibroids; estimated blood loss 150 mL. Preoperative hemoglobin was 11.8 g/dL and postoperative hemoglobin was 11.2 g/dL on the day of discharge. Pain was controlled with oral medications. Tolerating a regular diet and ambulating independently.",
         "labs": {"Hemoglobin": [("2023-07-10 06:45", 11.8), ("2023-07-12 06:00", 11.2)]},
         "dx": [("D25.1", "Intramural leiomyoma of uterus"), ("D50.0", "Iron deficiency anemia secondary to chronic blood loss")],
         "meds": [M("Acetaminophen", "1000 mg", "PO", "every 8 hours as needed", "new"),
                  M("Ibuprofen", "600 mg", "PO", "every 6 hours as needed for 5 days", "new"),
                  M("Ferrous sulfate", "325 mg", "PO", "every other day"), M("Sumatriptan", "50 mg", "PO", "as needed for migraine")],
         "stopped": [("Tranexamic acid", "no longer needed after myomectomy")],
         "followup": "Gynecology postoperative visit in 2 weeks."},
     ],
     "clinic": [PROG("2023-04-15 09:30", None,
                     "Outpatient hematology follow-up. Energy is improving and she can climb two flights of stairs. Menses remain heavy.",
                     "Mild pallor. Heart regular without murmur.",
                     "1. Iron deficiency anemia due to heavy menstrual bleeding: improving, hemoglobin 9.1 g/dL from 7.2 g/dL in March, ferritin 38 ng/mL; continue oral iron every other day.\n2. Uterine fibroids: myomectomy scheduled for July.",
                     labs=[("Hemoglobin", "2023-04-15 09:00", 9.1), ("Ferritin", "2023-04-15 09:00", 38)])]},

    # ---- F: hypertension, urgency then medication escalation --------------
    {"n": 6, "sex": "M", "age": 55, "pmh_short": "long-standing hypertension and obstructive sleep apnea",
     "pmh": ["Hypertension for 10 years", "Obstructive sleep apnea on CPAP", "Prediabetes"],
     "social": "Works as a bus driver. Never smoker. Drinks alcohol socially.",
     "base": {"Sodium": 140, "Potassium": 3.9, "Urea Nitrogen": 15, "Hemoglobin": 14.9, "White Blood Cells": 6.9, "Glucose": 108, "Platelet Count": 222},
     "admissions": [
        {"admit": "2023-06-19 22:10", "disch": "2023-06-21 10:00", "service": "Medicine",
         "cc": "Severe headache and very high blood pressure.",
         "hpi": "He reports an occipital headache and a home blood pressure of 220/120 after running out of amlodipine two weeks ago.",
         "vitals": "temperature 36.7 C, blood pressure 212/118, heart rate 84, respiratory rate 16, oxygen saturation 98% on room air",
         "exam": "Fundi without papilledema. Neurologic examination non-focal. No edema.",
         "course": "Admitted with hypertensive urgency without evidence of acute end-organ damage: troponin T was not elevated, creatinine was 1.0 mg/dL, urinalysis showed no protein, and head CT was normal. Blood pressure was lowered gradually with oral agents to 158/94 by discharge. Amlodipine 10 mg daily and lisinopril 20 mg daily were resumed and chlorthalidone 12.5 mg daily was added. Medication adherence and home blood pressure monitoring were reviewed.",
         "labs": {"Creatinine": [("2023-06-19 22:40", 1.0), ("2023-06-21 06:00", 1.0)], "Troponin T": [("2023-06-19 22:40", 0.01)]},
         "dx": [("I16.0", "Hypertensive urgency"), ("G47.33", "Obstructive sleep apnea"), ("R73.03", "Prediabetes")],
         "meds": [M("Chlorthalidone", "12.5 mg", "PO", "daily", "new"), M("Amlodipine", "10 mg", "PO", "daily", "continued", "resumed"),
                  M("Lisinopril", "20 mg", "PO", "daily")],
         "imaging": [IMG("CT head without contrast", "2023-06-19 23:30", "Severe headache with markedly elevated blood pressure.", "None.",
                         "No intracranial hemorrhage, mass effect, or acute territorial infarct. The ventricles and sulci are normal for age. No extra-axial collection.",
                         "No acute intracranial abnormality.")],
         "followup": "Primary care in 1 week with home blood pressure log and basic metabolic panel on chlorthalidone."},
        {"admit": "2024-05-02 14:30", "disch": "2024-05-04 11:00", "service": "Medicine",
         "cc": "Chest pressure.",
         "hpi": "He reports one week of intermittent exertional chest pressure. Blood pressure was 190/105 on arrival.",
         "vitals": "temperature 36.6 C, blood pressure 190/105, heart rate 78, respiratory rate 16, oxygen saturation 98% on room air",
         "exam": "Regular rhythm without murmur. Lungs clear. No edema.",
         "course": "Acute coronary syndrome was excluded with serial troponin T of 0.01 ng/mL and a normal electrocardiogram. Exercise stress echocardiogram showed no inducible ischemia. Chest pressure was attributed to uncontrolled hypertension. Carvedilol 12.5 mg twice daily was added and blood pressure was 136/84 at discharge.",
         "labs": {"Troponin T": [("2024-05-02 15:00", 0.01), ("2024-05-02 21:00", 0.01)], "Creatinine": [("2024-05-02 15:00", 1.1)]},
         "dx": [("I10", "Essential hypertension"), ("R07.89", "Other chest pain"), ("G47.33", "Obstructive sleep apnea")],
         "meds": [M("Carvedilol", "12.5 mg", "PO", "twice daily", "new"), M("Chlorthalidone", "12.5 mg", "PO", "daily"),
                  M("Amlodipine", "10 mg", "PO", "daily"), M("Lisinopril", "20 mg", "PO", "daily")],
         "followup": "Primary care in 2 weeks. Continue home blood pressure monitoring."},
     ]},

    # ---- G: HFpEF with permanent atrial fibrillation ------------------------
    {"n": 7, "sex": "F", "age": 79, "pmh_short": "heart failure with preserved ejection fraction and permanent atrial fibrillation",
     "pmh": ["Heart failure with preserved ejection fraction (LVEF 60%)", "Permanent atrial fibrillation on apixaban",
             "Obesity", "Obstructive sleep apnea", "Hypertension"],
     "social": "Retired seamstress. Never smoker. Lives with her son.",
     "base": {"Sodium": 138, "Potassium": 4.3, "Urea Nitrogen": 28, "Hemoglobin": 12.2, "White Blood Cells": 7.4, "Glucose": 121, "Platelet Count": 198},
     "admissions": [
        {"admit": "2023-10-05 12:00", "disch": "2023-10-10 13:00", "service": "Cardiology",
         "cc": "Leg swelling and breathlessness.",
         "hpi": "She reports one week of worsening leg swelling, a 5 kg weight gain, and shortness of breath at rest.",
         "exam": "Jugular venous pressure 13 cm. Decreased breath sounds at both bases. 3+ bilateral leg edema.",
         "course": "Admitted with acute on chronic heart failure with preserved ejection fraction and volume overload. Diuresed with IV bumetanide with net negative 6.3 liters. Atrial fibrillation remained rate controlled on metoprolol succinate 100 mg daily, and apixaban 5 mg twice daily was continued. Creatinine was 1.1 mg/dL on admission and 1.2 mg/dL at discharge. Discharged on bumetanide 1 mg twice daily, which replaces hydrochlorothiazide.",
         "labs": {"Creatinine": [("2023-10-05 12:30", 1.1), ("2023-10-10 06:00", 1.2)], "NT-proBNP": [("2023-10-05 12:30", 3120)]},
         "dx": [("I50.33", "Acute on chronic diastolic heart failure"), ("I48.21", "Permanent atrial fibrillation"),
                ("I11.0", "Hypertensive heart disease with heart failure"), ("E66.9", "Obesity"), ("G47.33", "Obstructive sleep apnea")],
         "inpatient": [M("Bumetanide", "2 mg", "IV", "twice daily")],
         "meds": [M("Bumetanide", "1 mg", "PO", "twice daily", "new", "replaces hydrochlorothiazide"),
                  M("Metoprolol succinate", "100 mg", "PO", "daily"), M("Apixaban", "5 mg", "PO", "twice daily")],
         "stopped": [("Hydrochlorothiazide", "replaced by bumetanide")],
         "imaging": [IMG("Chest radiograph, portable AP", "2023-10-05 13:10", "Dyspnea and edema.", "None.",
                         "Moderate bilateral pleural effusions, larger on the right, with adjacent atelectasis. Mild interstitial edema. Enlarged cardiac silhouette.",
                         "Moderate bilateral pleural effusions and mild pulmonary edema.")],
         "followup": "Heart failure clinic in 1 week. Daily weights."},
        {"admit": "2024-06-11 09:20", "disch": "2024-06-15 12:00", "service": "Cardiology",
         "cc": "Worsening shortness of breath.",
         "hpi": "She reports one week of worsening dyspnea and orthopnea after missing several bumetanide doses.",
         "exam": "Jugular venous pressure 12 cm. Bibasilar crackles. 2+ leg edema.",
         "course": "Recurrent exacerbation of heart failure with preserved ejection fraction after missed diuretic doses. Diuresed with IV bumetanide. Empagliflozin 10 mg daily was started for heart failure with preserved ejection fraction. Creatinine was 1.2 mg/dL on admission and 1.3 mg/dL at discharge. A pill organizer and home nursing visits were arranged.",
         "labs": {"Creatinine": [("2024-06-11 09:50", 1.2), ("2024-06-15 06:00", 1.3)], "NT-proBNP": [("2024-06-11 09:50", 2780)]},
         "dx": [("I50.33", "Acute on chronic diastolic heart failure"), ("I48.21", "Permanent atrial fibrillation"),
                ("I11.0", "Hypertensive heart disease with heart failure")],
         "meds": [M("Empagliflozin", "10 mg", "PO", "daily", "new"), M("Bumetanide", "1 mg", "PO", "twice daily"),
                  M("Metoprolol succinate", "100 mg", "PO", "daily"), M("Apixaban", "5 mg", "PO", "twice daily")],
         "followup": "Heart failure clinic in 1 week. Home nursing for medication review."},
     ]},

    # ---- H: sepsis from pneumonia, then aspiration pneumonitis -------------
    {"n": 8, "sex": "M", "age": 83, "pmh_short": "Parkinson disease with mild dysphagia",
     "pmh": ["Parkinson disease with mild dysphagia", "Benign prostatic hyperplasia", "Hypertension"],
     "social": "Retired electrician. Lives in an assisted living facility. Walks with a cane.",
     "base": {"Sodium": 141, "Potassium": 4.2, "Urea Nitrogen": 22, "Hemoglobin": 13.0, "White Blood Cells": 7.0, "Glucose": 102, "Platelet Count": 205},
     "admissions": [
        {"admit": "2024-01-15 18:00", "disch": "2024-01-21 12:30", "service": "Medicine",
         "cc": "Confusion, fever, and cough.",
         "hpi": "Staff at his facility reported two days of cough, fever to 39.2 C, and new confusion. Blood pressure was 88/52 on arrival.",
         "vitals": "temperature 39.0 C, blood pressure 88/52, heart rate 118, respiratory rate 26, oxygen saturation 90% on room air",
         "exam": "Drowsy but arousable, oriented to self only. Crackles at the left lung base. Resting tremor.",
         "course": "Admitted with sepsis due to left lower lobe pneumonia, possibly related to aspiration given his dysphagia. Received 30 mL/kg of IV crystalloid and piperacillin-tazobactam; blood pressure responded without vasopressors. Lactate cleared from 3.4 to 1.2 mmol/L. Blood cultures were negative. Creatinine improved from 1.6 to 1.0 mg/dL with fluids. Speech-language pathology recommended a mechanical soft diet. Antibiotics were narrowed to amoxicillin-clavulanate to complete 7 days. Mental status returned to baseline by day 3. Lisinopril was held for low blood pressure.",
         "labs": {"Lactate": [("2024-01-15 18:30", 3.4), ("2024-01-16 06:00", 1.8), ("2024-01-17 06:00", 1.2)],
                  "White Blood Cells": [("2024-01-15 18:30", 18.9), ("2024-01-21 06:00", 8.7)],
                  "Creatinine": [("2024-01-15 18:30", 1.6), ("2024-01-21 06:00", 1.0)]},
         "dx": [("A41.9", "Sepsis, unspecified organism"), ("J18.1", "Lobar pneumonia, left lower lobe"),
                ("N17.9", "Acute kidney injury"), ("G20", "Parkinson disease"), ("R13.10", "Dysphagia")],
         "inpatient": [M("Piperacillin-tazobactam", "3.375 g", "IV", "every 8 hours")],
         "meds": [M("Amoxicillin-clavulanate", "875-125 mg", "PO", "twice daily for 2 more days", "new"),
                  M("Carbidopa-levodopa", "25-100 mg", "PO", "three times daily"), M("Tamsulosin", "0.4 mg", "PO", "nightly")],
         "stopped": [("Lisinopril", "held for low blood pressure; reassess at follow-up")],
         "imaging": [IMG("Chest radiograph, portable AP", "2024-01-15 19:00", "Fever, cough, and hypotension.", "None.",
                         "Consolidation in the left lower lobe with a small left parapneumonic pleural effusion. The right lung is clear. Normal heart size.",
                         "Left lower lobe pneumonia with a small parapneumonic effusion.")],
         "followup": "Facility physician in 3 days. Speech-language pathology follow-up."},
        {"admit": "2024-03-02 10:10", "disch": "2024-03-05 11:45", "service": "Medicine",
         "cc": "Coughing and choking after meals.",
         "hpi": "Facility staff noted coughing and choking episodes during meals for one week and a low-grade temperature of 37.8 C.",
         "exam": "Alert, at baseline mental status. Coarse breath sounds at the right base.",
         "course": "Admitted with mild aspiration pneumonitis. Lactate was normal and he remained hemodynamically stable. CT chest showed an improving left lower lobe opacity compared with January and a new mild dependent right lower lobe ground-glass opacity. Managed with supportive care without antibiotics. A repeat swallow study showed silent aspiration with thin liquids, and the diet was changed to nectar-thick liquids.",
         "labs": {"White Blood Cells": [("2024-03-02 10:40", 9.6)], "Lactate": [("2024-03-02 10:40", 1.1)]},
         "dx": [("J69.0", "Pneumonitis due to inhalation of food and vomit"), ("G20", "Parkinson disease"), ("R13.10", "Dysphagia")],
         "meds": [M("Carbidopa-levodopa", "25-100 mg", "PO", "three times daily"), M("Tamsulosin", "0.4 mg", "PO", "nightly")],
         "imaging": [IMG("CT chest without contrast", "2024-03-02 13:30", "Cough and choking after meals; prior left lower lobe pneumonia.",
                         "Chest radiograph from 2024-01-15.",
                         "Residual patchy opacity in the left lower lobe, substantially decreased compared with the January radiograph. New mild ground-glass opacity in the dependent right lower lobe. Small hiatal hernia. No pleural effusion or lymphadenopathy.",
                         "Improving left lower lobe pneumonia. New dependent right lower lobe ground-glass opacity, compatible with aspiration.")],
         "followup": "Speech-language pathology at the facility within 1 week. Nectar-thick liquids."},
     ]},

    # ---- I: type 2 diabetes, hyperglycemia then hypoglycemia --------------
    {"n": 9, "sex": "M", "age": 49, "pmh_short": "type 2 diabetes mellitus and obesity",
     "pmh": ["Type 2 diabetes mellitus", "Obesity (BMI 36)", "Nonalcoholic fatty liver disease"],
     "social": "Works in retail management. Never smoker. Rare alcohol use.",
     "base": {"Sodium": 136, "Potassium": 4.1, "Urea Nitrogen": 14, "Hemoglobin": 15.1, "White Blood Cells": 8.1, "Glucose": 190, "Platelet Count": 262},
     "admissions": [
        {"admit": "2023-04-11 13:00", "disch": "2023-04-13 12:00", "service": "Medicine",
         "cc": "Blurred vision, thirst, and frequent urination.",
         "hpi": "He reports three weeks of blurred vision, thirst, and frequent urination with a 5 kg unintentional weight loss. He takes metformin 1000 mg twice daily.",
         "exam": "Dry mucous membranes. Acanthosis nigricans. No focal deficits.",
         "course": "Admitted with severe hyperglycemia without ketoacidosis: glucose 540 mg/dL, bicarbonate 22 mEq/L, and negative serum ketones. Hemoglobin A1c was 11.4%. Started basal-bolus insulin with insulin glargine 20 units nightly and insulin lispro 6 units with meals; metformin 1000 mg twice daily was continued. Diabetes education was provided.",
         "labs": {"Glucose": [("2023-04-11 13:30", 540), ("2023-04-12 06:00", 248), ("2023-04-13 06:00", 176)],
                  "Hemoglobin A1c": [("2023-04-12 06:00", 11.4)], "Bicarbonate": [("2023-04-11 13:30", 22)]},
         "dx": [("E11.65", "Type 2 diabetes mellitus with hyperglycemia"), ("E66.9", "Obesity"), ("K76.0", "Fatty liver disease")],
         "meds": [M("Insulin glargine", "20 units", "SC", "nightly", "new"), M("Insulin lispro", "6 units", "SC", "with meals", "new"),
                  M("Metformin", "1000 mg", "PO", "twice daily")],
         "followup": "Endocrinology in 2 weeks. Check fingerstick glucose four times daily."},
        {"admit": "2024-02-06 11:20", "disch": "2024-02-08 10:30", "service": "Medicine",
         "cc": "Found confused and sweaty at work.",
         "hpi": "Coworkers found him confused and diaphoretic; fingerstick glucose was 48 mg/dL. He had taken insulin lispro before skipping lunch.",
         "exam": "Alert and oriented after dextrose. Diaphoresis resolved. No focal deficits.",
         "course": "Admitted after symptomatic hypoglycemia treated with IV dextrose. Hemoglobin A1c was 8.0%, improved from 11.4%. Insulin glargine was reduced from 20 to 14 units nightly, mealtime insulin lispro was stopped, and semaglutide 0.25 mg weekly was started with a plan to titrate.",
         "labs": {"Glucose": [("2024-02-06 11:40", 48), ("2024-02-06 16:00", 132), ("2024-02-08 06:00", 118)],
                  "Hemoglobin A1c": [("2024-02-07 06:00", 8.0)]},
         "dx": [("E11.649", "Type 2 diabetes mellitus with hypoglycemia without coma"), ("E66.9", "Obesity")],
         "meds": [M("Insulin glargine", "14 units", "SC", "nightly", "changed", "reduced from 20 units"),
                  M("Semaglutide", "0.25 mg", "SC", "weekly", "new"), M("Metformin", "1000 mg", "PO", "twice daily")],
         "stopped": [("Insulin lispro", "stopped after hypoglycemia")],
         "followup": "Endocrinology in 2 weeks for semaglutide titration."},
     ]},

    # ---- J: severe COPD, recurrent exacerbations, home oxygen -------------
    {"n": 10, "sex": "F", "age": 70, "pmh_short": "severe COPD",
     "pmh": ["Severe chronic obstructive pulmonary disease (FEV1 38% predicted)", "Former smoker, 50 pack-years",
             "Osteoporosis", "Anxiety"],
     "social": "Retired bank teller. Quit smoking in 2018. Lives alone with support from a neighbor.",
     "base": {"Sodium": 139, "Potassium": 4.2, "Urea Nitrogen": 17, "Hemoglobin": 14.1, "White Blood Cells": 9.0, "Glucose": 112, "Platelet Count": 287},
     "admissions": [
        {"admit": "2022-12-01 08:15", "disch": "2022-12-05 12:00", "service": "Medicine",
         "cc": "Worsening shortness of breath and sputum.",
         "hpi": "She reports five days of worsening dyspnea, increased purulent sputum, and wheezing.",
         "vitals": "temperature 37.4 C, blood pressure 142/84, heart rate 104, respiratory rate 26, oxygen saturation 88% on room air",
         "exam": "Tripod positioning, diffuse wheezes, poor air movement.",
         "course": "Admitted with a severe COPD exacerbation and acute hypercapnic respiratory failure; arterial blood gas showed pH 7.31 with PaCO2 58 mmHg. Treated with bilevel positive airway pressure overnight, nebulized bronchodilators, prednisone 40 mg daily, and doxycycline. Weaned to room air with resting saturation 92% at discharge.",
         "labs": {"Bicarbonate": [("2022-12-01 08:45", 30), ("2022-12-05 06:00", 31)],
                  "White Blood Cells": [("2022-12-01 08:45", 13.1), ("2022-12-05 06:00", 10.2)]},
         "dx": [("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"), ("J96.02", "Acute respiratory failure with hypercapnia"),
                ("M81.0", "Osteoporosis"), ("F41.9", "Anxiety disorder")],
         "meds": [M("Prednisone", "40 mg", "PO", "daily for 2 more days", "new"), M("Doxycycline", "100 mg", "PO", "twice daily for 3 more days", "new"),
                  M("Budesonide-formoterol", "160-4.5 mcg, 2 puffs", "inhaled", "twice daily"), M("Tiotropium", "18 mcg", "inhaled", "daily"),
                  M("Albuterol inhaler", "2 puffs", "inhaled", "every 4 hours as needed")],
         "followup": "Pulmonology in 2 weeks."},
        {"admit": "2023-04-18 14:40", "disch": "2023-04-22 11:15", "service": "Medicine",
         "cc": "Shortness of breath after a cold.",
         "hpi": "She reports one week of cough and worsening dyspnea following a viral illness.",
         "exam": "Diffuse wheezes and prolonged expiration. No crackles.",
         "course": "Second COPD exacerbation in five months. Treated with prednisone, nebulized bronchodilators, and azithromycin. Required 2 liters of oxygen during the first two days. Given frequent exacerbations, azithromycin 250 mg three times weekly was started for exacerbation prevention.",
         "labs": {"Bicarbonate": [("2023-04-18 15:10", 31)], "White Blood Cells": [("2023-04-18 15:10", 11.6)]},
         "dx": [("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"), ("F41.9", "Anxiety disorder")],
         "meds": [M("Azithromycin", "250 mg", "PO", "three times weekly", "new", "exacerbation prevention"),
                  M("Prednisone", "40 mg", "PO", "daily for 2 more days", "new"),
                  M("Budesonide-formoterol", "160-4.5 mcg, 2 puffs", "inhaled", "twice daily"), M("Tiotropium", "18 mcg", "inhaled", "daily"),
                  M("Albuterol inhaler", "2 puffs", "inhaled", "every 4 hours as needed")],
         "followup": "Pulmonology in 4 weeks with electrocardiogram for QT on azithromycin."},
        {"admit": "2023-11-08 09:30", "disch": "2023-11-13 13:00", "service": "Medicine",
         "cc": "Breathlessness and low oxygen at home.",
         "hpi": "Her home pulse oximeter read 84% and she reports breathlessness walking across a room.",
         "vitals": "temperature 36.9 C, blood pressure 136/82, heart rate 98, respiratory rate 24, oxygen saturation 85% on room air",
         "exam": "Pursed-lip breathing, distant breath sounds, scattered wheezes.",
         "course": "Third COPD exacerbation in one year. Chest radiograph showed hyperinflation without pneumonia. Treated with prednisone and bronchodilators. Resting oxygen saturation remained 86% on room air at discharge, so home oxygen at 2 liters per minute by nasal cannula continuously was started. Roflumilast 500 mcg daily was added.",
         "labs": {"Bicarbonate": [("2023-11-08 10:00", 33), ("2023-11-13 06:00", 32)], "White Blood Cells": [("2023-11-08 10:00", 10.4)]},
         "dx": [("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"), ("J96.11", "Chronic respiratory failure with hypoxia")],
         "meds": [M("Home oxygen", "2 L/min", "nasal cannula", "continuous", "new"), M("Roflumilast", "500 mcg", "PO", "daily", "new"),
                  M("Azithromycin", "250 mg", "PO", "three times weekly"),
                  M("Budesonide-formoterol", "160-4.5 mcg, 2 puffs", "inhaled", "twice daily"), M("Tiotropium", "18 mcg", "inhaled", "daily"),
                  M("Albuterol inhaler", "2 puffs", "inhaled", "every 4 hours as needed")],
         "imaging": [IMG("Chest radiograph, PA and lateral", "2023-11-08 11:00", "COPD exacerbation, hypoxemia.", "Chest radiograph from 2022-12-01.",
                         "Marked hyperinflation with flattened hemidiaphragms and increased retrosternal air space. No focal consolidation, effusion, or pneumothorax. Normal heart size.",
                         "Severe emphysematous hyperinflation without pneumonia.")],
         "followup": "Pulmonology in 2 weeks. Home oxygen supplier visit arranged."},
     ]},

    # ---- K: AKI on CKD from volume depletion, recovery ---------------------
    {"n": 11, "sex": "M", "age": 76, "pmh_short": "chronic kidney disease stage 3a, hypertension, and gout",
     "pmh": ["Chronic kidney disease stage 3a (baseline creatinine 1.3 mg/dL)", "Hypertension", "Gout"],
     "social": "Retired farmer. Former smoker. Lives with his wife.",
     "base": {"Sodium": 139, "Potassium": 4.4, "Urea Nitrogen": 24, "Hemoglobin": 12.9, "White Blood Cells": 7.3, "Glucose": 110, "Platelet Count": 190},
     "admissions": [
        {"admit": "2024-07-08 16:20", "disch": "2024-07-12 11:00", "service": "Medicine",
         "cc": "Diarrhea and dizziness.",
         "hpi": "He reports four days of watery diarrhea, poor oral intake, and dizziness on standing while continuing lisinopril and hydrochlorothiazide.",
         "vitals": "temperature 37.1 C, blood pressure 96/58, heart rate 108, respiratory rate 18, oxygen saturation 97% on room air",
         "exam": "Dry mucous membranes, reduced skin turgor. Abdomen soft with active bowel sounds.",
         "course": "Admitted with acute kidney injury on chronic kidney disease due to volume depletion from viral gastroenteritis. Creatinine was 3.0 mg/dL on admission, from a baseline of 1.3 mg/dL, and improved to 2.2 mg/dL on day 3 and 1.6 mg/dL at discharge with IV fluids. Lisinopril and hydrochlorothiazide were held. Amlodipine 5 mg daily was started for blood pressure. Allopurinol was reduced to 100 mg daily for kidney function.",
         "labs": {"Creatinine": [("2024-07-08 16:50", 3.0), ("2024-07-09 06:00", 2.6), ("2024-07-10 06:00", 2.2), ("2024-07-12 06:00", 1.6)],
                  "Urea Nitrogen": [("2024-07-08 16:50", 64), ("2024-07-12 06:00", 28)],
                  "Sodium": [("2024-07-08 16:50", 133), ("2024-07-12 06:00", 139)]},
         "dx": [("N17.9", "Acute kidney injury"), ("N18.31", "Chronic kidney disease, stage 3a"), ("E86.0", "Dehydration"),
                ("A08.4", "Viral intestinal infection"), ("I12.9", "Hypertensive chronic kidney disease")],
         "meds": [M("Amlodipine", "5 mg", "PO", "daily", "new"), M("Allopurinol", "100 mg", "PO", "daily", "changed", "reduced from 300 mg daily")],
         "stopped": [("Lisinopril", "held for acute kidney injury"), ("Hydrochlorothiazide", "held for acute kidney injury")],
         "progress": [PROG("2024-07-09 10:00", 2,
                           "Diarrhea slowing. Still lightheaded on standing.",
                           "Orthostatic blood pressure drop of 22 mmHg. Mucous membranes less dry.",
                           "1. Acute kidney injury on chronic kidney disease stage 3a, prerenal from volume depletion: creatinine 2.6 mg/dL from 3.0 mg/dL; continue IV fluids and hold lisinopril and hydrochlorothiazide.\n2. Viral gastroenteritis: supportive care.\n3. Gout: reduce allopurinol dose for kidney function.",
                           labs=["Creatinine"])],
         "followup": "Primary care in 1 week with basic metabolic panel. Do not restart lisinopril or hydrochlorothiazide until reviewed."},
        {"admit": "2024-09-22 07:45", "disch": "2024-09-24 12:15", "service": "Medicine",
         "cc": "Passed out after standing up.",
         "hpi": "He fainted briefly after getting out of bed and recovered within seconds. No chest pain or palpitations.",
         "exam": "Orthostatic blood pressure drop of 28 mmHg. Regular rhythm. No focal deficits.",
         "course": "Admitted with syncope due to orthostatic hypotension. Telemetry showed no arrhythmia and echocardiogram was not required after a normal electrocardiogram. Creatinine was 1.4 mg/dL, near his baseline. Amlodipine was reduced from 5 mg to 2.5 mg daily. Lisinopril and hydrochlorothiazide remain stopped.",
         "labs": {"Creatinine": [("2024-09-22 08:15", 1.4), ("2024-09-24 06:00", 1.4)]},
         "dx": [("R55", "Syncope and collapse"), ("I95.1", "Orthostatic hypotension"), ("N18.31", "Chronic kidney disease, stage 3a")],
         "meds": [M("Amlodipine", "2.5 mg", "PO", "daily", "changed", "reduced from 5 mg daily"), M("Allopurinol", "100 mg", "PO", "daily")],
         "followup": "Primary care in 2 weeks. Rise slowly and increase fluid intake."},
     ]},

    # ---- L: NSTEMI with new cardiomyopathy, then CABG -----------------------
    {"n": 12, "sex": "M", "age": 61, "pmh_short": "hypertension, hyperlipidemia, type 2 diabetes, and active smoking",
     "pmh": ["Hypertension", "Hyperlipidemia", "Type 2 diabetes mellitus", "Current smoker, 30 pack-years"],
     "social": "Works as a truck dispatcher. Smokes one pack per day. Drinks two beers on weekends.",
     "base": {"Sodium": 138, "Potassium": 4.2, "Urea Nitrogen": 18, "Hemoglobin": 14.4, "White Blood Cells": 8.8, "Glucose": 156, "Platelet Count": 241},
     "admissions": [
        {"admit": "2024-02-11 05:20", "disch": "2024-02-16 12:00", "service": "Cardiology",
         "cc": "Chest pain.",
         "hpi": "He reports two hours of substernal chest pressure radiating to the left arm with diaphoresis that woke him from sleep.",
         "exam": "Diaphoretic on arrival. Regular rhythm. Faint bibasilar crackles.",
         "course": "Admitted with non-ST-elevation myocardial infarction. Troponin T rose from 0.45 to a peak of 1.12 ng/mL. Coronary angiography showed severe three-vessel coronary artery disease, and cardiac surgery recommended coronary artery bypass grafting. Echocardiogram showed a new ischemic cardiomyopathy with left ventricular ejection fraction 35%. Started aspirin, high-intensity atorvastatin, metoprolol succinate, and low-dose lisinopril. Smoking cessation counseling provided and nicotine patch started.",
         "labs": {"Troponin T": [("2024-02-11 05:50", 0.45), ("2024-02-11 12:00", 1.12), ("2024-02-12 06:00", 0.86)],
                  "Creatinine": [("2024-02-11 05:50", 1.0)], "Hemoglobin A1c": [("2024-02-12 06:00", 7.6)]},
         "dx": [("I21.4", "Non-ST elevation myocardial infarction"), ("I25.10", "Coronary artery disease"), ("I25.5", "Ischemic cardiomyopathy"),
                ("E11.9", "Type 2 diabetes mellitus"), ("F17.210", "Nicotine dependence, cigarettes")],
         "inpatient": [M("Heparin", "weight-based", "IV", "continuous infusion")],
         "meds": [M("Aspirin", "81 mg", "PO", "daily", "new"), M("Atorvastatin", "80 mg", "PO", "nightly", "new"),
                  M("Metoprolol succinate", "25 mg", "PO", "daily", "new"), M("Lisinopril", "2.5 mg", "PO", "daily", "new"),
                  M("Nicotine patch", "21 mg", "transdermal", "daily", "new"), M("Metformin", "500 mg", "PO", "twice daily")],
         "imaging": [IMG("Transthoracic echocardiogram", "2024-02-13 09:30", "Non-ST-elevation myocardial infarction.", "None.",
                         "Mildly dilated left ventricle with hypokinesis of the inferior and lateral walls. Left ventricular ejection fraction is estimated at 35%. Mild mitral regurgitation. Normal right ventricular function.",
                         "New ischemic cardiomyopathy with left ventricular ejection fraction 35% and inferolateral hypokinesis.")],
         "followup": "Cardiac surgery clinic in 2 weeks for bypass surgery planning. Cardiology in 4 weeks."},
        {"admit": "2024-04-02 06:00", "disch": "2024-04-09 11:00", "service": "Cardiac Surgery",
         "cc": "Elective coronary artery bypass grafting.",
         "hpi": "He has three-vessel coronary artery disease after a non-ST-elevation myocardial infarction in February and was admitted for planned surgery.",
         "exam": "Well appearing. Regular rhythm. Lungs clear.",
         "course": "Underwent coronary artery bypass grafting times three (left internal mammary artery to the LAD, saphenous vein grafts to the obtuse marginal and posterior descending arteries). Postoperative atrial fibrillation on day 2 was treated with amiodarone with conversion to sinus rhythm. Anticoagulation was deferred given early postoperative bleeding risk and the short episode. Metoprolol succinate was increased to 50 mg daily. Postoperative hemoglobin nadir was 9.8 g/dL.",
         "labs": {"Hemoglobin": [("2024-04-03 06:00", 9.8), ("2024-04-09 06:00", 10.4)],
                  "Creatinine": [("2024-04-02 06:30", 1.0), ("2024-04-09 06:00", 1.1)]},
         "dx": [("I25.10", "Coronary artery disease"), ("Z95.1", "Presence of aortocoronary bypass graft"),
                ("I48.91", "Postoperative atrial fibrillation"), ("E11.9", "Type 2 diabetes mellitus")],
         "meds": [M("Amiodarone", "200 mg", "PO", "daily for 4 weeks", "new"), M("Acetaminophen", "1000 mg", "PO", "every 8 hours as needed", "new"),
                  M("Metoprolol succinate", "50 mg", "PO", "daily", "changed", "increased from 25 mg daily"),
                  M("Aspirin", "81 mg", "PO", "daily"), M("Atorvastatin", "80 mg", "PO", "nightly"),
                  M("Lisinopril", "2.5 mg", "PO", "daily"), M("Metformin", "500 mg", "PO", "twice daily")],
         "imaging": [IMG("Chest radiograph, portable AP", "2024-04-03 07:10", "Postoperative day 1 after coronary artery bypass grafting.", "Chest radiograph from 2024-02-11.",
                         "Postoperative changes of median sternotomy with intact sternal wires. Small left pleural effusion with left basilar atelectasis. No pneumothorax. Mediastinal contour within expected postoperative limits.",
                         "Expected postoperative appearance after bypass surgery with a small left pleural effusion and basilar atelectasis.")],
         "followup": "Cardiac surgery clinic in 2 weeks. Cardiac rehabilitation referral."},
     ]},

    # ---- M: CKD stage 4 with hyperkalemia and anemia of CKD ----------------
    {"n": 13, "sex": "F", "age": 68, "pmh_short": "chronic kidney disease stage 4 and hypertension",
     "pmh": ["Chronic kidney disease stage 4 due to hypertensive nephrosclerosis", "Anemia of chronic kidney disease",
             "Hypertension", "Gout"],
     "social": "Retired nurse aide. Never smoker. Lives with her sister.",
     "base": {"Sodium": 138, "Potassium": 4.9, "Urea Nitrogen": 52, "Hemoglobin": 9.2, "White Blood Cells": 6.4, "Glucose": 97, "Platelet Count": 176},
     "admissions": [
        {"admit": "2023-08-20 19:30", "disch": "2023-08-23 12:00", "service": "Medicine",
         "cc": "Muscle weakness and an abnormal potassium result.",
         "hpi": "She reports two days of leg weakness. Routine labs showed potassium 6.2 mEq/L.",
         "exam": "Mild proximal leg weakness. Trace ankle edema. Pale conjunctivae.",
         "course": "Admitted with hyperkalemia in chronic kidney disease stage 4 while taking losartan. Electrocardiogram showed peaked T waves. Treated with IV calcium gluconate, insulin with dextrose, and sodium zirconium cyclosilicate. Potassium improved from 6.2 to 4.7 mEq/L. Losartan was stopped, and sodium zirconium cyclosilicate 10 g daily and furosemide 20 mg daily were started. Hemoglobin was 8.6 g/dL from anemia of chronic kidney disease; epoetin alfa weekly will start at nephrology clinic.",
         "labs": {"Potassium": [("2023-08-20 20:00", 6.2), ("2023-08-21 06:00", 5.3), ("2023-08-23 06:00", 4.7)],
                  "Hemoglobin": [("2023-08-20 20:00", 8.6), ("2023-08-23 06:00", 8.4)],
                  "Creatinine": [("2023-08-20 20:00", 3.2), ("2023-08-23 06:00", 3.1)]},
         "dx": [("E87.5", "Hyperkalemia"), ("N18.4", "Chronic kidney disease, stage 4"), ("D63.1", "Anemia in chronic kidney disease"),
                ("I12.9", "Hypertensive chronic kidney disease")],
         "meds": [M("Sodium zirconium cyclosilicate", "10 g", "PO", "daily", "new"), M("Furosemide", "20 mg", "PO", "daily", "new"),
                  M("Epoetin alfa", "4000 units", "SC", "weekly", "new", "to start at nephrology clinic"),
                  M("Amlodipine", "10 mg", "PO", "daily"), M("Allopurinol", "100 mg", "PO", "daily")],
         "stopped": [("Losartan", "stopped for hyperkalemia")],
         "progress": [PROG("2023-08-21 09:45", 2,
                           "Weakness improving. No palpitations.",
                           "Strength improved. Trace ankle edema.",
                           "1. Hyperkalemia: potassium 5.3 mEq/L from 6.2 mEq/L after temporizing therapy; continue sodium zirconium cyclosilicate; losartan stopped.\n2. Chronic kidney disease stage 4: creatinine at baseline; low-potassium diet teaching.\n3. Anemia of chronic kidney disease: hemoglobin 8.6 g/dL; arrange epoetin alfa with nephrology.",
                           labs=["Potassium"])],
         "followup": "Nephrology in 1 week with potassium check. Low-potassium diet."},
        {"admit": "2024-01-29 06:30", "disch": "2024-02-01 11:00", "service": "Vascular Surgery",
         "cc": "Planned dialysis access surgery.",
         "hpi": "She has progressive chronic kidney disease stage 4 and was admitted for creation of a left radiocephalic arteriovenous fistula in preparation for future dialysis.",
         "exam": "Well appearing. Good radial pulse on the left. Trace edema.",
         "course": "Underwent uncomplicated creation of a left radiocephalic arteriovenous fistula with a palpable thrill postoperatively. Hemoglobin was 9.8 g/dL on epoetin alfa, improved from 8.6 g/dL in August. Creatinine was 3.4 mg/dL. Potassium was 4.9 mEq/L on sodium zirconium cyclosilicate.",
         "labs": {"Hemoglobin": [("2024-01-29 07:00", 9.8)], "Creatinine": [("2024-01-29 07:00", 3.4), ("2024-02-01 06:00", 3.4)],
                  "Potassium": [("2024-01-29 07:00", 4.9)]},
         "dx": [("N18.4", "Chronic kidney disease, stage 4"), ("D63.1", "Anemia in chronic kidney disease"),
                ("I12.9", "Hypertensive chronic kidney disease")],
         "meds": [M("Epoetin alfa", "4000 units", "SC", "weekly"), M("Sodium zirconium cyclosilicate", "10 g", "PO", "daily"),
                  M("Furosemide", "20 mg", "PO", "daily"), M("Amlodipine", "10 mg", "PO", "daily"), M("Allopurinol", "100 mg", "PO", "daily")],
         "followup": "Vascular surgery in 6 weeks to assess fistula maturation. Avoid blood pressure checks and blood draws in the left arm."},
     ]},

    # ---- N: decompensated cirrhosis, ascites -> SBP -> encephalopathy -------
    {"n": 14, "sex": "M", "age": 58, "pmh_short": "cirrhosis with portal hypertension and ascites",
     "pmh": ["Cirrhosis due to metabolic dysfunction-associated steatohepatitis", "Portal hypertension with ascites",
             "Esophageal varices, grade 1, on surveillance", "Type 2 diabetes mellitus"],
     "social": "Works part time as a warehouse supervisor. No alcohol for six years. Never smoker.",
     "base": {"Sodium": 132, "Potassium": 4.1, "Urea Nitrogen": 18, "Hemoglobin": 11.4, "White Blood Cells": 4.2, "Glucose": 142, "Platelet Count": 88},
     "admissions": [
        {"admit": "2023-02-14 17:40", "disch": "2023-02-20 11:30", "service": "Hepatology", "atype": "Emergency",
         "cc": "Abdominal distension and weight gain.",
         "hpi": "He reports four weeks of progressive abdominal distension, early satiety, and a 7 kg weight gain. He has never had a paracentesis and was not previously taking a diuretic.",
         "vitals": "temperature 36.6 C, blood pressure 106/64, heart rate 88, respiratory rate 18, oxygen saturation 97% on room air",
         "exam": "Scleral icterus. Tense ascites with a fluid wave. Spider angiomata on the chest. 1+ lower extremity edema. No asterixis.",
         "course": "Admitted with newly symptomatic ascites from cirrhosis. Diagnostic and therapeutic paracentesis removed 6.5 liters; the serum-ascites albumin gradient was 1.6 g/dL and the ascitic fluid cell count showed 85 neutrophils per microliter, so there was no evidence of infection on this admission. Spironolactone 100 mg daily and furosemide 40 mg daily were started as his first diuretic regimen. Sodium was 131 mEq/L on admission and 132 mEq/L at discharge. Creatinine remained 0.9 mg/dL throughout. Lactulose was started for primary prevention of hepatic encephalopathy and titrated to three soft stools daily.",
         "labs": {"Sodium": [("2023-02-14 18:10", 131), ("2023-02-20 06:00", 132)],
                  "Creatinine": [("2023-02-14 18:10", 0.9), ("2023-02-20 06:00", 0.9)],
                  "Albumin": [("2023-02-14 18:10", 2.8)], "Total Bilirubin": [("2023-02-14 18:10", 2.4)],
                  "INR": [("2023-02-14 18:10", 1.5)], "Platelet Count": [("2023-02-14 18:10", 88)]},
         "dx": [("K70.31", "Alcoholic cirrhosis of liver with ascites"), ("R18.8", "Other ascites"),
                ("K76.6", "Portal hypertension"), ("E11.9", "Type 2 diabetes mellitus without complications")],
         "inpatient": [M("Albumin 25%", "50 g", "IV", "once after paracentesis")],
         "meds": [M("Spironolactone", "100 mg", "PO", "daily", "new", "first diuretic"),
                  M("Furosemide", "40 mg", "PO", "daily", "new"),
                  M("Lactulose", "30 mL", "PO", "twice daily", "new", "titrate to three stools daily"),
                  M("Metformin", "500 mg", "PO", "twice daily"), M("Nadolol", "20 mg", "PO", "daily")],
         "imaging": [IMG("Abdominal ultrasound with Doppler", "2023-02-14 19:20", "Abdominal distension in cirrhosis.", "None available.",
                         "The liver is small and nodular with a coarsened echotexture. There is large volume ascites. The portal vein is patent with hepatopetal flow. The spleen measures 15 cm. No focal hepatic lesion is identified.",
                         "Cirrhotic morphology with large volume ascites, splenomegaly, and a patent portal vein. No focal lesion.")],
         "followup": "Hepatology in 2 weeks with a basic metabolic panel. Sodium restriction to 2 g daily. Daily weights."},
        {"admit": "2023-09-05 03:15", "disch": "2023-09-12 14:00", "service": "Hepatology", "atype": "Emergency",
         "cc": "Fever and abdominal pain.",
         "hpi": "He reports two days of diffuse abdominal pain and fever to 38.6 C at home. He has continued spironolactone and furosemide since February and has required outpatient paracentesis twice.",
         "vitals": "temperature 38.4 C, blood pressure 94/58, heart rate 104, respiratory rate 20, oxygen saturation 96% on room air",
         "exam": "Diffuse abdominal tenderness without rebound. Moderate ascites. Mild asterixis. Jaundiced.",
         "course": "Admitted with spontaneous bacterial peritonitis; ascitic fluid showed 640 neutrophils per microliter and cultures grew a gram-negative rod. Treated with ceftriaxone 2 g IV daily for five days plus albumin on days 1 and 3. Creatinine rose from 0.9 to a peak of 1.8 mg/dL on hospital day 3 despite albumin; spironolactone and furosemide were held and creatinine improved to 1.2 mg/dL by discharge. Diuretics were not restarted at discharge and will be reassessed in clinic. Ciprofloxacin 500 mg daily was started for secondary prophylaxis against spontaneous bacterial peritonitis. Sodium fell to 128 mEq/L and improved to 130 mEq/L with fluid restriction.",
         "labs": {"Creatinine": [("2023-09-05 03:40", 0.9), ("2023-09-08 05:45", 1.8), ("2023-09-12 06:00", 1.2)],
                  "Sodium": [("2023-09-05 03:40", 128), ("2023-09-12 06:00", 130)],
                  "White Blood Cells": [("2023-09-05 03:40", 12.8), ("2023-09-12 06:00", 6.1)],
                  "Total Bilirubin": [("2023-09-05 03:40", 3.6)], "Albumin": [("2023-09-05 03:40", 2.4)],
                  "Lactate": [("2023-09-05 03:40", 2.6)]},
         "dx": [("K65.2", "Spontaneous bacterial peritonitis"), ("N17.9", "Acute kidney injury"),
                ("K70.31", "Alcoholic cirrhosis of liver with ascites"), ("E87.1", "Hypo-osmolality and hyponatremia")],
         "inpatient": [M("Ceftriaxone", "2 g", "IV", "daily"), M("Albumin 25%", "75 g", "IV", "daily for two doses")],
         "meds": [M("Ciprofloxacin", "500 mg", "PO", "daily", "new", "secondary prophylaxis for spontaneous bacterial peritonitis"),
                  M("Lactulose", "30 mL", "PO", "twice daily"), M("Nadolol", "20 mg", "PO", "daily"),
                  M("Metformin", "500 mg", "PO", "twice daily")],
         "stopped": [("Spironolactone", "held for acute kidney injury; reassess in hepatology clinic"),
                     ("Furosemide", "held for acute kidney injury; reassess in hepatology clinic")],
         "progress": [PROG("2023-09-08 09:10", 4,
                           "Fever has resolved but urine output is low at 0.4 mL/kg/h. He is drowsy but oriented.",
                           "Soft abdomen with less tenderness. Moderate ascites. Asterixis present.",
                           "1. Spontaneous bacterial peritonitis: afebrile on ceftriaxone day 4; continue to complete five days.\n2. Acute kidney injury, creatinine 1.8 mg/dL from 0.9 mg/dL: concern for hepatorenal physiology; diuretics held and albumin given. Recheck daily.\n3. Hyponatremia: sodium 128 mEq/L; free water restriction to 1.5 liters.\n4. Hepatic encephalopathy, grade 1: increase lactulose titration to three stools daily.",
                           labs=["Creatinine", "Sodium"])],
         "followup": "Hepatology in 1 week to reassess diuretics and kidney function. Continue ciprofloxacin prophylaxis indefinitely."},
        {"admit": "2024-04-22 21:05", "disch": "2024-04-30 12:15", "service": "Hepatology", "atype": "Emergency",
         "cc": "Confusion and sleepiness.",
         "hpi": "His spouse reports three days of daytime sleepiness, word-finding difficulty, and two episodes of disorientation at home. He had run out of lactulose ten days earlier.",
         "vitals": "temperature 36.4 C, blood pressure 98/60, heart rate 92, respiratory rate 18, oxygen saturation 96% on room air",
         "exam": "Somnolent but rousable, oriented to person only. Prominent asterixis. Moderate ascites. Jaundiced.",
         "course": "Admitted with grade 2 hepatic encephalopathy precipitated by lactulose nonadherence and constipation. Mental status returned to baseline by hospital day 3 with lactulose titrated to 30 mL three times daily; rifaximin 550 mg twice daily was added for secondary prophylaxis. Liver synthetic function has worsened compared with the 2023 admissions: total bilirubin was 4.1 mg/dL, albumin 2.2 g/dL, and INR 1.9, and sodium was 126 mEq/L on admission. Nadolol was stopped because of a systolic blood pressure persistently below 100. He was referred for liver transplantation evaluation.",
         "labs": {"Sodium": [("2024-04-22 21:30", 126), ("2024-04-30 06:00", 129)],
                  "Total Bilirubin": [("2024-04-22 21:30", 4.1), ("2024-04-30 06:00", 3.8)],
                  "Albumin": [("2024-04-22 21:30", 2.2)], "INR": [("2024-04-22 21:30", 1.9)],
                  "Creatinine": [("2024-04-22 21:30", 1.3), ("2024-04-30 06:00", 1.2)],
                  "Platelet Count": [("2024-04-22 21:30", 74)]},
         "dx": [("K72.90", "Hepatic failure, unspecified without coma"), ("K70.31", "Alcoholic cirrhosis of liver with ascites"),
                ("E87.1", "Hypo-osmolality and hyponatremia"), ("K76.6", "Portal hypertension")],
         "meds": [M("Rifaximin", "550 mg", "PO", "twice daily", "new", "secondary prophylaxis for hepatic encephalopathy"),
                  M("Lactulose", "30 mL", "PO", "three times daily", "changed", "increased from twice daily"),
                  M("Ciprofloxacin", "500 mg", "PO", "daily"), M("Metformin", "500 mg", "PO", "twice daily")],
         "stopped": [("Nadolol", "stopped for low systolic blood pressure")],
         "followup": "Hepatology in 1 week. Transplant evaluation appointment in 3 weeks. Do not stop lactulose; call if stools fall below two per day."},
     ],
     "clinic": [PROG("2024-08-06 10:15", None,
                     "Hepatology follow-up four months after the encephalopathy admission. He has had no further confusion on lactulose and rifaximin. Abdominal girth is stable and he has not needed paracentesis since April.",
                     "Alert and fully oriented. No asterixis. Mild ascites. Chronic jaundice.",
                     "1. Cirrhosis with portal hypertension: compensated on current therapy; transplant evaluation in progress.\n2. Hepatic encephalopathy: no recurrence since April on lactulose three times daily and rifaximin.\n3. Ascites: stable; spironolactone remains off since September 2023 because of the prior acute kidney injury.",
                     labs=[("Sodium", "2024-08-06 09:30", 131), ("Creatinine", "2024-08-06 09:30", 1.1),
                           ("Total Bilirubin", "2024-08-06 09:30", 3.2)])]},

    # ---- O: ulcerative colitis, steroid course then biologic ----------------
    {"n": 15, "sex": "F", "age": 33, "pmh_short": "ulcerative colitis diagnosed four years ago",
     "pmh": ["Ulcerative colitis, left-sided, diagnosed 2019", "Iron deficiency anemia related to colitis", "Anxiety"],
     "social": "Graphic designer. Never smoker. Rare alcohol use.",
     "base": {"Sodium": 138, "Potassium": 3.8, "Urea Nitrogen": 11, "Hemoglobin": 11.6, "White Blood Cells": 8.4, "Glucose": 92, "Platelet Count": 368},
     "admissions": [
        {"admit": "2023-03-06 12:40", "disch": "2023-03-13 15:20", "service": "Gastroenterology", "atype": "Emergency",
         "cc": "Bloody diarrhea and abdominal cramping.",
         "hpi": "She reports twelve days of up to ten bloody stools daily with urgency and cramping, not improved by mesalamine. She has lost 4 kg.",
         "vitals": "temperature 37.9 C, blood pressure 104/66, heart rate 108, respiratory rate 18, oxygen saturation 99% on room air",
         "exam": "Tachycardic. Diffuse lower abdominal tenderness without peritoneal signs. Pale conjunctivae.",
         "course": "Admitted with a severe ulcerative colitis flare. Stool studies including Clostridioides difficile were negative. Flexible sigmoidoscopy showed continuous friable mucosa with ulceration to the splenic flexure, Mayo endoscopic subscore 3. She received methylprednisolone 20 mg IV every 8 hours with a partial response by day 3, so infliximab 5 mg/kg was started on hospital day 4 as the first biologic she has received. Stool frequency fell to three per day by discharge. C-reactive protein fell from 88 mg/L to 21 mg/L and hemoglobin fell from 10.1 to 9.4 g/dL without requiring transfusion. She was discharged on a prednisone 40 mg taper.",
         "labs": {"C-Reactive Protein": [("2023-03-06 13:05", 88), ("2023-03-13 06:00", 21)],
                  "Hemoglobin": [("2023-03-06 13:05", 10.1), ("2023-03-13 06:00", 9.4)],
                  "Potassium": [("2023-03-06 13:05", 3.2), ("2023-03-13 06:00", 3.9)],
                  "Albumin": [("2023-03-06 13:05", 2.9)]},
         "dx": [("K51.90", "Ulcerative colitis, unspecified, without complications"), ("D50.9", "Iron deficiency anemia"),
                ("E87.6", "Hypokalemia")],
         "inpatient": [M("Methylprednisolone", "20 mg", "IV", "every 8 hours"), M("Infliximab", "5 mg/kg", "IV", "once on day 4")],
         "meds": [M("Prednisone", "40 mg", "PO", "daily", "new", "taper by 5 mg weekly"),
                  M("Infliximab", "5 mg/kg", "IV", "weeks 0, 2, 6 then every 8 weeks", "new", "first biologic"),
                  M("Mesalamine", "2.4 g", "PO", "daily"), M("Ferrous sulfate", "325 mg", "PO", "every other day", "new")],
         "followup": "Gastroenterology in 10 days. Infusion centre for the week 2 infliximab dose. Repeat C-reactive protein and complete blood count before the next infusion."},
        {"admit": "2023-10-24 09:00", "disch": "2023-10-25 13:30", "service": "Gastroenterology", "atype": "Elective",
         "cc": "Planned surveillance colonoscopy.",
         "hpi": "She was admitted for a planned colonoscopy with dysplasia surveillance after her March flare. She has had no rectal bleeding for four months on infliximab.",
         "exam": "Well appearing. Abdomen soft and non-tender.",
         "course": "Elective colonoscopy showed mucosal healing to the splenic flexure with a Mayo endoscopic subscore of 1 and no dysplasia on random biopsies. C-reactive protein was 4 mg/L, improved from 88 mg/L in March. Infliximab was continued at the same dose and prednisone remains off since June. She was discharged the following morning after an uneventful recovery.",
         "labs": {"C-Reactive Protein": [("2023-10-24 08:10", 4)], "Hemoglobin": [("2023-10-24 08:10", 12.2)],
                  "Albumin": [("2023-10-24 08:10", 4.1)]},
         "dx": [("K51.90", "Ulcerative colitis, unspecified, without complications"), ("Z12.11", "Encounter for screening for malignant neoplasm of colon")],
         "meds": [M("Infliximab", "5 mg/kg", "IV", "every 8 weeks"), M("Mesalamine", "2.4 g", "PO", "daily"),
                  M("Ferrous sulfate", "325 mg", "PO", "every other day")],
         "followup": "Gastroenterology in 3 months. Continue infliximab every 8 weeks."},
        {"admit": "2024-07-02 16:10", "disch": "2024-07-06 11:45", "service": "Gastroenterology", "atype": "Urgent",
         "cc": "Return of bloody stools.",
         "hpi": "She reports eight days of six bloody stools daily. Her infliximab infusion was delayed by five weeks because of an insurance interruption.",
         "vitals": "temperature 37.4 C, blood pressure 110/70, heart rate 96, respiratory rate 16, oxygen saturation 99% on room air",
         "exam": "Mild left lower quadrant tenderness. No rebound or guarding.",
         "course": "Moderate ulcerative colitis flare after an interrupted infliximab schedule. Infliximab trough was undetectable with no antibodies, so the dose was intensified from 5 mg/kg to 10 mg/kg every 8 weeks rather than switching class. A short prednisone 30 mg taper was used as a bridge. C-reactive protein was 46 mg/L on admission and 18 mg/L at discharge. Hemoglobin was 10.8 g/dL. Stool frequency was two per day at discharge.",
         "labs": {"C-Reactive Protein": [("2024-07-02 16:40", 46), ("2024-07-06 06:00", 18)],
                  "Hemoglobin": [("2024-07-02 16:40", 10.8), ("2024-07-06 06:00", 10.6)]},
         "dx": [("K51.90", "Ulcerative colitis, unspecified, without complications"), ("D50.9", "Iron deficiency anemia")],
         "meds": [M("Infliximab", "10 mg/kg", "IV", "every 8 weeks", "changed", "dose intensified from 5 mg/kg"),
                  M("Prednisone", "30 mg", "PO", "daily", "new", "taper by 10 mg weekly over 3 weeks"),
                  M("Mesalamine", "2.4 g", "PO", "daily"), M("Ferrous sulfate", "325 mg", "PO", "every other day")],
         "followup": "Gastroenterology in 2 weeks with C-reactive protein. Do not miss infusion appointments; the clinic will confirm insurance authorisation."},
     ]},

    # ---- P: ischemic stroke, then atrial fibrillation found on monitoring ---
    {"n": 16, "sex": "M", "age": 74, "pmh_short": "hypertension, hyperlipidemia, and a prior ischemic stroke",
     "pmh": ["Hypertension", "Hyperlipidemia", "Left middle cerebral artery ischemic stroke in 2023", "Benign prostatic hyperplasia"],
     "social": "Retired postal worker. Former smoker, quit in 2004. Lives with his daughter.",
     "base": {"Sodium": 139, "Potassium": 4.0, "Urea Nitrogen": 19, "Hemoglobin": 13.8, "White Blood Cells": 7.8, "Glucose": 106, "Platelet Count": 236},
     "admissions": [
        {"admit": "2023-05-18 08:05", "disch": "2023-05-25 14:00", "service": "Neurology", "atype": "Emergency",
         "cc": "Sudden right-sided weakness and difficulty speaking.",
         "hpi": "He was last known well at 06:30 and was found with right arm weakness and expressive aphasia at 07:40. He arrived within the thrombolysis window.",
         "vitals": "temperature 36.8 C, blood pressure 186/96, heart rate 84, respiratory rate 16, oxygen saturation 97% on room air",
         "exam": "Expressive aphasia. Right facial droop. Right arm strength 2 out of 5, right leg 4 out of 5. NIH Stroke Scale 9.",
         "course": "Acute left middle cerebral artery territory ischemic stroke treated with intravenous thrombolysis at 08:52 with no hemorrhagic transformation on repeat imaging. NIH Stroke Scale improved from 9 to 4 by day 3. Carotid ultrasound showed 40% left internal carotid stenosis not requiring intervention, and transthoracic echocardiogram showed an ejection fraction of 55% without thrombus. A 30-day ambulatory cardiac monitor was placed at discharge because no atrial fibrillation was seen on telemetry. Clopidogrel 75 mg daily and atorvastatin 80 mg nightly were started; low-density lipoprotein cholesterol was 132 mg/dL. He was discharged to a skilled nursing facility for rehabilitation with residual mild expressive aphasia.",
         "labs": {"Glucose": [("2023-05-18 08:20", 138), ("2023-05-25 06:00", 104)],
                  "Creatinine": [("2023-05-18 08:20", 1.1), ("2023-05-25 06:00", 1.0)],
                  "Hemoglobin A1c": [("2023-05-18 08:20", 6.1)]},
         "dx": [("I63.511", "Cerebral infarction due to occlusion of right middle cerebral artery"), ("I10", "Essential hypertension"),
                ("E78.5", "Hyperlipidemia"), ("R47.01", "Aphasia")],
         "inpatient": [M("Alteplase", "0.9 mg/kg", "IV", "once")],
         "meds": [M("Clopidogrel", "75 mg", "PO", "daily", "new", "secondary stroke prevention"),
                  M("Atorvastatin", "80 mg", "PO", "nightly", "new"), M("Amlodipine", "5 mg", "PO", "daily"),
                  M("Tamsulosin", "0.4 mg", "PO", "nightly")],
         "imaging": [IMG("CT head without contrast", "2023-05-18 08:25", "Acute right-sided weakness and aphasia.", "None available.",
                         "No acute intracranial hemorrhage. There is loss of grey-white differentiation in the left insular cortex. No mass effect or midline shift. ASPECTS 9.",
                         "Early ischemic change in the left middle cerebral artery territory. No hemorrhage."),
                     IMG("MRI brain without contrast", "2023-05-19 10:15", "Confirm infarct extent after thrombolysis.",
                         "CT head from 2023-05-18.",
                         "Diffusion restriction involves the left insula and posterior frontal operculum measuring 2.6 cm. No hemorrhagic transformation. Scattered chronic small vessel ischemic change in the white matter.",
                         "Acute left middle cerebral artery branch territory infarct without hemorrhagic transformation.")],
         "progress": [PROG("2023-05-21 08:40", 4,
                           "Speech is more fluent and he is naming most objects. He walked 30 metres with physical therapy.",
                           "Mild expressive aphasia. Right arm strength 4 out of 5. NIH Stroke Scale 4.",
                           "1. Left middle cerebral artery ischemic stroke after thrombolysis: improving; continue clopidogrel and high-intensity statin.\n2. Hypertension: permissive targets past; now titrate amlodipine toward a systolic below 140.\n3. Disposition: skilled nursing facility for rehabilitation; ambulatory cardiac monitor to be placed before discharge.")],
         "followup": "Neurology in 4 weeks. The 30-day cardiac monitor report will be reviewed in clinic. Rehabilitation at a skilled nursing facility on discharge."},
        {"admit": "2023-11-03 19:25", "disch": "2023-11-06 12:00", "service": "Neurology", "atype": "Emergency",
         "cc": "Transient right hand numbness and slurred speech.",
         "hpi": "He describes twenty minutes of right hand numbness with slurred speech that resolved completely. This is his first event since the May stroke, and he has been taking clopidogrel daily.",
         "vitals": "temperature 36.7 C, blood pressure 158/88, heart rate 96, respiratory rate 16, oxygen saturation 98% on room air",
         "exam": "Speech fluent at baseline. No focal weakness or sensory loss. Irregularly irregular pulse.",
         "course": "Readmitted with a transient ischemic attack. MRI showed no new infarct. The 30-day ambulatory monitor placed in May had recorded 4.5 hours of paroxysmal atrial fibrillation, which had not yet been acted on in clinic, and an irregular rhythm was confirmed on this admission. Clopidogrel was stopped and apixaban 5 mg twice daily was started for cardioembolic stroke prevention. Creatinine was 1.0 mg/dL and weight 78 kg, so no dose reduction was needed. Amlodipine was increased from 5 mg to 10 mg daily for a systolic blood pressure of 158.",
         "labs": {"Creatinine": [("2023-11-03 19:50", 1.0), ("2023-11-06 06:00", 1.0)],
                  "Hemoglobin": [("2023-11-03 19:50", 13.6)]},
         "dx": [("G45.9", "Transient cerebral ischemic attack, unspecified"), ("I48.0", "Paroxysmal atrial fibrillation"),
                ("I10", "Essential hypertension"), ("Z86.73", "Personal history of transient ischemic attack and cerebral infarction")],
         "meds": [M("Apixaban", "5 mg", "PO", "twice daily", "new", "replaces clopidogrel for atrial fibrillation"),
                  M("Amlodipine", "10 mg", "PO", "daily", "changed", "increased from 5 mg daily"),
                  M("Atorvastatin", "80 mg", "PO", "nightly"), M("Tamsulosin", "0.4 mg", "PO", "nightly")],
         "stopped": [("Clopidogrel", "replaced by apixaban for atrial fibrillation")],
         "imaging": [IMG("MRI brain without contrast", "2023-11-04 07:30", "Transient right hand numbness and dysarthria.",
                         "MRI brain from 2023-05-19.",
                         "Encephalomalacia in the left insula and posterior frontal operculum consistent with the known May infarct. No new diffusion restriction. No hemorrhage.",
                         "No acute infarct. Expected evolution of the prior left middle cerebral artery infarct.")],
         "followup": "Neurology in 6 weeks. Do not take clopidogrel and apixaban together; clopidogrel has been stopped."},
     ]},

    # ---- Q: recurrent urinary infection, escalating resistance -------------
    {"n": 17, "sex": "F", "age": 29, "pmh_short": "recurrent urinary tract infections and vesicoureteral reflux repaired in childhood",
     "pmh": ["Recurrent urinary tract infections", "Vesicoureteral reflux, surgically repaired in childhood", "Migraine without aura"],
     "social": "Primary school teacher. Never smoker. No alcohol use.",
     "base": {"Sodium": 139, "Potassium": 4.1, "Urea Nitrogen": 12, "Hemoglobin": 12.8, "White Blood Cells": 9.2, "Glucose": 88, "Platelet Count": 268},
     "admissions": [
        {"admit": "2023-07-08 23:10", "disch": "2023-07-11 10:30", "service": "Medicine", "atype": "Emergency",
         "cc": "Fever and left flank pain.",
         "hpi": "She reports two days of dysuria followed by fever to 39 C and left flank pain. She was treated with nitrofurantoin as an outpatient last month for cystitis.",
         "vitals": "temperature 38.9 C, blood pressure 112/68, heart rate 110, respiratory rate 20, oxygen saturation 99% on room air",
         "exam": "Left costovertebral angle tenderness. Abdomen otherwise soft. No rash.",
         "course": "Admitted with acute pyelonephritis. Urine culture grew pan-sensitive Escherichia coli and blood cultures were negative. Treated with ceftriaxone 1 g IV daily, defervescing within 36 hours, then transitioned to oral cephalexin to complete a 10-day course. White blood cells fell from 16.4 to 8.9 K/uL. Creatinine was 0.8 mg/dL throughout. Renal ultrasound showed no obstruction or abscess.",
         "labs": {"White Blood Cells": [("2023-07-08 23:35", 16.4), ("2023-07-11 06:00", 8.9)],
                  "Creatinine": [("2023-07-08 23:35", 0.8), ("2023-07-11 06:00", 0.8)],
                  "Lactate": [("2023-07-08 23:35", 1.6)]},
         "dx": [("N10", "Acute pyelonephritis"), ("B96.20", "Escherichia coli as the cause of diseases classified elsewhere")],
         "inpatient": [M("Ceftriaxone", "1 g", "IV", "daily")],
         "meds": [M("Cephalexin", "500 mg", "PO", "four times daily", "new", "complete a 10-day total course"),
                  M("Sumatriptan", "50 mg", "PO", "as needed for migraine")],
         "imaging": [IMG("Renal ultrasound", "2023-07-09 09:40", "Fever with flank pain; exclude obstruction.", "None available.",
                         "Both kidneys are normal in size and echotexture. There is no hydronephrosis, calculus, or perinephric collection. The bladder is unremarkable.",
                         "No hydronephrosis or renal abscess.")],
         "followup": "Primary care in 1 week. Complete the full course of cephalexin even if symptoms resolve."},
        {"admit": "2024-01-14 04:50", "disch": "2024-01-18 16:00", "service": "Medicine", "atype": "Emergency",
         "cc": "Fever, vomiting, and low blood pressure.",
         "hpi": "She reports one day of rigors, vomiting, and right flank pain. This is her second hospital admission for a kidney infection in six months.",
         "vitals": "temperature 39.4 C, blood pressure 86/52, heart rate 124, respiratory rate 24, oxygen saturation 97% on room air",
         "exam": "Ill appearing and diaphoretic. Right costovertebral angle tenderness. Capillary refill 3 seconds.",
         "course": "Admitted with urosepsis. Lactate was 3.4 mmol/L on arrival and improved to 1.2 mmol/L after 30 mL/kg of crystalloid; no vasopressors were required. Urine and blood cultures grew an extended-spectrum beta-lactamase producing Escherichia coli, so ceftriaxone was changed to ertapenem 1 g IV daily on hospital day 2. She defervesced by day 3. Creatinine rose to 1.4 mg/dL from a baseline of 0.8 and returned to 0.9 mg/dL with fluids. She was discharged on ertapenem by outpatient infusion to complete a 14-day course. Cephalexin should not be used for future infections given the resistance pattern.",
         "labs": {"Lactate": [("2024-01-14 05:10", 3.4), ("2024-01-14 11:00", 1.2)],
                  "White Blood Cells": [("2024-01-14 05:10", 19.8), ("2024-01-18 06:00", 9.6)],
                  "Creatinine": [("2024-01-14 05:10", 1.4), ("2024-01-18 06:00", 0.9)]},
         "dx": [("A41.51", "Sepsis due to Escherichia coli"), ("N10", "Acute pyelonephritis"), ("N17.9", "Acute kidney injury")],
         "inpatient": [M("Ceftriaxone", "1 g", "IV", "daily"), M("Ertapenem", "1 g", "IV", "daily")],
         "meds": [M("Ertapenem", "1 g", "IV", "daily", "new", "outpatient infusion to complete 14 days"),
                  M("Sumatriptan", "50 mg", "PO", "as needed for migraine")],
         "stopped": [("Cephalexin", "organism is resistant; do not use for future urinary infections")],
         "progress": [PROG("2024-01-16 08:20", 3,
                           "Afebrile for 18 hours. Tolerating oral intake and urine output has normalised.",
                           "Comfortable. Mild right costovertebral angle tenderness. Blood pressure 108/66.",
                           "1. Urosepsis due to extended-spectrum beta-lactamase producing Escherichia coli: improving on ertapenem day 2; complete 14 days.\n2. Acute kidney injury: creatinine 1.1 mg/dL from a peak of 1.4 mg/dL; continue maintenance fluids.\n3. Recurrent urinary tract infection: refer to urology for imaging of the reconstructed ureter after this episode.",
                           labs=["Creatinine"])],
         "followup": "Infectious diseases in 1 week. Urology referral for evaluation of recurrent infection."},
        {"admit": "2024-09-09 14:35", "disch": "2024-09-11 11:00", "service": "Urology", "atype": "Elective",
         "cc": "Planned cystoscopy for recurrent urinary infection.",
         "hpi": "She was admitted for planned cystoscopy and retrograde imaging after two hospitalisations for kidney infection. She has had no fevers since January.",
         "exam": "Well appearing. No costovertebral angle tenderness.",
         "course": "Elective cystoscopy with retrograde pyelogram showed a mildly narrowed right ureteral orifice at the site of the childhood reimplantation without obstruction. No intervention was performed. Urine culture before the procedure was negative. Post-exposure prophylaxis with a single dose of ertapenem was given because of the prior resistant organism. She was observed overnight and discharged well.",
         "labs": {"White Blood Cells": [("2024-09-09 13:40", 7.4)], "Creatinine": [("2024-09-09 13:40", 0.8)]},
         "dx": [("N39.0", "Urinary tract infection, site not specified"), ("Z87.440", "Personal history of urinary system disease")],
         "meds": [M("Sumatriptan", "50 mg", "PO", "as needed for migraine"),
                  M("Methenamine hippurate", "1 g", "PO", "twice daily", "new", "non-antibiotic prophylaxis")],
         "followup": "Urology in 3 months. Seek care early for fever with flank pain."},
     ]},

    # ---- R: sickle cell disease, frequent readmission ----------------------
    {"n": 18, "sex": "M", "age": 24, "pmh_short": "sickle cell disease, hemoglobin SS",
     "pmh": ["Sickle cell disease, hemoglobin SS genotype", "Recurrent vaso-occlusive pain episodes",
             "Avascular necrosis of the right hip", "Chronic anemia"],
     "social": "University student. Never smoker. No alcohol use. Lives in student housing.",
     "base": {"Sodium": 138, "Potassium": 4.2, "Urea Nitrogen": 9, "Hemoglobin": 8.4, "White Blood Cells": 11.4, "Glucose": 94, "Platelet Count": 412},
     "admissions": [
        {"admit": "2023-04-02 02:30", "disch": "2023-04-06 13:15", "service": "Hematology", "atype": "Emergency",
         "cc": "Severe pain in both legs and lower back.",
         "hpi": "He reports two days of escalating bilateral leg and back pain typical of his prior vaso-occlusive episodes, triggered by a cold weekend and poor oral intake.",
         "vitals": "temperature 37.2 C, blood pressure 118/70, heart rate 102, respiratory rate 18, oxygen saturation 98% on room air",
         "exam": "Uncomfortable and guarding the legs. No joint swelling or erythema. Chest clear.",
         "course": "Admitted with an uncomplicated vaso-occlusive pain episode. Treated with intravenous fluids and patient-controlled hydromorphone, transitioning to oral oxycodone on day 3. Hemoglobin was 7.9 g/dL, near his baseline of 8.4, and no transfusion was required. There were no respiratory symptoms and the chest radiograph was clear. Hydroxyurea was continued at 1000 mg daily.",
         "labs": {"Hemoglobin": [("2023-04-02 03:00", 7.9), ("2023-04-06 06:00", 8.2)],
                  "White Blood Cells": [("2023-04-02 03:00", 13.8), ("2023-04-06 06:00", 10.9)],
                  "Total Bilirubin": [("2023-04-02 03:00", 2.8)]},
         "dx": [("D57.00", "Hb-SS disease with crisis, unspecified"), ("M87.051", "Idiopathic aseptic necrosis of right femur"),
                ("G89.29", "Other chronic pain")],
         "inpatient": [M("Hydromorphone", "0.3 mg", "IV", "patient-controlled analgesia")],
         "meds": [M("Oxycodone", "10 mg", "PO", "every 6 hours as needed for severe pain", "new", "10 tablets dispensed, no refills"),
                  M("Hydroxyurea", "1000 mg", "PO", "daily"), M("Folic acid", "1 mg", "PO", "daily")],
         "followup": "Hematology in 2 weeks. Hydration and warmth precautions reviewed."},
        {"admit": "2023-04-21 20:45", "disch": "2023-04-25 12:00", "service": "Hematology", "atype": "Emergency",
         "cc": "Return of severe leg pain.",
         "hpi": "He returns fifteen days after his last discharge with the same bilateral leg pain. He admits missing hydroxyurea doses during examinations.",
         "vitals": "temperature 37.4 C, blood pressure 122/74, heart rate 106, respiratory rate 18, oxygen saturation 97% on room air",
         "exam": "In visible pain. No focal tenderness or swelling. Lungs clear.",
         "course": "Readmission fifteen days after the previous discharge with a further vaso-occlusive pain episode, likely related to missed hydroxyurea doses. Pain was managed with the same intravenous regimen and settled by day 3. Hemoglobin was 7.6 g/dL. A pharmacist reviewed adherence strategies and a weekly pill organiser was arranged. Hydroxyurea was increased from 1000 mg to 1500 mg daily once a complete blood count confirmed an adequate neutrophil count.",
         "labs": {"Hemoglobin": [("2023-04-21 21:10", 7.6), ("2023-04-25 06:00", 8.0)],
                  "Absolute Neutrophil Count": [("2023-04-21 21:10", 4.8), ("2023-04-25 06:00", 4.1)]},
         "dx": [("D57.00", "Hb-SS disease with crisis, unspecified"), ("Z91.14", "Patient's other noncompliance with medication regimen")],
         "inpatient": [M("Hydromorphone", "0.3 mg", "IV", "patient-controlled analgesia")],
         "meds": [M("Hydroxyurea", "1500 mg", "PO", "daily", "changed", "increased from 1000 mg daily"),
                  M("Oxycodone", "10 mg", "PO", "every 6 hours as needed for severe pain"),
                  M("Folic acid", "1 mg", "PO", "daily")],
         "followup": "Hematology in 10 days with a complete blood count to check the neutrophil count on the higher hydroxyurea dose."},
        {"admit": "2023-10-11 06:20", "disch": "2023-10-19 15:40", "service": "Hematology", "atype": "Emergency",
         "cc": "Chest pain, fever, and shortness of breath.",
         "hpi": "He reports three days of pain that began in the arms and moved to the chest, with fever to 38.7 C and new breathlessness. This presentation is different from his usual pain episodes.",
         "vitals": "temperature 38.7 C, blood pressure 108/62, heart rate 118, respiratory rate 28, oxygen saturation 88% on room air",
         "exam": "Tachypneic, using accessory muscles. Decreased breath sounds at the left base with crackles.",
         "course": "Admitted with acute chest syndrome, a more severe presentation than his April pain episodes. He required up to 6 liters of oxygen by nasal cannula and received ceftriaxone with azithromycin, incentive spirometry, and a simple transfusion of two units of red cells for a hemoglobin of 6.4 g/dL. Hemoglobin rose to 9.1 g/dL after transfusion. Oxygen was weaned to room air by hospital day 6. Hydroxyurea was continued at 1500 mg daily and chronic transfusion was discussed but not started.",
         "labs": {"Hemoglobin": [("2023-10-11 06:45", 6.4), ("2023-10-13 06:00", 9.1), ("2023-10-19 06:00", 8.8)],
                  "White Blood Cells": [("2023-10-11 06:45", 18.6), ("2023-10-19 06:00", 9.8)],
                  "Lactate": [("2023-10-11 06:45", 2.1)], "Total Bilirubin": [("2023-10-11 06:45", 3.4)]},
         "dx": [("D57.01", "Hb-SS disease with acute chest syndrome"), ("J18.9", "Pneumonia, unspecified organism"),
                ("D57.00", "Hb-SS disease with crisis, unspecified")],
         "inpatient": [M("Ceftriaxone", "1 g", "IV", "daily"), M("Azithromycin", "500 mg", "IV", "daily"),
                       M("Hydromorphone", "0.3 mg", "IV", "patient-controlled analgesia")],
         "meds": [M("Hydroxyurea", "1500 mg", "PO", "daily"), M("Folic acid", "1 mg", "PO", "daily"),
                  M("Amoxicillin-clavulanate", "875 mg", "PO", "twice daily", "new", "complete a 7-day total course"),
                  M("Oxycodone", "5 mg", "PO", "every 8 hours as needed for severe pain", "changed", "reduced from 10 mg")],
         "imaging": [IMG("Chest radiograph, portable", "2023-10-11 07:15", "Fever, chest pain, and hypoxemia in sickle cell disease.",
                         "Chest radiograph from 2023-04-02.",
                         "There is a new left lower lobe airspace opacity with a small left pleural effusion. The right lung is clear. The cardiac silhouette is normal in size.",
                         "New left lower lobe consolidation with a small effusion, consistent with acute chest syndrome."),
                     IMG("Chest radiograph, PA and lateral", "2023-10-18 09:00", "Follow-up of acute chest syndrome before discharge.",
                         "Chest radiograph from 2023-10-11.",
                         "The left lower lobe opacity has nearly resolved with only minimal residual atelectasis. The pleural effusion has resolved. No new opacity.",
                         "Near complete resolution of the left lower lobe consolidation.")],
         "progress": [PROG("2023-10-13 11:00", 3,
                           "Breathing is easier after transfusion. Oxygen requirement down from 6 liters to 3 liters by nasal cannula.",
                           "Respiratory rate 22. Crackles at the left base, improved. Saturation 95% on 3 liters.",
                           "1. Acute chest syndrome: improving after simple transfusion; hemoglobin 9.1 g/dL from 6.4 g/dL. Continue antibiotics and incentive spirometry, wean oxygen.\n2. Vaso-occlusive pain: transitioning from patient-controlled analgesia to oral opioids.\n3. Sickle cell disease: continue hydroxyurea 1500 mg daily; discuss chronic transfusion at follow-up.",
                           labs=["Hemoglobin"])],
         "followup": "Hematology in 1 week. Return immediately for fever, chest pain, or breathlessness."},
     ]},

    # ---- S: gallstone pancreatitis then elective cholecystectomy -----------
    {"n": 19, "sex": "F", "age": 41, "pmh_short": "obesity and gallstones",
     "pmh": ["Cholelithiasis", "Obesity", "Gastroesophageal reflux disease"],
     "social": "Works in retail management. Never smoker. Occasional alcohol, less than two drinks weekly.",
     "base": {"Sodium": 138, "Potassium": 3.9, "Urea Nitrogen": 14, "Hemoglobin": 13.1, "White Blood Cells": 9.8, "Glucose": 104, "Platelet Count": 288},
     "admissions": [
        {"admit": "2023-06-11 01:15", "disch": "2023-06-16 12:40", "service": "Surgery", "atype": "Emergency",
         "cc": "Severe upper abdominal pain radiating to the back.",
         "hpi": "She reports eight hours of severe epigastric pain radiating to the back with vomiting, beginning after a fatty meal.",
         "vitals": "temperature 37.6 C, blood pressure 128/78, heart rate 112, respiratory rate 20, oxygen saturation 97% on room air",
         "exam": "Marked epigastric tenderness with voluntary guarding. Bowel sounds reduced. No jaundice.",
         "course": "Admitted with acute gallstone pancreatitis. Lipase was 1240 IU/L on admission and fell to 180 IU/L by discharge. Alanine aminotransferase was 284 IU/L, supporting a biliary cause. Managed with aggressive lactated Ringer solution, analgesia, and early oral intake from day 3. There was no organ failure and no necrosis on imaging. Cholecystectomy was deferred to an elective admission because of local expertise scheduling, and she was counselled to avoid fatty meals in the interim.",
         "labs": {"Lipase": [("2023-06-11 01:40", 1240), ("2023-06-13 06:00", 410), ("2023-06-16 06:00", 180)],
                  "Alanine Aminotransferase": [("2023-06-11 01:40", 284), ("2023-06-16 06:00", 92)],
                  "Total Bilirubin": [("2023-06-11 01:40", 2.1), ("2023-06-16 06:00", 1.0)],
                  "White Blood Cells": [("2023-06-11 01:40", 14.6), ("2023-06-16 06:00", 8.2)],
                  "Calcium": [("2023-06-11 01:40", 8.4)]},
         "dx": [("K85.10", "Biliary acute pancreatitis without necrosis or infection"), ("K80.20", "Calculus of gallbladder without cholecystitis"),
                ("E66.9", "Obesity, unspecified")],
         "inpatient": [M("Lactated Ringer solution", "250 mL", "IV", "per hour"), M("Hydromorphone", "0.5 mg", "IV", "every 4 hours as needed")],
         "meds": [M("Acetaminophen", "1 g", "PO", "every 8 hours as needed for pain", "new"),
                  M("Pantoprazole", "40 mg", "PO", "daily"), M("Ondansetron", "4 mg", "PO", "every 8 hours as needed for nausea", "new")],
         "imaging": [IMG("Abdominal ultrasound", "2023-06-11 08:30", "Epigastric pain with elevated lipase.", "None available.",
                         "Multiple mobile gallstones are present in the gallbladder, the largest measuring 9 mm. The gallbladder wall is not thickened and there is no pericholecystic fluid. The common bile duct measures 5 mm and is not dilated. The visualised pancreas is obscured by bowel gas.",
                         "Cholelithiasis without cholecystitis. No biliary dilatation.")],
         "followup": "Surgery clinic in 3 weeks to schedule cholecystectomy. Low fat diet. Return for recurrent pain, fever, or jaundice."},
        {"admit": "2023-07-10 06:45", "disch": "2023-07-12 11:20", "service": "Surgery", "atype": "Elective",
         "cc": "Planned gallbladder removal.",
         "hpi": "She was admitted for a planned laparoscopic cholecystectomy four weeks after her admission for gallstone pancreatitis. She has had two episodes of mild self-limited upper abdominal pain since then.",
         "exam": "Well appearing. Abdomen soft with mild residual epigastric tenderness.",
         "course": "Underwent uncomplicated laparoscopic cholecystectomy with intraoperative cholangiogram showing no retained stone. Lipase was normal at 42 IU/L before surgery, down from a peak of 1240 IU/L during the June admission. She tolerated diet on postoperative day 1 and was discharged home. Pathology showed chronic cholecystitis with cholelithiasis.",
         "labs": {"Lipase": [("2023-07-10 06:00", 42)], "Alanine Aminotransferase": [("2023-07-10 06:00", 34)],
                  "White Blood Cells": [("2023-07-10 06:00", 7.9), ("2023-07-12 06:00", 9.1)],
                  "Total Bilirubin": [("2023-07-10 06:00", 0.7)]},
         "dx": [("K81.1", "Chronic cholecystitis"), ("K80.20", "Calculus of gallbladder without cholecystitis"),
                ("Z98.890", "Other specified postprocedural states")],
         "meds": [M("Acetaminophen", "1 g", "PO", "every 8 hours as needed for pain"),
                  M("Pantoprazole", "40 mg", "PO", "daily")],
         "stopped": [("Ondansetron", "no longer needed after surgery")],
         "followup": "Surgery clinic in 2 weeks for wound check. Resume a normal diet as tolerated."},
     ]},

    # ---- T: lung mass -> cancer diagnosis -> malignant effusion ------------
    {"n": 20, "sex": "M", "age": 69, "pmh_short": "COPD and a 45 pack-year smoking history",
     "pmh": ["Chronic obstructive pulmonary disease, GOLD stage 2", "Hypertension", "Former smoker, 45 pack-years"],
     "social": "Retired dock worker. Quit smoking in 2023 after the first admission. Lives alone with a supportive neighbour.",
     "base": {"Sodium": 137, "Potassium": 4.3, "Urea Nitrogen": 16, "Hemoglobin": 13.4, "White Blood Cells": 8.8, "Glucose": 110, "Platelet Count": 298},
     "admissions": [
        {"admit": "2023-03-21 15:50", "disch": "2023-03-24 11:10", "service": "Pulmonology", "atype": "Emergency",
         "cc": "Coughing up blood.",
         "hpi": "He reports five days of blood-streaked sputum and a 6 kg unintentional weight loss over three months. He continues to smoke.",
         "vitals": "temperature 36.9 C, blood pressure 142/84, heart rate 92, respiratory rate 18, oxygen saturation 94% on room air",
         "exam": "Decreased breath sounds in the right upper zone. No clubbing. No palpable lymphadenopathy.",
         "course": "Admitted with hemoptysis. CT of the chest showed a 3.8 cm right upper lobe mass with mediastinal lymphadenopathy. Bleeding settled with conservative management and no bronchial artery embolisation was required. No tissue diagnosis was obtained during this admission; bronchoscopy was arranged as a planned admission. Smoking cessation counselling was given with nicotine replacement. Hemoglobin was stable at 13.1 g/dL.",
         "labs": {"Hemoglobin": [("2023-03-21 16:20", 13.1), ("2023-03-24 06:00", 13.0)],
                  "Sodium": [("2023-03-21 16:20", 134), ("2023-03-24 06:00", 135)],
                  "White Blood Cells": [("2023-03-21 16:20", 9.4)]},
         "dx": [("R04.2", "Hemoptysis"), ("R91.1", "Solitary pulmonary nodule"), ("J44.9", "Chronic obstructive pulmonary disease, unspecified")],
         "meds": [M("Nicotine patch", "21 mg", "TD", "daily", "new", "smoking cessation"),
                  M("Tiotropium", "18 mcg", "INH", "daily"), M("Amlodipine", "10 mg", "PO", "daily")],
         "imaging": [IMG("CT chest with contrast", "2023-03-22 09:00", "Hemoptysis with weight loss in a smoker.", "None available.",
                         "There is a 3.8 cm spiculated mass in the right upper lobe abutting the pleura. Right hilar and subcarinal lymph nodes measure up to 1.8 cm in short axis. There is centrilobular emphysema. No pleural effusion. No lytic bone lesion in the imaged skeleton.",
                         "Spiculated 3.8 cm right upper lobe mass with hilar and subcarinal lymphadenopathy, suspicious for primary lung malignancy. Tissue diagnosis recommended.")],
         "followup": "Pulmonology in 1 week. Bronchoscopy with biopsy will be arranged as a planned admission. Stop smoking."},
        {"admit": "2023-05-02 07:00", "disch": "2023-05-04 14:20", "service": "Pulmonology", "atype": "Elective",
         "cc": "Planned bronchoscopy and biopsy.",
         "hpi": "He was admitted for a planned bronchoscopy with endobronchial ultrasound to biopsy the right upper lobe mass found in March. He has stopped smoking since that admission.",
         "exam": "Well appearing. Decreased breath sounds in the right upper zone unchanged.",
         "course": "Underwent bronchoscopy with endobronchial ultrasound-guided sampling of the right upper lobe mass and subcarinal node. Pathology showed non-small cell lung cancer, adenocarcinoma subtype, with subcarinal nodal involvement, giving a clinical stage of IIIA. Molecular testing was sent. A small post-procedure pneumothorax was managed conservatively and resolved on repeat imaging before discharge. He was referred to oncology for concurrent chemoradiotherapy.",
         "labs": {"Hemoglobin": [("2023-05-02 06:15", 12.9)], "Sodium": [("2023-05-02 06:15", 133)],
                  "Creatinine": [("2023-05-02 06:15", 1.0)], "Albumin": [("2023-05-02 06:15", 3.6)]},
         "dx": [("C34.11", "Malignant neoplasm of upper lobe, right bronchus or lung"), ("J93.11", "Primary spontaneous pneumothorax"),
                ("J44.9", "Chronic obstructive pulmonary disease, unspecified")],
         "meds": [M("Tiotropium", "18 mcg", "INH", "daily"), M("Amlodipine", "10 mg", "PO", "daily"),
                  M("Nicotine patch", "14 mg", "TD", "daily", "changed", "step down from 21 mg")],
         "imaging": [IMG("Chest radiograph, PA and lateral", "2023-05-03 08:10", "Post-bronchoscopy; exclude pneumothorax.",
                         "CT chest from 2023-03-22.",
                         "There is a small apical right pneumothorax measuring 8 mm without mediastinal shift. The known right upper lobe mass is partially obscured. No new consolidation.",
                         "Small right apical pneumothorax, suitable for conservative management.")],
         "followup": "Oncology in 1 week to plan concurrent chemoradiotherapy. Return for sudden breathlessness or chest pain."},
        {"admit": "2024-01-16 18:30", "disch": "2024-01-22 13:00", "service": "Oncology", "atype": "Emergency",
         "cc": "Worsening breathlessness.",
         "hpi": "He reports two weeks of progressive breathlessness, now at rest, ten months after his lung cancer diagnosis and six months after completing chemoradiotherapy.",
         "vitals": "temperature 36.8 C, blood pressure 118/72, heart rate 104, respiratory rate 24, oxygen saturation 90% on room air",
         "exam": "Dullness to percussion and absent breath sounds over the right lower two thirds of the chest. Tracheal position midline.",
         "course": "Admitted with a large right malignant pleural effusion representing disease progression. Therapeutic thoracentesis removed 1.4 liters of straw-coloured fluid; cytology confirmed adenocarcinoma. An indwelling pleural catheter was placed on hospital day 3 after the effusion reinfused. Breathlessness improved and oxygen was weaned to room air. Sodium was 129 mEq/L, lower than the 134 mEq/L recorded in March 2023, attributed to the syndrome of inappropriate antidiuretic hormone secretion; fluid restriction improved it to 132 mEq/L. Palliative care was involved for symptom management and goals of care.",
         "labs": {"Sodium": [("2024-01-16 19:00", 129), ("2024-01-22 06:00", 132)],
                  "Hemoglobin": [("2024-01-16 19:00", 10.8), ("2024-01-22 06:00", 10.6)],
                  "Albumin": [("2024-01-16 19:00", 3.0)], "Creatinine": [("2024-01-16 19:00", 0.9)]},
         "dx": [("J91.0", "Malignant pleural effusion"), ("C34.11", "Malignant neoplasm of upper lobe, right bronchus or lung"),
                ("E22.2", "Syndrome of inappropriate secretion of antidiuretic hormone"), ("J44.9", "Chronic obstructive pulmonary disease, unspecified")],
         "meds": [M("Morphine sulfate immediate release", "5 mg", "PO", "every 4 hours as needed for breathlessness", "new", "palliative"),
                  M("Dexamethasone", "4 mg", "PO", "daily", "new"), M("Tiotropium", "18 mcg", "INH", "daily"),
                  M("Amlodipine", "10 mg", "PO", "daily")],
         "imaging": [IMG("Chest radiograph, PA and lateral", "2024-01-16 19:40", "Progressive breathlessness in treated lung cancer.",
                         "Chest radiograph from 2023-05-03.",
                         "There is a large right pleural effusion obscuring two thirds of the right hemithorax with passive atelectasis. The left lung is clear. No pneumothorax.",
                         "New large right pleural effusion. Comparison with the 2023 studies shows interval progression."),
                     IMG("Chest radiograph, portable", "2024-01-19 08:30", "After indwelling pleural catheter placement.",
                         "Chest radiograph from 2024-01-16.",
                         "The indwelling pleural catheter is in good position in the right pleural space. The effusion has substantially decreased with re-expansion of the right lower lobe. Small residual effusion remains.",
                         "Interval decrease in the right pleural effusion after catheter drainage.")],
         "progress": [PROG("2024-01-19 10:45", 4,
                           "He is more comfortable after catheter placement and is drinking normally within his fluid restriction.",
                           "Respiratory rate 18. Air entry improved at the right base. Saturation 94% on room air.",
                           "1. Malignant pleural effusion: indwelling pleural catheter placed; district nursing to drain three times weekly.\n2. Hyponatremia from syndrome of inappropriate antidiuretic hormone: sodium 130 mEq/L from 129 mEq/L; continue 1 liter fluid restriction.\n3. Metastatic non-small cell lung cancer: palliative care involved; oncology to discuss second-line systemic therapy at follow-up.",
                           labs=["Sodium"])],
         "followup": "Oncology in 5 days to discuss second-line therapy. District nursing for pleural catheter drainage three times weekly. Fluid restriction of 1 liter daily."},
     ]},

    # ---- U: simple elective hernia repair (single short admission) ---------
    {"n": 21, "sex": "M", "age": 52, "pmh_short": "an uncomplicated right inguinal hernia",
     "pmh": ["Right inguinal hernia", "Seasonal allergic rhinitis"],
     "social": "Works as an electrician. Never smoker. Occasional alcohol use.",
     "base": {"Sodium": 140, "Potassium": 4.2, "Urea Nitrogen": 13, "Hemoglobin": 15.1, "White Blood Cells": 6.2, "Glucose": 94, "Platelet Count": 244},
     "admissions": [
        {"admit": "2023-10-03 07:15", "disch": "2023-10-04 10:30", "service": "Surgery", "atype": "Elective",
         "cc": "Planned repair of a right inguinal hernia.",
         "hpi": "He has an eighteen-month history of a reducible right groin bulge that aches at the end of a working day. There has been no incarceration or obstruction.",
         "exam": "Reducible right inguinal hernia. Abdomen otherwise soft and non-tender. No scrotal swelling.",
         "course": "Underwent an uncomplicated open right inguinal hernia repair with mesh under spinal anaesthesia. He passed urine, tolerated a diet, and mobilised independently the same evening. There were no complications and no new medications other than short-course analgesia. He was discharged the following morning.",
         "labs": {"Hemoglobin": [("2023-10-03 06:30", 15.0)], "Creatinine": [("2023-10-03 06:30", 1.0)]},
         "dx": [("K40.90", "Unilateral inguinal hernia without obstruction or gangrene"), ("J30.2", "Other seasonal allergic rhinitis")],
         "inpatient": [M("Cefazolin", "2 g", "IV", "once before incision")],
         "meds": [M("Acetaminophen", "1 g", "PO", "every 8 hours as needed for pain", "new"),
                  M("Ibuprofen", "400 mg", "PO", "every 8 hours as needed for pain", "new", "take with food for up to 5 days"),
                  M("Loratadine", "10 mg", "PO", "daily")],
         "followup": "Surgery clinic in 3 weeks. No lifting over 10 kg for 4 weeks. Return for fever or increasing groin pain."},
     ]},

    # ---- V: diabetic foot osteomyelitis progressing to amputation ----------
    {"n": 22, "sex": "M", "age": 63, "pmh_short": "type 2 diabetes mellitus with peripheral neuropathy and peripheral artery disease",
     "pmh": ["Type 2 diabetes mellitus for 18 years", "Diabetic peripheral neuropathy", "Peripheral artery disease",
             "Hypertension", "Diabetic retinopathy"],
     "social": "Retired lorry driver. Former smoker, quit in 2018. Lives with his wife who assists with dressings.",
     "base": {"Sodium": 136, "Potassium": 4.5, "Urea Nitrogen": 24, "Hemoglobin": 11.8, "White Blood Cells": 9.6, "Glucose": 186, "Platelet Count": 312},
     "admissions": [
        {"admit": "2023-09-18 11:20", "disch": "2023-09-27 14:00", "service": "Medicine", "atype": "Urgent",
         "cc": "A non-healing ulcer on the right foot.",
         "hpi": "He reports a plantar ulcer under the right second metatarsal head present for seven weeks, now with malodorous drainage. He has no pain because of neuropathy.",
         "vitals": "temperature 37.8 C, blood pressure 148/82, heart rate 96, respiratory rate 18, oxygen saturation 96% on room air",
         "exam": "A 2 cm plantar ulcer under the right second metatarsal head probing to bone. Surrounding erythema to the mid foot. Absent dorsalis pedis and posterior tibial pulses on the right.",
         "course": "Admitted with diabetic foot osteomyelitis. MRI confirmed marrow signal change in the second metatarsal head. Bone biopsy grew methicillin-sensitive Staphylococcus aureus. Treated with intravenous cefazolin and discharged on a six-week course by outpatient infusion. Hemoglobin A1c was 9.8%, so insulin glargine was increased from 24 to 32 units nightly and metformin was continued. Ankle-brachial index was 0.6 on the right and vascular surgery recommended outpatient angiography. White blood cells fell from 14.2 to 8.4 K/uL.",
         "labs": {"Hemoglobin A1c": [("2023-09-18 11:45", 9.8)],
                  "White Blood Cells": [("2023-09-18 11:45", 14.2), ("2023-09-27 06:00", 8.4)],
                  "Glucose": [("2023-09-18 11:45", 268), ("2023-09-27 06:00", 172)],
                  "C-Reactive Protein": [("2023-09-18 11:45", 112), ("2023-09-27 06:00", 34)],
                  "Creatinine": [("2023-09-18 11:45", 1.3), ("2023-09-27 06:00", 1.2)]},
         "dx": [("M86.171", "Other acute osteomyelitis, right ankle and foot"), ("E11.621", "Type 2 diabetes mellitus with foot ulcer"),
                ("E11.42", "Type 2 diabetes mellitus with diabetic polyneuropathy"), ("I70.219", "Atherosclerosis of native arteries of extremities with intermittent claudication")],
         "inpatient": [M("Cefazolin", "2 g", "IV", "every 8 hours")],
         "meds": [M("Cefazolin", "2 g", "IV", "every 8 hours", "new", "six-week course by outpatient infusion"),
                  M("Insulin glargine", "32 units", "SC", "nightly", "changed", "increased from 24 units nightly"),
                  M("Metformin", "1000 mg", "PO", "twice daily"), M("Atorvastatin", "40 mg", "PO", "nightly"),
                  M("Lisinopril", "20 mg", "PO", "daily"), M("Gabapentin", "300 mg", "PO", "three times daily")],
         "imaging": [IMG("MRI right foot without contrast", "2023-09-19 14:30", "Plantar ulcer probing to bone.", "None available.",
                         "There is confluent marrow oedema in the second metatarsal head with cortical destruction and an adjacent plantar soft tissue ulcer measuring 2.1 cm. A small rim-enhancing collection is not identified. No involvement of the remaining metatarsals.",
                         "Osteomyelitis of the right second metatarsal head with an overlying plantar ulcer.")],
         "progress": [PROG("2023-09-22 09:15", 5,
                           "Afebrile for three days. The wound is less malodorous and drainage has decreased.",
                           "Ulcer base clean with granulation. Erythema has receded to the forefoot. Absent distal pulses on the right.",
                           "1. Diabetic foot osteomyelitis: improving on cefazolin; bone culture grew methicillin-sensitive Staphylococcus aureus. Plan six weeks total.\n2. Peripheral artery disease with ankle-brachial index 0.6: outpatient angiography; this limits healing potential.\n3. Type 2 diabetes with hemoglobin A1c 9.8%: glargine increased to 32 units nightly; diabetes educator to see before discharge.",
                           labs=["Glucose"])],
         "followup": "Infectious diseases in 1 week, podiatry weekly for offloading, vascular surgery for angiography within 4 weeks. Total contact cast applied."},
        {"admit": "2024-03-04 08:40", "disch": "2024-03-12 15:30", "service": "Surgery", "atype": "Urgent",
         "cc": "Black discoloration of the right forefoot.",
         "hpi": "He reports two weeks of darkening of the right second and third toes despite completing antibiotics in November. He did not attend the vascular angiography appointment arranged at the previous discharge.",
         "vitals": "temperature 37.3 C, blood pressure 152/86, heart rate 94, respiratory rate 18, oxygen saturation 95% on room air",
         "exam": "Dry gangrene of the right second and third toes with a line of demarcation at the metatarsophalangeal joints. No proximal cellulitis.",
         "course": "Readmitted five months after the osteomyelitis admission with dry gangrene of the right forefoot. Angiography showed occlusion of the anterior tibial artery with a poor peroneal runoff, and angioplasty of the posterior tibial artery was performed to improve inflow. He then underwent a right transmetatarsal amputation with primary closure. Hemoglobin fell from 11.4 to 9.6 g/dL postoperatively without transfusion. Hemoglobin A1c had improved from 9.8% to 8.1% since September. He was discharged with home health nursing for dressings.",
         "labs": {"Hemoglobin A1c": [("2024-03-04 09:00", 8.1)],
                  "Hemoglobin": [("2024-03-04 09:00", 11.4), ("2024-03-12 06:00", 9.6)],
                  "White Blood Cells": [("2024-03-04 09:00", 10.8), ("2024-03-12 06:00", 8.0)],
                  "Creatinine": [("2024-03-04 09:00", 1.4), ("2024-03-12 06:00", 1.3)],
                  "Glucose": [("2024-03-04 09:00", 198), ("2024-03-12 06:00", 148)]},
         "dx": [("I96", "Gangrene, not elsewhere classified"), ("E11.52", "Type 2 diabetes mellitus with diabetic peripheral angiopathy with gangrene"),
                ("Z89.431", "Acquired absence of right foot"), ("I70.261", "Atherosclerosis of native arteries of right leg with gangrene")],
         "inpatient": [M("Cefazolin", "2 g", "IV", "every 8 hours"), M("Heparin", "5000 units", "SC", "every 8 hours")],
         "meds": [M("Aspirin", "81 mg", "PO", "daily", "new", "after peripheral angioplasty"),
                  M("Clopidogrel", "75 mg", "PO", "daily", "new", "for 3 months after angioplasty"),
                  M("Insulin glargine", "28 units", "SC", "nightly", "changed", "reduced from 32 units after reduced intake"),
                  M("Metformin", "1000 mg", "PO", "twice daily"), M("Atorvastatin", "80 mg", "PO", "nightly", "changed", "increased from 40 mg"),
                  M("Gabapentin", "300 mg", "PO", "three times daily")],
         "stopped": [("Lisinopril", "held for a creatinine of 1.4 mg/dL around angiography; restart at follow-up if stable")],
         "followup": "Vascular surgery in 2 weeks for wound review. Home health nursing daily for dressings. Prosthetics and orthotics referral for a custom shoe. Do not miss the angiography follow-up."},
     ]},

    # ---- W: pulmonary embolism then anticoagulation-related bleed ----------
    {"n": 23, "sex": "F", "age": 58, "pmh_short": "hypertension and a recent provoked pulmonary embolism",
     "pmh": ["Pulmonary embolism, provoked by a long-haul flight, 2023", "Hypertension", "Osteoarthritis of the knees"],
     "social": "Works as a travel agent. Never smoker. Two glasses of wine weekly.",
     "base": {"Sodium": 139, "Potassium": 4.0, "Urea Nitrogen": 15, "Hemoglobin": 13.2, "White Blood Cells": 7.6, "Glucose": 98, "Platelet Count": 254},
     "admissions": [
        {"admit": "2023-10-05 13:40", "disch": "2023-10-09 11:15", "service": "Medicine", "atype": "Emergency",
         "cc": "Sudden shortness of breath and chest pain on breathing in.",
         "hpi": "She reports sudden breathlessness and right-sided pleuritic chest pain two days after a twelve-hour flight. She has no leg swelling and takes no hormonal therapy.",
         "vitals": "temperature 37.1 C, blood pressure 124/76, heart rate 112, respiratory rate 24, oxygen saturation 92% on room air",
         "exam": "Tachycardic and tachypneic. Lungs clear. No calf tenderness or asymmetry.",
         "course": "Admitted with an acute segmental pulmonary embolism of the right lower lobe, provoked by recent travel. She was haemodynamically stable with a normal troponin T of 0.02 ng/mL and a normal right ventricle on echocardiogram, so systemic thrombolysis was not indicated. Apixaban was started at 10 mg twice daily for seven days followed by 5 mg twice daily, planned for a total of three months. Oxygen was weaned to room air by day 2. Hemoglobin was 13.0 g/dL at discharge. Ibuprofen was stopped because of the bleeding risk on anticoagulation.",
         "labs": {"Troponin T": [("2023-10-05 14:10", 0.02), ("2023-10-05 20:00", 0.02)],
                  "Hemoglobin": [("2023-10-05 14:10", 13.4), ("2023-10-09 06:00", 13.0)],
                  "Creatinine": [("2023-10-05 14:10", 0.9)]},
         "dx": [("I26.99", "Other pulmonary embolism without acute cor pulmonale"), ("I10", "Essential hypertension"),
                ("M17.0", "Bilateral primary osteoarthritis of knee")],
         "inpatient": [M("Enoxaparin", "80 mg", "SC", "every 12 hours")],
         "meds": [M("Apixaban", "10 mg", "PO", "twice daily for 7 days then 5 mg twice daily", "new", "planned 3-month course"),
                  M("Amlodipine", "5 mg", "PO", "daily"), M("Acetaminophen", "1 g", "PO", "every 8 hours as needed for knee pain", "new")],
         "stopped": [("Ibuprofen", "bleeding risk on apixaban; use acetaminophen instead")],
         "imaging": [IMG("CT pulmonary angiogram", "2023-10-05 16:20", "Pleuritic chest pain and hypoxemia after long-haul travel.",
                         "None available.",
                         "There are filling defects in the right lower lobe segmental and subsegmental pulmonary arteries. The right ventricle to left ventricle diameter ratio is 0.8. There is a small peripheral wedge-shaped opacity at the right base. No pleural effusion.",
                         "Acute right lower lobe segmental pulmonary embolism without right heart strain. Small peripheral infarct.")],
         "followup": "Primary care in 2 weeks. Anticoagulation clinic to confirm the three-month stop date. Avoid non-steroidal anti-inflammatory drugs while on apixaban."},
        {"admit": "2024-02-11 02:05", "disch": "2024-02-16 13:50", "service": "Gastroenterology", "atype": "Emergency",
         "cc": "Black stools and light-headedness.",
         "hpi": "She reports three days of black tarry stools and light-headedness on standing. She has continued apixaban beyond the planned three months because the stop date was never confirmed in clinic.",
         "vitals": "temperature 36.5 C, blood pressure 96/58, heart rate 116, respiratory rate 18, oxygen saturation 98% on room air",
         "exam": "Pale. Resting tachycardia with a postural drop of 22 mmHg. Melena on rectal examination. Abdomen soft.",
         "course": "Admitted with an upper gastrointestinal bleed on apixaban. Hemoglobin fell from a baseline of 13.0 to 7.1 g/dL and she received two units of red cells, rising to 9.4 g/dL. Apixaban was held on admission. Upper endoscopy on hospital day 1 showed a clean-based duodenal ulcer with no active bleeding; Helicobacter pylori testing was positive and eradication therapy was started. There was no further bleeding and hemoglobin was 9.6 g/dL at discharge. The gastroenterology team and the medical team documented different plans for anticoagulation during this admission, and the discharge decision is recorded below.\nDocumentation note: the hospital day 3 progress note records a plan to resume apixaban at discharge, while this discharge summary records apixaban as remaining on hold. The discharge instruction is the one to follow: apixaban is on hold pending gastroenterology review at 6 weeks.",
         "labs": {"Hemoglobin": [("2024-02-11 02:30", 7.1), ("2024-02-12 06:00", 9.4), ("2024-02-16 06:00", 9.6)],
                  "Urea Nitrogen": [("2024-02-11 02:30", 38), ("2024-02-16 06:00", 14)],
                  "Platelet Count": [("2024-02-11 02:30", 232)],
                  "Creatinine": [("2024-02-11 02:30", 1.1), ("2024-02-16 06:00", 0.9)]},
         "dx": [("K92.2", "Gastrointestinal hemorrhage, unspecified"), ("K26.4", "Chronic or unspecified duodenal ulcer with hemorrhage"),
                ("D62", "Acute posthemorrhagic anemia"), ("B96.81", "Helicobacter pylori as the cause of diseases classified elsewhere")],
         "inpatient": [M("Pantoprazole", "8 mg", "IV", "per hour for 72 hours")],
         "meds": [M("Pantoprazole", "40 mg", "PO", "twice daily", "new", "8 weeks then reassess"),
                  M("Amoxicillin", "1 g", "PO", "twice daily", "new", "Helicobacter pylori eradication for 14 days"),
                  M("Clarithromycin", "500 mg", "PO", "twice daily", "new", "Helicobacter pylori eradication for 14 days"),
                  M("Amlodipine", "5 mg", "PO", "daily"), M("Acetaminophen", "1 g", "PO", "every 8 hours as needed for knee pain")],
         "stopped": [("Apixaban", "held after gastrointestinal bleed; on hold pending gastroenterology review at 6 weeks")],
         "progress": [PROG("2024-02-13 10:20", 3,
                           "No further melena for 48 hours. She is eating and mobilising without light-headedness.",
                           "Less pale. Heart rate 84, blood pressure 112/70 without a postural drop.",
                           "1. Upper gastrointestinal bleed from a duodenal ulcer: clean base on endoscopy, no rebleeding; continue high-dose proton pump inhibitor.\n2. Anticoagulation: the index pulmonary embolism was provoked and more than three months have elapsed, so the balance favours stopping. Our plan is to resume apixaban at discharge only if gastroenterology agrees.\n3. Helicobacter pylori: eradication therapy started; urea breath test at 8 weeks.",
                           labs=["Hemoglobin"])],
         "followup": "Gastroenterology in 6 weeks with a urea breath test; anticoagulation will be reviewed at that visit. Do not restart apixaban until that review. Primary care in 1 week for a complete blood count."},
     ]},

    # ---- X: rheumatoid arthritis, methotrexate held and restarted ----------
    {"n": 24, "sex": "F", "age": 62, "pmh_short": "seropositive rheumatoid arthritis on methotrexate",
     "pmh": ["Seropositive rheumatoid arthritis diagnosed 2016", "Hypertension", "Osteopenia"],
     "social": "Retired librarian. Never smoker. No alcohol use while on methotrexate.",
     "base": {"Sodium": 139, "Potassium": 4.1, "Urea Nitrogen": 17, "Hemoglobin": 12.0, "White Blood Cells": 7.0, "Glucose": 96, "Platelet Count": 298},
     "admissions": [
        {"admit": "2023-08-14 16:30", "disch": "2023-08-18 12:10", "service": "Medicine", "atype": "Emergency",
         "cc": "Fever and a productive cough.",
         "hpi": "She reports four days of fever, a productive cough, and right-sided pleuritic pain. She takes methotrexate 20 mg weekly and had her last dose three days before admission.",
         "vitals": "temperature 38.6 C, blood pressure 118/70, heart rate 102, respiratory rate 22, oxygen saturation 93% on room air",
         "exam": "Crackles and bronchial breathing at the right base. Symmetric metacarpophalangeal swelling without active synovitis.",
         "course": "Admitted with community-acquired pneumonia. Treated with ceftriaxone and azithromycin, transitioned to oral amoxicillin-clavulanate to complete seven days. Methotrexate was held on admission because of the active infection. Oxygen requirement peaked at 2 liters by nasal cannula and she was on room air by day 3. White blood cells were 15.6 K/uL on admission and 7.8 K/uL at discharge. Rheumatology advised restarting methotrexate only after the infection had fully resolved, which is documented as a clinic decision rather than a discharge medication.",
         "labs": {"White Blood Cells": [("2023-08-14 17:00", 15.6), ("2023-08-18 06:00", 7.8)],
                  "C-Reactive Protein": [("2023-08-14 17:00", 164), ("2023-08-18 06:00", 38)],
                  "Alanine Aminotransferase": [("2023-08-14 17:00", 28)],
                  "Creatinine": [("2023-08-14 17:00", 0.9)]},
         "dx": [("J18.9", "Pneumonia, unspecified organism"), ("M05.79", "Rheumatoid arthritis with rheumatoid factor, multiple sites"),
                ("I10", "Essential hypertension")],
         "inpatient": [M("Ceftriaxone", "1 g", "IV", "daily"), M("Azithromycin", "500 mg", "IV", "daily")],
         "meds": [M("Amoxicillin-clavulanate", "875 mg", "PO", "twice daily", "new", "complete a 7-day total course"),
                  M("Prednisone", "5 mg", "PO", "daily"), M("Amlodipine", "5 mg", "PO", "daily"),
                  M("Calcium carbonate with vitamin D", "500 mg", "PO", "twice daily")],
         "stopped": [("Methotrexate", "held for active pneumonia; rheumatology to decide on restarting")],
         "imaging": [IMG("Chest radiograph, PA and lateral", "2023-08-14 17:40", "Fever and productive cough.", "None available.",
                         "There is a right lower lobe airspace opacity with air bronchograms. No pleural effusion or pneumothorax. The cardiac silhouette is normal.",
                         "Right lower lobe pneumonia.")],
         "followup": "Primary care in 1 week. Rheumatology in 4 weeks to decide about restarting methotrexate. Do not restart methotrexate before that review."},
        {"admit": "2024-05-20 10:00", "disch": "2024-05-24 14:30", "service": "Medicine", "atype": "Urgent",
         "cc": "A red, hot, swollen right lower leg.",
         "hpi": "She reports three days of spreading redness of the right shin after a scratch while gardening. Methotrexate was restarted at 15 mg weekly in October 2023 and she has been taking it since.",
         "vitals": "temperature 38.2 C, blood pressure 126/76, heart rate 98, respiratory rate 18, oxygen saturation 97% on room air",
         "exam": "Confluent erythema of the right shin with warmth and a marked border. No fluctuance or crepitus. No joint effusion.",
         "course": "Admitted with cellulitis of the right lower leg. Treated with intravenous cefazolin for 48 hours then oral cephalexin to complete ten days. Methotrexate was held again for the duration of the infection, the second time it has been interrupted for infection, and was restarted at 15 mg weekly on the day of discharge because the cellulitis had clearly responded. C-reactive protein fell from 96 to 22 mg/L. Ultrasound excluded deep vein thrombosis and abscess.",
         "labs": {"C-Reactive Protein": [("2024-05-20 10:30", 96), ("2024-05-24 06:00", 22)],
                  "White Blood Cells": [("2024-05-20 10:30", 13.2), ("2024-05-24 06:00", 8.6)],
                  "Alanine Aminotransferase": [("2024-05-20 10:30", 24)],
                  "Creatinine": [("2024-05-20 10:30", 1.0)]},
         "dx": [("L03.115", "Cellulitis of right lower limb"), ("M05.79", "Rheumatoid arthritis with rheumatoid factor, multiple sites"),
                ("I10", "Essential hypertension")],
         "inpatient": [M("Cefazolin", "2 g", "IV", "every 8 hours")],
         "meds": [M("Cephalexin", "500 mg", "PO", "four times daily", "new", "complete a 10-day total course"),
                  M("Methotrexate", "15 mg", "PO", "weekly", "changed", "restarted at discharge after being held for cellulitis"),
                  M("Folic acid", "5 mg", "PO", "weekly"), M("Prednisone", "5 mg", "PO", "daily"),
                  M("Amlodipine", "5 mg", "PO", "daily")],
         "followup": "Primary care in 1 week. Rheumatology in 6 weeks with liver function tests and a complete blood count on methotrexate."},
        {"admit": "2024-11-12 08:20", "disch": "2024-11-13 12:00", "service": "Rheumatology", "atype": "Elective",
         "cc": "Planned joint injection and infusion assessment.",
         "hpi": "She was admitted for a planned bilateral knee corticosteroid injection and assessment for biologic therapy after two infection-related interruptions of methotrexate.",
         "exam": "Bilateral knee effusions with crepitus. Metacarpophalangeal joints swollen but not hot.",
         "course": "Elective admission for ultrasound-guided bilateral knee injections, performed without complication. Disease activity remains moderate on methotrexate 15 mg weekly. Given two infection-related interruptions and persistent activity, the plan is to start a biologic after screening; interferon gamma release assay and hepatitis serology were sent and were pending at discharge. Methotrexate was continued unchanged.",
         "labs": {"C-Reactive Protein": [("2024-11-12 07:40", 18)], "Hemoglobin": [("2024-11-12 07:40", 11.8)],
                  "Alanine Aminotransferase": [("2024-11-12 07:40", 31)]},
         "dx": [("M05.79", "Rheumatoid arthritis with rheumatoid factor, multiple sites"), ("M17.0", "Bilateral primary osteoarthritis of knee")],
         "meds": [M("Methotrexate", "15 mg", "PO", "weekly"), M("Folic acid", "5 mg", "PO", "weekly"),
                  M("Prednisone", "5 mg", "PO", "daily"), M("Amlodipine", "5 mg", "PO", "daily")],
         "followup": "Rheumatology in 4 weeks to review screening results before starting a biologic."},
     ]},

    # ---- Y: alcohol withdrawal then early readmission with hepatitis -------
    {"n": 25, "sex": "M", "age": 45, "pmh_short": "alcohol use disorder and hepatic steatosis",
     "pmh": ["Alcohol use disorder", "Hepatic steatosis on prior imaging", "Generalised anxiety disorder"],
     "social": "Works intermittently in construction. Reports drinking a litre of spirits daily before the first admission. Smokes ten cigarettes daily.",
     "base": {"Sodium": 136, "Potassium": 3.5, "Urea Nitrogen": 8, "Hemoglobin": 12.6, "White Blood Cells": 8.2, "Glucose": 102, "Platelet Count": 142},
     "admissions": [
        {"admit": "2023-11-02 23:50", "disch": "2023-11-06 10:40", "service": "Medicine", "atype": "Emergency",
         "cc": "Shaking, sweating, and anxiety after stopping drinking.",
         "hpi": "He stopped drinking 36 hours before arrival and reports tremor, sweating, nausea, and two episodes of visual misperception. He has had withdrawal seizures in the past.",
         "vitals": "temperature 37.4 C, blood pressure 158/96, heart rate 118, respiratory rate 20, oxygen saturation 98% on room air",
         "exam": "Diaphoretic with a coarse resting tremor. Oriented but anxious. No asterixis or jaundice.",
         "course": "Admitted with alcohol withdrawal. Managed with a symptom-triggered lorazepam protocol; the peak withdrawal score was on hospital day 2 and he required 18 mg of lorazepam in total. There were no seizures or delirium tremens. He received thiamine, folate, and a multivitamin. Potassium was 3.0 mEq/L and magnesium 1.4 mg/dL on admission, both replaced. Alanine aminotransferase was 96 IU/L. He accepted a referral to an addiction service and was started on oral thiamine at discharge, but declined naltrexone at this admission.",
         "labs": {"Potassium": [("2023-11-02 23:59", 3.0), ("2023-11-06 06:00", 3.9)],
                  "Magnesium": [("2023-11-02 23:59", 1.4), ("2023-11-06 06:00", 2.0)],
                  "Alanine Aminotransferase": [("2023-11-02 23:59", 96)],
                  "Total Bilirubin": [("2023-11-02 23:59", 1.4)],
                  "Platelet Count": [("2023-11-02 23:59", 138)]},
         "dx": [("F10.239", "Alcohol dependence with withdrawal, unspecified"), ("E87.6", "Hypokalemia"),
                ("K76.0", "Fatty (change of) liver, not elsewhere classified")],
         "inpatient": [M("Lorazepam", "2 mg", "IV", "as needed by symptom-triggered protocol"),
                       M("Thiamine", "500 mg", "IV", "three times daily for 3 days")],
         "meds": [M("Thiamine", "100 mg", "PO", "daily", "new"), M("Folic acid", "1 mg", "PO", "daily", "new"),
                  M("Multivitamin", "1 tablet", "PO", "daily", "new"),
                  M("Sertraline", "50 mg", "PO", "daily")],
         "followup": "Addiction service appointment in 5 days. Primary care in 1 week with liver function tests. Return immediately for confusion, fever, or vomiting blood."},
        {"admit": "2023-11-15 15:20", "disch": "2023-11-20 11:00", "service": "Hepatology", "atype": "Emergency",
         "cc": "Yellow eyes and abdominal pain.",
         "hpi": "He returns nine days after discharge with jaundice, right upper quadrant pain, and poor appetite. He resumed drinking heavily four days after the previous discharge and did not attend the addiction appointment.",
         "vitals": "temperature 37.9 C, blood pressure 122/74, heart rate 104, respiratory rate 18, oxygen saturation 97% on room air",
         "exam": "Jaundiced with tender hepatomegaly. No ascites or asterixis. Mild tremor.",
         "course": "Readmission nine days after the previous discharge, this time with alcohol-associated hepatitis rather than isolated withdrawal. Total bilirubin rose from 1.4 mg/dL at the previous admission to 8.6 mg/dL and INR was 1.6; alanine aminotransferase was 118 IU/L with an aspartate-predominant pattern. A discriminant function above 32 supported corticosteroids, so prednisolone 40 mg daily was started on hospital day 2 after excluding infection. Bilirubin fell to 6.9 mg/dL by discharge. A further short lorazepam course covered withdrawal. He accepted naltrexone at this admission and was referred again to the addiction service with a peer navigator.",
         "labs": {"Total Bilirubin": [("2023-11-15 15:45", 8.6), ("2023-11-20 06:00", 6.9)],
                  "INR": [("2023-11-15 15:45", 1.6), ("2023-11-20 06:00", 1.4)],
                  "Alanine Aminotransferase": [("2023-11-15 15:45", 118), ("2023-11-20 06:00", 96)],
                  "Albumin": [("2023-11-15 15:45", 2.9)],
                  "White Blood Cells": [("2023-11-15 15:45", 13.4), ("2023-11-20 06:00", 9.2)],
                  "Creatinine": [("2023-11-15 15:45", 0.7)]},
         "dx": [("K70.10", "Alcoholic hepatitis without ascites"), ("F10.239", "Alcohol dependence with withdrawal, unspecified"),
                ("K76.0", "Fatty (change of) liver, not elsewhere classified"), ("R17", "Unspecified jaundice")],
         "inpatient": [M("Lorazepam", "1 mg", "PO", "as needed by symptom-triggered protocol")],
         "meds": [M("Prednisolone", "40 mg", "PO", "daily", "new", "28-day course with a day 7 response check"),
                  M("Naltrexone", "50 mg", "PO", "daily", "new", "for alcohol use disorder"),
                  M("Thiamine", "100 mg", "PO", "daily"), M("Folic acid", "1 mg", "PO", "daily"),
                  M("Sertraline", "50 mg", "PO", "daily"), M("Pantoprazole", "40 mg", "PO", "daily", "new", "while on corticosteroids")],
         "progress": [PROG("2023-11-18 09:30", 4,
                           "Appetite is returning and abdominal pain is milder. No confusion.",
                           "Jaundice unchanged. Liver edge tender but less so. No asterixis.",
                           "1. Alcohol-associated hepatitis: prednisolone day 3; bilirubin 7.6 mg/dL, trending down. Reassess response on day 7 with the Lille score.\n2. Alcohol use disorder: naltrexone accepted this admission; addiction service to see before discharge.\n3. Alcohol withdrawal: scores are low, lorazepam rarely required.",
                           labs=["Total Bilirubin"])],
         "followup": "Hepatology in 7 days for the day 7 steroid response assessment. Addiction service in 3 days. Complete abstinence is essential."},
     ]},

    # ---- Z: hip fracture with postoperative delirium, then recovery --------
    {"n": 26, "sex": "F", "age": 84, "pmh_short": "osteoporosis, mild cognitive impairment, and hypertension",
     "pmh": ["Osteoporosis", "Mild cognitive impairment", "Hypertension", "Hypothyroidism"],
     "social": "Widowed, lives alone in a bungalow with a daughter nearby. Never smoker. No alcohol use.",
     "base": {"Sodium": 137, "Potassium": 4.0, "Urea Nitrogen": 21, "Hemoglobin": 11.6, "White Blood Cells": 8.0, "Glucose": 100, "Platelet Count": 262},
     "admissions": [
        {"admit": "2023-12-01 12:10", "disch": "2023-12-09 15:00", "service": "Orthopedics", "atype": "Emergency",
         "cc": "Fall at home with right hip pain and inability to stand.",
         "hpi": "She tripped on a rug and fell onto her right side. There was no loss of consciousness, chest pain, or palpitation before the fall.",
         "vitals": "temperature 36.5 C, blood pressure 142/78, heart rate 88, respiratory rate 18, oxygen saturation 96% on room air",
         "exam": "The right leg is shortened and externally rotated. No head injury. Abbreviated mental test score 8 out of 10, at her baseline.",
         "course": "Admitted with a displaced right femoral neck fracture after a mechanical fall and underwent hemiarthroplasty within 24 hours. Hemoglobin fell from 11.4 to 8.9 g/dL postoperatively and she received one unit of red cells. She developed hypoactive delirium on postoperative day 2, managed without antipsychotics using orientation, hearing aids, early mobilisation, and removal of the urinary catheter; delirium had resolved by day 5. Bone protection was started with alendronate and vitamin D after a vitamin D level of 14 ng/mL. She was discharged to a skilled nursing facility for rehabilitation.",
         "labs": {"Hemoglobin": [("2023-12-01 12:40", 11.4), ("2023-12-03 06:00", 8.9), ("2023-12-09 06:00", 9.8)],
                  "Creatinine": [("2023-12-01 12:40", 1.0), ("2023-12-09 06:00", 0.9)],
                  "Calcium": [("2023-12-01 12:40", 8.9)],
                  "Thyroid Stimulating Hormone": [("2023-12-02 06:00", 3.4)]},
         "dx": [("S72.001A", "Fracture of unspecified part of neck of right femur, initial encounter"),
                ("F05", "Delirium due to known physiological condition"), ("M81.0", "Age-related osteoporosis without current pathological fracture"),
                ("D62", "Acute posthemorrhagic anemia"), ("E03.9", "Hypothyroidism, unspecified")],
         "inpatient": [M("Cefazolin", "2 g", "IV", "every 8 hours for 24 hours"), M("Enoxaparin", "40 mg", "SC", "daily")],
         "meds": [M("Alendronate", "70 mg", "PO", "weekly", "new", "with calcium and vitamin D"),
                  M("Cholecalciferol", "2000 units", "PO", "daily", "new", "for a vitamin D level of 14 ng/mL"),
                  M("Calcium carbonate", "500 mg", "PO", "twice daily", "new"),
                  M("Enoxaparin", "40 mg", "SC", "daily", "new", "for 28 days after hip surgery"),
                  M("Levothyroxine", "75 mcg", "PO", "daily"), M("Amlodipine", "5 mg", "PO", "daily"),
                  M("Acetaminophen", "1 g", "PO", "three times daily", "new", "scheduled, not as needed")],
         "imaging": [IMG("Radiograph of the right hip and pelvis", "2023-12-01 12:50", "Fall with an inability to weight bear.", "None available.",
                         "There is a displaced subcapital fracture of the right femoral neck with varus angulation. The left hip is intact. The pelvic ring is preserved. Diffuse osteopenia is noted.",
                         "Displaced right femoral neck fracture. Diffuse osteopenia.")],
         "progress": [PROG("2023-12-05 08:50", 5,
                           "She is brighter, recognises her daughter, and sat out of bed for four hours. Sleeping at night again.",
                           "Alert and oriented to person, place, and month. Wound clean and dry. Transfers with one assistant.",
                           "1. Right femoral neck fracture after hemiarthroplasty: wound healthy; weight bear as tolerated.\n2. Postoperative delirium: resolved with non-pharmacological measures; no antipsychotic was used.\n3. Postoperative anemia: hemoglobin 9.5 g/dL after one unit; no further transfusion.\n4. Osteoporosis: alendronate and vitamin D started; falls assessment done.",
                           labs=["Hemoglobin"])],
         "followup": "Orthopedics in 6 weeks with a radiograph. Rehabilitation at a skilled nursing facility on discharge. Continue enoxaparin for 28 days from surgery."},
     ],
     "clinic": [PROG("2024-01-15 11:00", None,
                     "Geriatrics follow-up six weeks after hip fracture surgery. She returned home from the skilled nursing facility two weeks ago and walks indoors with a frame. There has been no further confusion and no further falls.",
                     "Walks 20 metres with a frame. Hip wound healed. Abbreviated mental test score 9 out of 10.",
                     "1. Right hip hemiarthroplasty: recovering well; continue home physiotherapy.\n2. Postoperative delirium: fully resolved, cognition back to baseline.\n3. Osteoporosis: continue alendronate weekly with calcium and vitamin D; hemoglobin has recovered to 11.2 g/dL.\n4. Enoxaparin completed at 28 days and has been stopped.",
                     labs=[("Hemoglobin", "2024-01-15 10:20", 11.2), ("Calcium", "2024-01-15 10:20", 9.2),
                           ("Creatinine", "2024-01-15 10:20", 0.9)])]},

    # ---- AA: asthma, escalating controller therapy ------------------------
    {"n": 27, "sex": "F", "age": 22, "pmh_short": "moderate persistent asthma since childhood",
     "pmh": ["Asthma since age 6", "Allergic rhinitis", "Eczema"],
     "social": "University student living in shared housing with a cat. Never smoker. Occasional alcohol use.",
     "base": {"Sodium": 139, "Potassium": 4.0, "Urea Nitrogen": 10, "Hemoglobin": 13.0, "White Blood Cells": 8.6, "Glucose": 92, "Platelet Count": 276},
     "admissions": [
        {"admit": "2023-02-03 04:40", "disch": "2023-02-05 11:20", "service": "Medicine", "atype": "Emergency",
         "cc": "Wheezing and breathlessness not relieved by her inhaler.",
         "hpi": "She reports two days of worsening wheeze after a cold, using her salbutamol inhaler every two hours without relief. She uses her preventer inhaler only when symptomatic.",
         "vitals": "temperature 37.0 C, blood pressure 118/70, heart rate 118, respiratory rate 26, oxygen saturation 93% on room air",
         "exam": "Speaking in short phrases. Widespread polyphonic wheeze with a prolonged expiratory phase. No accessory muscle fatigue.",
         "course": "Admitted with a moderate asthma exacerbation triggered by a viral infection and poor controller adherence. Treated with nebulised salbutamol and ipratropium and a five-day course of prednisolone 40 mg daily. Peak expiratory flow improved from 42% to 78% of predicted. She was changed from a salbutamol-only regimen to a regular inhaled corticosteroid with formoterol as maintenance and reliever therapy, and inhaler technique was taught. She did not require magnesium or intensive care.",
         "labs": {"White Blood Cells": [("2023-02-03 05:00", 11.2), ("2023-02-05 06:00", 9.0)],
                  "Potassium": [("2023-02-03 05:00", 3.4), ("2023-02-05 06:00", 3.9)]},
         "dx": [("J45.41", "Moderate persistent asthma with acute exacerbation"), ("J30.1", "Allergic rhinitis due to pollen"),
                ("E87.6", "Hypokalemia")],
         "inpatient": [M("Salbutamol", "5 mg", "NEB", "every 4 hours"), M("Ipratropium", "500 mcg", "NEB", "every 6 hours")],
         "meds": [M("Budesonide-formoterol", "200-6 mcg", "INH", "two puffs twice daily and as needed", "new",
                    "maintenance and reliever therapy replacing salbutamol alone"),
                  M("Prednisolone", "40 mg", "PO", "daily", "new", "5-day course, no taper needed"),
                  M("Fluticasone nasal spray", "50 mcg", "INH", "two sprays daily")],
         "stopped": [("Salbutamol inhaler", "replaced by budesonide-formoterol as reliever")],
         "followup": "Primary care in 1 week with a written asthma action plan. Asthma clinic in 6 weeks with spirometry."},
        {"admit": "2023-08-28 21:15", "disch": "2023-08-30 10:00", "service": "Medicine", "atype": "Emergency",
         "cc": "Wheeze after moving into a house with a cat.",
         "hpi": "She reports five days of wheeze and night waking after moving into shared housing with a cat. She has been using budesonide-formoterol regularly since February.",
         "vitals": "temperature 36.8 C, blood pressure 112/68, heart rate 104, respiratory rate 22, oxygen saturation 95% on room air",
         "exam": "Full sentences. Expiratory wheeze throughout. No tachypnea at rest after nebulisation.",
         "course": "Second asthma admission of the year, milder than February, with a clear allergen trigger. Treated with nebulised bronchodilators and prednisolone for five days. Peak expiratory flow improved from 61% to 85% of predicted, a better starting point than the 42% recorded in February. The budesonide-formoterol dose was stepped up from 200-6 to 400-12 mcg twice daily, and montelukast was added given the allergic phenotype. Blood eosinophils were 0.6 K/uL and total immunoglobulin E was elevated; she was referred to the severe asthma clinic for biologic assessment if a third exacerbation occurs.",
         "labs": {"White Blood Cells": [("2023-08-28 21:40", 10.4), ("2023-08-30 06:00", 9.2)],
                  "Absolute Neutrophil Count": [("2023-08-28 21:40", 7.2)],
                  "Potassium": [("2023-08-28 21:40", 3.6)]},
         "dx": [("J45.41", "Moderate persistent asthma with acute exacerbation"), ("J30.81", "Allergic rhinitis due to animal hair and dander")],
         "inpatient": [M("Salbutamol", "5 mg", "NEB", "every 6 hours")],
         "meds": [M("Budesonide-formoterol", "400-12 mcg", "INH", "two puffs twice daily and as needed", "changed", "stepped up from 200-6 mcg"),
                  M("Montelukast", "10 mg", "PO", "nightly", "new", "allergic phenotype"),
                  M("Prednisolone", "40 mg", "PO", "daily", "new", "5-day course"),
                  M("Fluticasone nasal spray", "50 mcg", "INH", "two sprays daily")],
         "followup": "Asthma clinic in 4 weeks. Allergen avoidance advice given regarding the cat. Severe asthma clinic referral if another exacerbation occurs."},
        {"admit": "2024-06-18 13:30", "disch": "2024-06-19 12:40", "service": "Pulmonology", "atype": "Elective",
         "cc": "Planned assessment for biologic therapy.",
         "hpi": "She was admitted for a planned day assessment for biologic therapy after two exacerbations in 2023. She has had no exacerbation in the last ten months.",
         "exam": "No wheeze at rest. Peak expiratory flow 88% of predicted.",
         "course": "Elective severe asthma assessment. Spirometry showed a forced expiratory volume in one second of 82% predicted with 14% bronchodilator reversibility. Blood eosinophils were 0.5 K/uL. Because she has had no exacerbation since August 2023 on stepped-up therapy, a biologic was deferred and current treatment was continued unchanged. Adherence monitoring by pharmacy refill data was arranged.",
         "labs": {"Absolute Neutrophil Count": [("2024-06-18 12:50", 4.2)], "White Blood Cells": [("2024-06-18 12:50", 7.8)]},
         "dx": [("J45.40", "Moderate persistent asthma, uncomplicated"), ("J30.81", "Allergic rhinitis due to animal hair and dander")],
         "meds": [M("Budesonide-formoterol", "400-12 mcg", "INH", "two puffs twice daily and as needed"),
                  M("Montelukast", "10 mg", "PO", "nightly"), M("Fluticasone nasal spray", "50 mcg", "INH", "two sprays daily")],
         "followup": "Asthma clinic in 6 months. Return to the severe asthma clinic if an exacerbation requires oral corticosteroids."},
     ]},

    # ---- AB: Graves disease, medical therapy then thyroidectomy ------------
    {"n": 28, "sex": "F", "age": 37, "pmh_short": "Graves disease",
     "pmh": ["Graves disease diagnosed 2023", "Mild thyroid eye disease", "Iron deficiency anemia, resolved"],
     "social": "Works as a paramedic. Never smoker, which is relevant to her thyroid eye disease. Rare alcohol use.",
     "base": {"Sodium": 139, "Potassium": 4.1, "Urea Nitrogen": 11, "Hemoglobin": 12.4, "White Blood Cells": 6.8, "Glucose": 90, "Platelet Count": 252},
     "admissions": [
        {"admit": "2023-05-09 17:55", "disch": "2023-05-12 13:20", "service": "Endocrinology", "atype": "Emergency",
         "cc": "Palpitations, weight loss, and tremor.",
         "hpi": "She reports six weeks of palpitations, 8 kg of weight loss despite a good appetite, heat intolerance, and tremor, with a heart rate of 140 recorded at work.",
         "vitals": "temperature 37.6 C, blood pressure 138/62, heart rate 136, respiratory rate 20, oxygen saturation 99% on room air",
         "exam": "Fine resting tremor. Warm moist skin. Smooth symmetric goitre with a bruit. Mild lid retraction without proptosis.",
         "course": "Admitted with newly diagnosed thyrotoxicosis from Graves disease. Thyroid stimulating hormone was undetectable at 0.01 uIU/mL with a raised free thyroxine, and thyroid stimulating immunoglobulin was positive. There was no fever or confusion and the Burch-Wartofsky score did not meet criteria for thyroid storm. Methimazole 20 mg daily and propranolol 40 mg three times daily were started, the first antithyroid therapy she has received. Heart rate fell from 136 to 88 by discharge. Baseline neutrophil count was 3.8 K/uL and alanine aminotransferase 28 IU/L before methimazole.",
         "labs": {"Thyroid Stimulating Hormone": [("2023-05-09 18:20", 0.01), ("2023-05-12 06:00", 0.01)],
                  "Absolute Neutrophil Count": [("2023-05-09 18:20", 3.8), ("2023-05-12 06:00", 3.6)],
                  "Alanine Aminotransferase": [("2023-05-09 18:20", 28)],
                  "Calcium": [("2023-05-09 18:20", 9.8)]},
         "dx": [("E05.00", "Thyrotoxicosis with diffuse goiter without thyrotoxic crisis"), ("H05.20", "Unspecified exophthalmos"),
                ("R00.2", "Palpitations")],
         "inpatient": [M("Propranolol", "40 mg", "PO", "three times daily")],
         "meds": [M("Methimazole", "20 mg", "PO", "daily", "new", "first antithyroid drug"),
                  M("Propranolol", "40 mg", "PO", "three times daily", "new", "for symptom control"),
                  M("Selenium", "100 mcg", "PO", "twice daily", "new", "for mild thyroid eye disease")],
         "followup": "Endocrinology in 4 weeks with thyroid function tests. Seek urgent care for fever or sore throat on methimazole because of the risk of agranulocytosis."},
        {"admit": "2024-02-05 06:50", "disch": "2024-02-08 14:10", "service": "Surgery", "atype": "Elective",
         "cc": "Planned total thyroidectomy.",
         "hpi": "She was admitted for a planned total thyroidectomy after nine months of methimazole. She chose surgery over radioiodine because of her thyroid eye disease and her work with radiation-sensitive equipment.",
         "exam": "Euthyroid clinically. Heart rate 76. Goitre unchanged in size. No proptosis.",
         "course": "Underwent an uncomplicated total thyroidectomy. She was rendered euthyroid preoperatively on methimazole, which was stopped on the day of surgery along with propranolol. Corrected calcium fell to 7.6 mg/dL on postoperative day 1 from 9.8 mg/dL in May, consistent with transient hypoparathyroidism, and improved to 8.4 mg/dL on oral calcium and calcitriol. Levothyroxine 112 mcg daily was started on postoperative day 1. Pathology showed diffuse hyperplasia with no malignancy. There was no voice change and vocal cords were mobile on review.",
         "labs": {"Calcium": [("2024-02-05 06:10", 9.4), ("2024-02-06 06:00", 7.6), ("2024-02-08 06:00", 8.4)],
                  "Thyroid Stimulating Hormone": [("2024-02-05 06:10", 1.80)],
                  "Hemoglobin": [("2024-02-05 06:10", 12.8), ("2024-02-08 06:00", 11.9)]},
         "dx": [("E05.00", "Thyrotoxicosis with diffuse goiter without thyrotoxic crisis"), ("E89.2", "Postprocedural hypoparathyroidism"),
                ("E89.0", "Postprocedural hypothyroidism")],
         "inpatient": [M("Calcium gluconate", "2 g", "IV", "once for symptomatic hypocalcemia")],
         "meds": [M("Levothyroxine", "112 mcg", "PO", "daily", "new", "lifelong after total thyroidectomy"),
                  M("Calcium carbonate", "1 g", "PO", "three times daily", "new", "for postoperative hypocalcemia"),
                  M("Calcitriol", "0.25 mcg", "PO", "twice daily", "new", "wean as calcium recovers")],
         "stopped": [("Methimazole", "no longer needed after total thyroidectomy"),
                     ("Propranolol", "no longer needed after total thyroidectomy"),
                     ("Selenium", "course completed")],
         "followup": "Surgery in 2 weeks. Endocrinology in 6 weeks with thyroid function tests and calcium. Report tingling around the mouth or in the fingers immediately."},
     ]},

    # ---- AC: obstructing ureteral stone then elective stent removal --------
    {"n": 29, "sex": "M", "age": 38, "pmh_short": "recurrent calcium oxalate kidney stones",
     "pmh": ["Recurrent nephrolithiasis, calcium oxalate", "Gout", "Overweight"],
     "social": "Works as a chef in a hot kitchen with limited fluid intake during shifts. Never smoker.",
     "base": {"Sodium": 139, "Potassium": 4.2, "Urea Nitrogen": 16, "Hemoglobin": 14.6, "White Blood Cells": 8.4, "Glucose": 96, "Platelet Count": 238},
     "admissions": [
        {"admit": "2023-09-02 03:20", "disch": "2023-09-04 12:30", "service": "Urology", "atype": "Emergency",
         "cc": "Severe left flank pain and blood in the urine.",
         "hpi": "He reports six hours of colicky left flank pain radiating to the groin with visible blood in the urine. He has passed stones twice before without needing an operation.",
         "vitals": "temperature 37.5 C, blood pressure 146/88, heart rate 104, respiratory rate 18, oxygen saturation 98% on room air",
         "exam": "Restless and unable to find a comfortable position. Left costovertebral angle tenderness. Abdomen soft.",
         "course": "Admitted with an obstructing 7 mm stone at the left vesicoureteric junction with moderate hydronephrosis. Because he had a temperature of 37.9 C and a white blood cell count of 14.8 K/uL, an infected obstructed system could not be excluded, so a left ureteric stent was placed urgently and ceftriaxone was given. Urine culture was subsequently negative. Creatinine peaked at 1.6 mg/dL and improved to 1.1 mg/dL after decompression. Definitive stone treatment was planned as an elective admission. Tamsulosin was started and he was advised to strain his urine.",
         "labs": {"Creatinine": [("2023-09-02 03:45", 1.6), ("2023-09-04 06:00", 1.1)],
                  "White Blood Cells": [("2023-09-02 03:45", 14.8), ("2023-09-04 06:00", 9.6)],
                  "Calcium": [("2023-09-02 03:45", 9.6)]},
         "dx": [("N20.1", "Calculus of ureter"), ("N13.2", "Hydronephrosis with renal and ureteral calculous obstruction"),
                ("M10.9", "Gout, unspecified")],
         "inpatient": [M("Ceftriaxone", "1 g", "IV", "daily"), M("Ketorolac", "15 mg", "IV", "every 6 hours as needed")],
         "meds": [M("Tamsulosin", "0.4 mg", "PO", "nightly", "new", "medical expulsive therapy"),
                  M("Acetaminophen", "1 g", "PO", "every 8 hours as needed for pain", "new"),
                  M("Allopurinol", "300 mg", "PO", "daily")],
         "imaging": [IMG("CT of the abdomen and pelvis without contrast", "2023-09-02 04:30", "Acute left flank pain with hematuria.",
                         "None available.",
                         "There is a 7 mm calculus at the left vesicoureteric junction with moderate left hydronephrosis and perinephric stranding. A 4 mm non-obstructing calculus is present in the lower pole of the right kidney. No free fluid.",
                         "Obstructing 7 mm left vesicoureteric junction calculus with moderate hydronephrosis. Incidental non-obstructing 4 mm right lower pole calculus.")],
         "followup": "Urology in 2 weeks to arrange definitive stone treatment. The stent must be removed; do not miss the appointment. Increase fluid intake to produce 2.5 liters of urine daily."},
        {"admit": "2023-09-23 07:30", "disch": "2023-09-24 11:00", "service": "Urology", "atype": "Elective",
         "cc": "Planned stone treatment and stent removal.",
         "hpi": "He was admitted for a planned ureteroscopy with laser lithotripsy and removal of the stent placed three weeks earlier. The stent has caused urinary frequency and intermittent flank discomfort.",
         "exam": "Comfortable. No costovertebral angle tenderness. Abdomen soft.",
         "course": "Elective ureteroscopy with laser lithotripsy cleared the left vesicoureteric junction stone and the stent was removed at the end of the procedure. Stone analysis confirmed calcium oxalate monohydrate. Creatinine was 1.0 mg/dL, back to his usual level after the peak of 1.6 mg/dL during the September obstruction. A 24-hour urine collection was arranged as an outpatient. Tamsulosin was stopped as it is no longer needed. The incidental right lower pole stone was left alone and will be watched.",
         "labs": {"Creatinine": [("2023-09-23 07:00", 1.0), ("2023-09-24 06:00", 1.0)],
                  "White Blood Cells": [("2023-09-23 07:00", 7.6)], "Calcium": [("2023-09-23 07:00", 9.4)]},
         "dx": [("N20.1", "Calculus of ureter"), ("Z46.6", "Encounter for fitting and adjustment of urinary device")],
         "meds": [M("Potassium citrate", "10 mEq", "PO", "twice daily", "new", "stone prevention"),
                  M("Allopurinol", "300 mg", "PO", "daily"),
                  M("Acetaminophen", "1 g", "PO", "every 8 hours as needed for pain")],
         "stopped": [("Tamsulosin", "stent removed, no longer needed")],
         "followup": "Urology in 3 months with a 24-hour urine collection and a kidney ultrasound to follow the right lower pole stone."},
     ]},

    # ---- AD: chronic hepatitis B, single elective admission (simple) -------
    {"n": 30, "sex": "M", "age": 47, "pmh_short": "chronic hepatitis B infection on surveillance",
     "pmh": ["Chronic hepatitis B, hepatitis B e antigen negative", "Latent tuberculosis, treated in 2020"],
     "social": "Works in software. Never smoker. No alcohol use. Regularly attends hepatology follow-up.",
     "base": {"Sodium": 140, "Potassium": 4.0, "Urea Nitrogen": 13, "Hemoglobin": 14.8, "White Blood Cells": 5.8, "Glucose": 94, "Platelet Count": 186},
     "admissions": [
        {"admit": "2024-03-19 08:00", "disch": "2024-03-20 10:15", "service": "Hepatology", "atype": "Elective",
         "cc": "Planned liver biopsy.",
         "hpi": "He was admitted for a planned percutaneous liver biopsy to stage fibrosis after transient elastography suggested intermediate stiffness. He has had no jaundice, bleeding, or swelling.",
         "exam": "Well appearing. No stigmata of chronic liver disease. Liver edge not palpable. No ascites or edema.",
         "course": "Underwent an uncomplicated ultrasound-guided percutaneous liver biopsy and was observed for six hours without bleeding or pain. Histology showed stage 2 fibrosis with mild inflammation and no cirrhosis. Alanine aminotransferase was 52 IU/L and INR 1.0. Tenofovir alafenamide was started because of the fibrosis stage together with a detectable viral load; this is his first antiviral therapy for hepatitis B. Hepatocellular carcinoma surveillance with six-monthly ultrasound was arranged.",
         "labs": {"Alanine Aminotransferase": [("2024-03-19 07:20", 52)], "INR": [("2024-03-19 07:20", 1.0)],
                  "Total Bilirubin": [("2024-03-19 07:20", 0.8)], "Albumin": [("2024-03-19 07:20", 4.3)],
                  "Platelet Count": [("2024-03-19 07:20", 186)]},
         "dx": [("B18.1", "Chronic viral hepatitis B without delta-agent"), ("K74.00", "Hepatic fibrosis, unspecified"),
                ("Z86.15", "Personal history of latent tuberculosis infection")],
         "meds": [M("Tenofovir alafenamide", "25 mg", "PO", "daily", "new", "first antiviral therapy for hepatitis B")],
         "followup": "Hepatology in 3 months with liver function tests, hepatitis B viral load, and kidney function. Six-monthly ultrasound surveillance."},
     ]},

    # ---- AE: severe aortic stenosis, syncope then elective TAVR -----------
    {"n": 31, "sex": "F", "age": 81, "pmh_short": "severe aortic stenosis and hypertension",
     "pmh": ["Severe aortic stenosis", "Hypertension", "Chronic kidney disease stage 3a", "Osteoarthritis"],
     "social": "Retired teacher. Lives with her son. Never smoker. No alcohol use.",
     "base": {"Sodium": 138, "Potassium": 4.3, "Urea Nitrogen": 28, "Hemoglobin": 11.9, "White Blood Cells": 6.6, "Glucose": 102, "Platelet Count": 218},
     "admissions": [
        {"admit": "2023-07-25 10:45", "disch": "2023-07-28 13:00", "service": "Cardiology", "atype": "Emergency",
         "cc": "Blackout while walking uphill.",
         "hpi": "She reports a witnessed loss of consciousness for about twenty seconds while walking uphill, with rapid full recovery. She has had six months of exertional breathlessness and chest tightness.",
         "vitals": "temperature 36.6 C, blood pressure 126/78, heart rate 74, respiratory rate 16, oxygen saturation 97% on room air",
         "exam": "A harsh late-peaking ejection systolic murmur radiating to the carotids with a soft second heart sound. Slow-rising pulse. No edema.",
         "course": "Admitted with exertional syncope from severe aortic stenosis. Echocardiogram showed a peak velocity of 4.6 m/s, a mean gradient of 52 mmHg, and a valve area of 0.7 cm2, with preserved ejection fraction at 60%. Telemetry showed no arrhythmia. Coronary angiography showed non-obstructive disease. She was referred to the structural heart team and accepted for transcatheter aortic valve replacement as an elective procedure. Amlodipine was reduced from 10 mg to 5 mg to avoid preload reduction, and she was advised to avoid strenuous exertion and dehydration while waiting.",
         "labs": {"Creatinine": [("2023-07-25 11:10", 1.3), ("2023-07-28 06:00", 1.3)],
                  "NT-proBNP": [("2023-07-25 11:10", 2140)],
                  "Troponin T": [("2023-07-25 11:10", 0.03), ("2023-07-25 17:00", 0.03)],
                  "Hemoglobin": [("2023-07-25 11:10", 11.8)]},
         "dx": [("I35.0", "Nonrheumatic aortic (valve) stenosis"), ("R55", "Syncope and collapse"),
                ("N18.31", "Chronic kidney disease, stage 3a"), ("I10", "Essential hypertension")],
         "meds": [M("Amlodipine", "5 mg", "PO", "daily", "changed", "reduced from 10 mg daily"),
                  M("Atorvastatin", "20 mg", "PO", "nightly"), M("Acetaminophen", "1 g", "PO", "three times daily as needed for joint pain")],
         "followup": "Structural heart clinic in 2 weeks for transcatheter aortic valve replacement planning. Avoid strenuous exertion. Return for further blackout, chest pain, or breathlessness at rest."},
        {"admit": "2023-10-17 06:30", "disch": "2023-10-20 11:45", "service": "Cardiology", "atype": "Elective",
         "cc": "Planned transcatheter aortic valve replacement.",
         "hpi": "She was admitted for a planned transcatheter aortic valve replacement, twelve weeks after the syncope admission. She has had no further blackouts but exertional breathlessness has progressed.",
         "exam": "Ejection systolic murmur unchanged. No edema. Good femoral pulses bilaterally.",
         "course": "Underwent an uncomplicated transfemoral transcatheter aortic valve replacement with a balloon-expandable valve. Post-procedure echocardiogram showed a mean gradient of 9 mmHg, down from 52 mmHg in July, with trivial paravalvular leak. Creatinine rose from 1.3 to 1.6 mg/dL after contrast and improved to 1.4 mg/dL with hydration. There was no conduction block requiring a permanent pacemaker and telemetry was normal for 48 hours. Aspirin 81 mg daily was started as single antiplatelet therapy. She mobilised independently and was discharged home.",
         "labs": {"Creatinine": [("2023-10-17 06:00", 1.3), ("2023-10-18 06:00", 1.6), ("2023-10-20 06:00", 1.4)],
                  "Hemoglobin": [("2023-10-17 06:00", 11.7), ("2023-10-20 06:00", 10.9)],
                  "NT-proBNP": [("2023-10-20 06:00", 980)]},
         "dx": [("I35.0", "Nonrheumatic aortic (valve) stenosis"), ("Z95.2", "Presence of prosthetic heart valve"),
                ("N18.31", "Chronic kidney disease, stage 3a"), ("N17.9", "Acute kidney injury")],
         "inpatient": [M("Sodium chloride 0.9%", "1 mL", "IV", "per kg per hour for 12 hours before and after contrast")],
         "meds": [M("Aspirin", "81 mg", "PO", "daily", "new", "single antiplatelet after valve replacement"),
                  M("Amlodipine", "5 mg", "PO", "daily"), M("Atorvastatin", "20 mg", "PO", "nightly"),
                  M("Acetaminophen", "1 g", "PO", "three times daily as needed for joint pain")],
         "followup": "Structural heart clinic in 4 weeks with an echocardiogram. Basic metabolic panel in 1 week to confirm kidney function has recovered. Endocarditis prophylaxis card issued."},
     ]},

    # ---- AF: cellulitis with an unresolved allergy discrepancy -------------
    {"n": 32, "sex": "M", "age": 57, "pmh_short": "type 2 diabetes mellitus and chronic venous insufficiency",
     "pmh": ["Type 2 diabetes mellitus", "Chronic venous insufficiency with lower leg edema", "Obesity"],
     "social": "Works as a security guard, mostly standing. Former smoker, quit in 2012. No alcohol use.",
     "base": {"Sodium": 137, "Potassium": 4.4, "Urea Nitrogen": 18, "Hemoglobin": 13.6, "White Blood Cells": 10.2, "Glucose": 158, "Platelet Count": 284},
     "admissions": [
        {"admit": "2024-04-11 19:40", "disch": "2024-04-15 12:20", "service": "Medicine", "atype": "Urgent",
         "cc": "A red, painful left lower leg.",
         "hpi": "He reports four days of spreading redness and pain of the left lower leg with fever at home. He has chronic swelling of both legs and a history of similar episodes.",
         "vitals": "temperature 38.3 C, blood pressure 134/80, heart rate 100, respiratory rate 18, oxygen saturation 96% on room air",
         "exam": "Confluent warm erythema of the left lower leg from ankle to knee with a demarcated border. Chronic bilateral pitting edema and hemosiderin staining. No fluctuance, crepitus, or ulcer.",
         "course": "Admitted with cellulitis of the left lower leg on a background of chronic venous insufficiency. The allergy documentation is not consistent: the triage record and the electronic allergy list state a penicillin allergy with an unspecified reaction, while he reports taking amoxicillin for a dental infection last year without any problem and cannot recall the original reaction. Because the reaction history could not be clarified during the admission, he was treated with vancomycin rather than a beta-lactam, and an allergy clinic referral was made to resolve the discrepancy. This ambiguity is unresolved at discharge and should not be treated as a confirmed allergy or as a confirmed tolerance without formal testing. Erythema regressed and he was discharged on oral doxycycline to complete a ten-day course. Glucose ran between 180 and 260 mg/dL and insulin glargine was increased from 20 to 26 units nightly.",
         "labs": {"White Blood Cells": [("2024-04-11 20:05", 15.8), ("2024-04-15 06:00", 9.4)],
                  "C-Reactive Protein": [("2024-04-11 20:05", 128), ("2024-04-15 06:00", 41)],
                  "Glucose": [("2024-04-11 20:05", 262), ("2024-04-15 06:00", 184)],
                  "Hemoglobin A1c": [("2024-04-12 06:00", 8.6)],
                  "Creatinine": [("2024-04-11 20:05", 1.1), ("2024-04-15 06:00", 1.0)]},
         "dx": [("L03.116", "Cellulitis of left lower limb"), ("I87.2", "Venous insufficiency (chronic) (peripheral)"),
                ("E11.65", "Type 2 diabetes mellitus with hyperglycemia"), ("Z88.0", "Allergy status to penicillin")],
         "inpatient": [M("Vancomycin", "1250 mg", "IV", "every 12 hours")],
         "meds": [M("Doxycycline", "100 mg", "PO", "twice daily", "new", "complete a 10-day total course"),
                  M("Insulin glargine", "26 units", "SC", "nightly", "changed", "increased from 20 units nightly"),
                  M("Metformin", "1000 mg", "PO", "twice daily"),
                  M("Compression stockings", "20-30 mmHg", "TD", "daily during waking hours", "new")],
         "progress": [PROG("2024-04-13 09:40", 3,
                           "Afebrile overnight. Pain is better and he is walking to the bathroom unaided.",
                           "Erythema has receded 4 cm below the marked border. No fluctuance. Bilateral chronic edema unchanged.",
                           "1. Left lower leg cellulitis: improving on vancomycin; plan oral step down at 48 hours of clinical improvement.\n2. Penicillin allergy label: the recorded allergy conflicts with his reported tolerance of amoxicillin last year. We are not de-labelling on this admission; allergy clinic referral placed.\n3. Type 2 diabetes with hyperglycemia: glucose 180 to 260 mg/dL; glargine increased.\n4. Chronic venous insufficiency: compression and leg elevation once the cellulitis settles.",
                           labs=["Glucose"])],
         "followup": "Primary care in 1 week. Allergy clinic to clarify the penicillin allergy label before the next antibiotic course. Compression stockings daily and elevate the legs when sitting."},
     ]},

    # ---- AG: breast cancer on chemotherapy, febrile neutropenia -----------
    {"n": 33, "sex": "F", "age": 51, "pmh_short": "stage II hormone receptor positive breast cancer on adjuvant chemotherapy",
     "pmh": ["Stage II invasive ductal carcinoma of the left breast, hormone receptor positive, HER2 negative",
             "Left lumpectomy with sentinel node biopsy in 2023", "Hypothyroidism"],
     "social": "Works as an accountant, currently on reduced hours. Never smoker. No alcohol use during chemotherapy.",
     "base": {"Sodium": 138, "Potassium": 4.0, "Urea Nitrogen": 12, "Hemoglobin": 11.2, "White Blood Cells": 4.4, "Glucose": 98, "Platelet Count": 198},
     "admissions": [
        {"admit": "2023-04-27 23:15", "disch": "2023-05-02 14:00", "service": "Oncology", "atype": "Emergency",
         "cc": "Fever eight days after chemotherapy.",
         "hpi": "She reports a temperature of 38.6 C at home, eight days after the second cycle of doxorubicin and cyclophosphamide. She has a mild sore throat but no cough, dysuria, or diarrhea.",
         "vitals": "temperature 38.6 C, blood pressure 108/64, heart rate 112, respiratory rate 20, oxygen saturation 98% on room air",
         "exam": "Mild pharyngeal erythema. Chest clear. Port site clean and non-tender. No rash.",
         "course": "Admitted with febrile neutropenia. The absolute neutrophil count was 0.2 K/uL on admission, the lowest recorded for her. Empiric cefepime was started within 45 minutes of arrival and blood cultures from the port and a peripheral site were negative. Filgrastim was given daily until the neutrophil count recovered to 1.8 K/uL on day 5. She defervesced on day 3 and no source was identified. Chemotherapy was delayed by one week and primary filgrastim prophylaxis was added to subsequent cycles.",
         "labs": {"Absolute Neutrophil Count": [("2023-04-27 23:40", 0.2), ("2023-04-30 06:00", 0.6), ("2023-05-02 06:00", 1.8)],
                  "White Blood Cells": [("2023-04-27 23:40", 0.8), ("2023-05-02 06:00", 3.4)],
                  "Hemoglobin": [("2023-04-27 23:40", 10.4), ("2023-05-02 06:00", 10.1)],
                  "Platelet Count": [("2023-04-27 23:40", 128)],
                  "Lactate": [("2023-04-27 23:40", 1.4)]},
         "dx": [("D70.1", "Agranulocytosis secondary to cancer chemotherapy"), ("R50.81", "Fever presenting with conditions classified elsewhere"),
                ("C50.912", "Malignant neoplasm of unspecified site of left female breast"), ("E03.9", "Hypothyroidism, unspecified")],
         "inpatient": [M("Cefepime", "2 g", "IV", "every 8 hours"), M("Filgrastim", "300 mcg", "SC", "daily")],
         "meds": [M("Filgrastim", "300 mcg", "SC", "daily for 5 days after each cycle", "new", "primary prophylaxis from cycle 3"),
                  M("Levothyroxine", "88 mcg", "PO", "daily"),
                  M("Ondansetron", "8 mg", "PO", "twice daily as needed for nausea")],
         "followup": "Oncology in 5 days to restart chemotherapy. Take a temperature twice daily and attend immediately for a temperature at or above 38 C."},
        {"admit": "2023-06-14 20:30", "disch": "2023-06-16 12:15", "service": "Oncology", "atype": "Urgent",
         "cc": "Fever after the fourth chemotherapy cycle.",
         "hpi": "She reports a temperature of 38.2 C nine days after the fourth cycle, despite filgrastim prophylaxis. She feels better than during the April admission.",
         "vitals": "temperature 38.2 C, blood pressure 116/70, heart rate 96, respiratory rate 18, oxygen saturation 99% on room air",
         "exam": "Well appearing. Chest clear. Port site clean. No focal infection identified.",
         "course": "Second episode of neutropenic fever, milder than April. The absolute neutrophil count was 0.9 K/uL, higher than the 0.2 K/uL in April, reflecting the benefit of prophylactic filgrastim. She met low-risk criteria, received one dose of intravenous piperacillin-tazobactam, and was switched to oral ciprofloxacin with amoxicillin-clavulanate after 24 afebrile hours. Cultures were negative. The chemotherapy dose was reduced by 20% for the remaining cycles.",
         "labs": {"Absolute Neutrophil Count": [("2023-06-14 20:50", 0.9), ("2023-06-16 06:00", 1.6)],
                  "White Blood Cells": [("2023-06-14 20:50", 2.2), ("2023-06-16 06:00", 3.1)],
                  "Hemoglobin": [("2023-06-14 20:50", 10.0)],
                  "C-Reactive Protein": [("2023-06-14 20:50", 42), ("2023-06-16 06:00", 18)]},
         "dx": [("D70.1", "Agranulocytosis secondary to cancer chemotherapy"), ("R50.81", "Fever presenting with conditions classified elsewhere"),
                ("C50.912", "Malignant neoplasm of unspecified site of left female breast")],
         "inpatient": [M("Piperacillin-tazobactam", "4.5 g", "IV", "once")],
         "meds": [M("Ciprofloxacin", "500 mg", "PO", "twice daily", "new", "complete a 7-day total course"),
                  M("Amoxicillin-clavulanate", "875 mg", "PO", "twice daily", "new", "complete a 7-day total course"),
                  M("Filgrastim", "300 mcg", "SC", "daily for 5 days after each cycle"),
                  M("Levothyroxine", "88 mcg", "PO", "daily")],
         "followup": "Oncology in 4 days. Chemotherapy will continue at a 20% reduced dose."},
        {"admit": "2024-02-27 09:00", "disch": "2024-02-28 11:30", "service": "Oncology", "atype": "Elective",
         "cc": "Planned port removal and survivorship review.",
         "hpi": "She was admitted for planned removal of her implanted port after completing chemotherapy and radiotherapy. She has had no fever since June 2023.",
         "exam": "Port site healed. Left breast without recurrence. No lymphedema.",
         "course": "Elective removal of the implanted venous port under local anaesthetic without complication. The neutrophil count has fully recovered at 3.6 K/uL, compared with 0.2 K/uL during the April 2023 febrile neutropenia admission. Anastrozole was started for adjuvant endocrine therapy and a bone density scan was arranged given the aromatase inhibitor. Filgrastim was stopped as chemotherapy is complete.",
         "labs": {"Absolute Neutrophil Count": [("2024-02-27 08:20", 3.6)], "White Blood Cells": [("2024-02-27 08:20", 5.8)],
                  "Hemoglobin": [("2024-02-27 08:20", 12.4)], "Calcium": [("2024-02-27 08:20", 9.3)]},
         "dx": [("C50.912", "Malignant neoplasm of unspecified site of left female breast"), ("Z45.2", "Encounter for adjustment and management of vascular access device"),
                ("E03.9", "Hypothyroidism, unspecified")],
         "meds": [M("Anastrozole", "1 mg", "PO", "daily", "new", "adjuvant endocrine therapy for 5 years"),
                  M("Levothyroxine", "88 mcg", "PO", "daily"),
                  M("Cholecalciferol", "1000 units", "PO", "daily", "new", "bone protection on an aromatase inhibitor")],
         "stopped": [("Filgrastim", "chemotherapy complete"), ("Ondansetron", "chemotherapy complete")],
         "followup": "Oncology in 3 months. Bone density scan before the next visit. Report joint pain or hot flushes on anastrozole."},
     ]},

    # ---- AH: new-onset seizures, then breakthrough seizure ----------------
    {"n": 34, "sex": "M", "age": 31, "pmh_short": "newly diagnosed focal epilepsy",
     "pmh": ["Focal epilepsy with secondary generalisation, diagnosed 2023", "Migraine with aura"],
     "social": "Works as a warehouse picker; driving licence suspended after the first seizure. Never smoker. Drinks alcohol at weekends.",
     "base": {"Sodium": 139, "Potassium": 4.1, "Urea Nitrogen": 12, "Hemoglobin": 15.0, "White Blood Cells": 8.8, "Glucose": 98, "Platelet Count": 246},
     "admissions": [
        {"admit": "2023-01-24 05:15", "disch": "2023-01-27 13:40", "service": "Neurology", "atype": "Emergency",
         "cc": "First witnessed seizure.",
         "hpi": "His partner witnessed a generalised convulsion lasting about two minutes followed by thirty minutes of confusion. He had been sleeping poorly and had drunk alcohol the night before.",
         "vitals": "temperature 37.2 C, blood pressure 132/78, heart rate 92, respiratory rate 16, oxygen saturation 98% on room air",
         "exam": "Drowsy on arrival, fully alert within two hours. Tongue laceration on the left lateral border. No focal deficit.",
         "course": "Admitted after a first unprovoked seizure. MRI brain showed a small area of focal cortical dysplasia in the left frontal lobe, and the electroencephalogram showed left frontal epileptiform discharges, so the risk of recurrence was considered high and levetiracetam 500 mg twice daily was started as his first antiseizure medication. Creatine kinase and lactate were mildly raised on arrival and normalised. He was counselled about driving restrictions, shift work, sleep, and alcohol.",
         "labs": {"Lactate": [("2023-01-24 05:30", 4.2), ("2023-01-24 12:00", 1.3)],
                  "Sodium": [("2023-01-24 05:30", 138), ("2023-01-27 06:00", 139)],
                  "Glucose": [("2023-01-24 05:30", 118)],
                  "Magnesium": [("2023-01-24 05:30", 1.9)]},
         "dx": [("G40.209", "Localization-related symptomatic epilepsy with complex partial seizures, not intractable"),
                ("Q04.3", "Other reduction deformities of brain"), ("G43.109", "Migraine with aura, not intractable")],
         "meds": [M("Levetiracetam", "500 mg", "PO", "twice daily", "new", "first antiseizure medication"),
                  M("Sumatriptan", "50 mg", "PO", "as needed for migraine")],
         "imaging": [IMG("MRI brain with epilepsy protocol", "2023-01-25 11:20", "First unprovoked seizure.", "None available.",
                         "There is a small region of cortical thickening with blurring of the grey-white junction in the left superior frontal gyrus measuring 1.4 cm. No mass, hemorrhage, or abnormal enhancement. The hippocampi are symmetric.",
                         "Findings consistent with focal cortical dysplasia in the left superior frontal gyrus.")],
         "followup": "Neurology in 4 weeks. Do not drive; notify the licensing authority. Avoid sleep deprivation and binge drinking. Do not work at height."},
        {"admit": "2024-06-09 02:40", "disch": "2024-06-12 11:00", "service": "Neurology", "atype": "Emergency",
         "cc": "Two seizures in one night.",
         "hpi": "He had two generalised seizures four hours apart after missing several days of levetiracetam while travelling. This is his first recurrence since the 2023 diagnosis.",
         "vitals": "temperature 37.0 C, blood pressure 138/84, heart rate 98, respiratory rate 18, oxygen saturation 97% on room air",
         "exam": "Post-ictal drowsiness resolving. No focal deficit. No tongue injury this time.",
         "course": "Readmitted with breakthrough seizures after missed doses. The levetiracetam level was subtherapeutic on arrival. Levetiracetam was increased from 500 mg to 1000 mg twice daily and lamotrigine was added with a slow titration because of ongoing breakthrough risk; he was counselled that the lamotrigine rash risk requires the titration schedule to be followed exactly. MRI was unchanged from 2023. He was seizure free for 48 hours before discharge and epilepsy specialist nurse follow-up was arranged for adherence support.",
         "labs": {"Lactate": [("2024-06-09 03:00", 3.6), ("2024-06-09 09:00", 1.2)],
                  "Sodium": [("2024-06-09 03:00", 137), ("2024-06-12 06:00", 139)],
                  "Alanine Aminotransferase": [("2024-06-09 03:00", 34)]},
         "dx": [("G40.209", "Localization-related symptomatic epilepsy with complex partial seizures, not intractable"),
                ("Z91.14", "Patient's other noncompliance with medication regimen"), ("Q04.3", "Other reduction deformities of brain")],
         "meds": [M("Levetiracetam", "1000 mg", "PO", "twice daily", "changed", "increased from 500 mg twice daily"),
                  M("Lamotrigine", "25 mg", "PO", "daily", "new", "increase by 25 mg every 2 weeks to 100 mg twice daily"),
                  M("Sumatriptan", "50 mg", "PO", "as needed for migraine")],
         "progress": [PROG("2024-06-11 08:30", 3,
                           "No further seizures for 48 hours. He is alert and eating normally, and reports no rash.",
                           "Neurologic examination normal. No injuries.",
                           "1. Breakthrough focal epilepsy with secondary generalisation after missed doses: levetiracetam increased to 1000 mg twice daily and lamotrigine added at 25 mg daily.\n2. Adherence: epilepsy nurse to arrange a pill reminder; discuss a long-acting option at clinic.\n3. Driving: restart the licence clock from this seizure and inform the licensing authority.")],
         "followup": "Neurology in 6 weeks. Follow the lamotrigine titration exactly and stop it and seek care immediately if a rash develops. Do not drive."},
     ]},

    # ---- AI: hyperglycemic crisis then overcorrection, improving control --
    {"n": 35, "sex": "M", "age": 59, "pmh_short": "type 2 diabetes mellitus with poor glycemic control",
     "pmh": ["Type 2 diabetes mellitus diagnosed 2011", "Hypertension", "Hyperlipidemia", "Obesity"],
     "social": "Works night shifts in a distribution centre, which makes meal timing irregular. Never smoker. No alcohol use.",
     "base": {"Sodium": 136, "Potassium": 4.4, "Urea Nitrogen": 22, "Hemoglobin": 14.0, "White Blood Cells": 8.4, "Glucose": 212, "Platelet Count": 254},
     "admissions": [
        {"admit": "2023-03-02 01:30", "disch": "2023-03-07 15:00", "service": "Medicine", "atype": "Emergency",
         "cc": "Confusion, extreme thirst, and frequent urination.",
         "hpi": "His family reports five days of thirst, frequent urination, and progressive confusion. He had stopped taking his diabetes medication three months earlier when he changed jobs and lost his prescription coverage.",
         "vitals": "temperature 37.3 C, blood pressure 104/62, heart rate 118, respiratory rate 22, oxygen saturation 97% on room air",
         "exam": "Dry mucous membranes with poor skin turgor. Drowsy, oriented to person and place only. No focal deficit. No ketotic breath.",
         "course": "Admitted with a hyperosmolar hyperglycemic state. Glucose was 812 mg/dL with a serum osmolality of 328 and only trace ketones, and sodium was 128 mEq/L before correction for glucose. Treated with an intravenous insulin infusion and 6 liters of crystalloid over the first 24 hours, with potassium replacement. Mental status normalised within 12 hours and glucose was 168 mg/dL by discharge. Hemoglobin A1c was 13.2%. He was transitioned to insulin glargine 30 units nightly with mealtime insulin lispro, and metformin was restarted. A social worker arranged medication coverage before discharge.",
         "labs": {"Glucose": [("2023-03-02 01:45", 812), ("2023-03-03 06:00", 244), ("2023-03-07 06:00", 168)],
                  "Hemoglobin A1c": [("2023-03-02 01:45", 13.2)],
                  "Sodium": [("2023-03-02 01:45", 128), ("2023-03-07 06:00", 138)],
                  "Potassium": [("2023-03-02 01:45", 5.1), ("2023-03-03 06:00", 3.4), ("2023-03-07 06:00", 4.0)],
                  "Creatinine": [("2023-03-02 01:45", 1.9), ("2023-03-07 06:00", 1.1)],
                  "Bicarbonate": [("2023-03-02 01:45", 20), ("2023-03-07 06:00", 25)]},
         "dx": [("E11.00", "Type 2 diabetes mellitus with hyperosmolarity without nonketotic hyperglycemic-hyperosmolar coma"),
                ("N17.9", "Acute kidney injury"), ("E87.1", "Hypo-osmolality and hyponatremia"), ("I10", "Essential hypertension")],
         "inpatient": [M("Insulin regular", "0.1 units", "IV", "per kg per hour by infusion"),
                       M("Sodium chloride 0.9%", "1000 mL", "IV", "per hour for the first 2 hours then reassessed")],
         "meds": [M("Insulin glargine", "30 units", "SC", "nightly", "new", "first basal insulin"),
                  M("Insulin lispro", "8 units", "SC", "three times daily with meals", "new"),
                  M("Metformin", "1000 mg", "PO", "twice daily", "changed", "restarted after 3 months off"),
                  M("Lisinopril", "20 mg", "PO", "daily"), M("Atorvastatin", "40 mg", "PO", "nightly")],
         "progress": [PROG("2023-03-04 10:00", 3,
                           "Fully oriented since yesterday evening and eating a diabetic diet. Urine output is good.",
                           "Mucous membranes moist. Heart rate 88, blood pressure 118/70.",
                           "1. Hyperosmolar hyperglycemic state: resolved; off the insulin infusion and on subcutaneous basal-bolus insulin with overlap.\n2. Acute kidney injury from volume depletion: creatinine 1.3 mg/dL from 1.9 mg/dL; continue fluids.\n3. Hypokalemia during insulin therapy: potassium 3.4 mEq/L, replaced; recheck twice daily.\n4. Access to medication: social work engaged to restore prescription coverage before discharge.",
                           labs=["Glucose", "Potassium"])],
         "followup": "Endocrinology in 1 week. Diabetes educator before then for insulin technique and hypoglycemia recognition. Check glucose four times daily."},
        {"admit": "2023-12-05 04:10", "disch": "2023-12-07 12:30", "service": "Medicine", "atype": "Emergency",
         "cc": "Found sweaty and confused in the early morning.",
         "hpi": "His wife found him sweaty, shaking, and confused at 03:00. A finger-stick glucose was 38 mg/dL. He had increased his glargine himself after reading high evening numbers and had skipped his night-shift meal.",
         "vitals": "temperature 36.4 C, blood pressure 128/76, heart rate 96, respiratory rate 16, oxygen saturation 98% on room air",
         "exam": "Diaphoretic on arrival, alert after treatment. No focal deficit. No injury from the episode.",
         "course": "Admitted with severe hypoglycemia after self-titration of basal insulin combined with missed meals on night shifts. Glucose was 38 mg/dL on arrival and responded to intravenous dextrose, with a further mild episode overnight requiring a dextrose infusion. Insulin glargine was reduced from 38 units, which he had reached on his own, to 24 units nightly, and mealtime lispro was reduced and converted to a fixed dose with a meal rule for night shifts. Hemoglobin A1c had improved from 13.2% to 7.4% since March, showing overtreatment rather than undertreatment. Continuous glucose monitoring was arranged.",
         "labs": {"Glucose": [("2023-12-05 04:20", 38), ("2023-12-05 10:00", 126), ("2023-12-07 06:00", 142)],
                  "Hemoglobin A1c": [("2023-12-05 06:00", 7.4)],
                  "Potassium": [("2023-12-05 04:20", 3.8)],
                  "Creatinine": [("2023-12-05 04:20", 1.1)]},
         "dx": [("E11.649", "Type 2 diabetes mellitus with hypoglycemia without coma"), ("E16.2", "Hypoglycemia, unspecified"),
                ("I10", "Essential hypertension")],
         "inpatient": [M("Dextrose 50%", "25 g", "IV", "as needed for glucose below 70 mg/dL")],
         "meds": [M("Insulin glargine", "24 units", "SC", "nightly", "changed", "reduced from 38 units nightly after severe hypoglycemia"),
                  M("Insulin lispro", "5 units", "SC", "three times daily with meals", "changed", "reduced from 8 units; hold if a meal is skipped"),
                  M("Metformin", "1000 mg", "PO", "twice daily"), M("Lisinopril", "20 mg", "PO", "daily"),
                  M("Atorvastatin", "40 mg", "PO", "nightly")],
         "followup": "Endocrinology in 5 days. Continuous glucose monitor fitting this week. Do not change insulin doses without contacting the diabetes team."},
        {"admit": "2024-08-14 10:20", "disch": "2024-08-15 11:40", "service": "Endocrinology", "atype": "Elective",
         "cc": "Planned insulin pump education admission.",
         "hpi": "He was admitted for a planned two-day start of insulin pump therapy after a year of variable control on night shifts. He has had no severe hypoglycemia since December 2023.",
         "exam": "Well appearing. Body mass index 33. No injection site lipohypertrophy.",
         "course": "Elective structured start of continuous subcutaneous insulin infusion with diabetes educator supervision. Hemoglobin A1c was 6.8%, improved from 7.4% in December 2023 and 13.2% in March 2023, with no severe hypoglycemia in the preceding eight months. Glargine and lispro injections were stopped when the pump was started. He demonstrated correct infusion set changes and sick day rules before discharge.",
         "labs": {"Hemoglobin A1c": [("2024-08-14 09:40", 6.8)], "Glucose": [("2024-08-14 09:40", 134), ("2024-08-15 06:00", 118)],
                  "Creatinine": [("2024-08-14 09:40", 1.0)]},
         "dx": [("E11.9", "Type 2 diabetes mellitus without complications"), ("Z46.81", "Encounter for fitting and adjustment of insulin pump"),
                ("I10", "Essential hypertension")],
         "meds": [M("Insulin lispro by pump", "0.9 units", "SC", "per hour basal with mealtime boluses", "new", "continuous subcutaneous insulin infusion"),
                  M("Metformin", "1000 mg", "PO", "twice daily"), M("Lisinopril", "20 mg", "PO", "daily"),
                  M("Atorvastatin", "40 mg", "PO", "nightly")],
         "stopped": [("Insulin glargine", "replaced by pump basal insulin"), ("Insulin lispro", "replaced by pump bolus insulin")],
         "followup": "Diabetes clinic in 2 weeks for pump settings review. Keep long-acting insulin at home for pump failure and follow the written sick day plan."},
     ]},

    # ---- AJ: bariatric surgery then micronutrient deficiency --------------
    {"n": 36, "sex": "F", "age": 44, "pmh_short": "severe obesity treated with sleeve gastrectomy",
     "pmh": ["Severe obesity, body mass index 44 before surgery", "Obstructive sleep apnea on CPAP",
             "Type 2 diabetes mellitus, in remission after surgery", "Gastroesophageal reflux disease"],
     "social": "Works as a nurse on a surgical ward. Never smoker. Stopped alcohol before surgery.",
     "base": {"Sodium": 139, "Potassium": 4.1, "Urea Nitrogen": 12, "Hemoglobin": 12.2, "White Blood Cells": 7.4, "Glucose": 118, "Platelet Count": 276},
     "admissions": [
        {"admit": "2023-05-16 06:20", "disch": "2023-05-19 11:30", "service": "Surgery", "atype": "Elective",
         "cc": "Planned sleeve gastrectomy.",
         "hpi": "She was admitted for a planned laparoscopic sleeve gastrectomy after completing a six-month preoperative programme, with a preoperative weight of 121 kg and a body mass index of 44.",
         "exam": "Well appearing. Abdomen obese and soft. No hernia.",
         "course": "Underwent uncomplicated laparoscopic sleeve gastrectomy. A leak test was negative intraoperatively and a contrast study on postoperative day 1 showed no leak. She progressed through the bariatric fluid diet and was discharged on day 3. Hemoglobin A1c before surgery was 7.0% and metformin was stopped at discharge with a plan to monitor rather than treat. Lifelong multivitamin, vitamin B12, calcium, and iron supplementation was started and she was told that missing these can cause serious deficiency.",
         "labs": {"Hemoglobin A1c": [("2023-05-16 05:40", 7.0)],
                  "Hemoglobin": [("2023-05-16 05:40", 12.4), ("2023-05-19 06:00", 11.8)],
                  "Vitamin B12": [("2023-05-16 05:40", 388)], "Ferritin": [("2023-05-16 05:40", 44)],
                  "Albumin": [("2023-05-16 05:40", 4.0)], "Glucose": [("2023-05-16 05:40", 128), ("2023-05-19 06:00", 104)]},
         "dx": [("E66.01", "Morbid (severe) obesity due to excess calories"), ("Z98.84", "Bariatric surgery status"),
                ("G47.33", "Obstructive sleep apnea"), ("E11.9", "Type 2 diabetes mellitus without complications")],
         "inpatient": [M("Enoxaparin", "40 mg", "SC", "every 12 hours"), M("Cefazolin", "2 g", "IV", "once before incision")],
         "meds": [M("Bariatric multivitamin", "1 tablet", "PO", "twice daily", "new", "lifelong after sleeve gastrectomy"),
                  M("Cyanocobalamin", "1000 mcg", "PO", "daily", "new", "lifelong vitamin B12 supplementation"),
                  M("Calcium citrate with vitamin D", "500 mg", "PO", "three times daily", "new"),
                  M("Ferrous sulfate", "325 mg", "PO", "daily", "new"),
                  M("Pantoprazole", "40 mg", "PO", "daily", "new", "for 6 months after sleeve gastrectomy")],
         "stopped": [("Metformin", "stopped after surgery; monitor glucose rather than treat")],
         "followup": "Bariatric surgery in 2 weeks, dietitian in 4 weeks. Blood tests at 3, 6, and 12 months including vitamin B12, iron studies, and vitamin D. Take the supplements every day for life."},
        {"admit": "2024-04-03 15:45", "disch": "2024-04-06 13:10", "service": "Medicine", "atype": "Urgent",
         "cc": "Tiredness, numb feet, and breathlessness on climbing stairs.",
         "hpi": "She reports three months of progressive fatigue, tingling and numbness in both feet, and breathlessness on exertion. She stopped taking her bariatric supplements around six months after surgery because of cost and nausea, and missed her 6-month and 12-month blood tests.",
         "vitals": "temperature 36.6 C, blood pressure 108/64, heart rate 98, respiratory rate 18, oxygen saturation 98% on room air",
         "exam": "Pale conjunctivae. Reduced vibration sense to the mid shin bilaterally with absent ankle reflexes. Weight 79 kg, down 42 kg from before surgery.",
         "course": "Admitted eleven months after sleeve gastrectomy with combined vitamin B12 and iron deficiency after stopping supplements. Vitamin B12 was 96 pg/mL, down from 388 pg/mL before surgery, with a macrocytic anemia; hemoglobin was 8.6 g/dL and ferritin 6 ng/mL. She received intramuscular hydroxocobalamin loading and intravenous iron, and hemoglobin rose to 9.4 g/dL before discharge. Neurological symptoms are expected to improve slowly and may not fully resolve. Supplements were restarted with a cost-reduced formulation and a pharmacy assistance referral. Hemoglobin A1c was 5.4%, with diabetes in remission after surgery.",
         "labs": {"Vitamin B12": [("2024-04-03 16:10", 96), ("2024-04-06 06:00", 640)],
                  "Hemoglobin": [("2024-04-03 16:10", 8.6), ("2024-04-06 06:00", 9.4)],
                  "Ferritin": [("2024-04-03 16:10", 6)],
                  "Hemoglobin A1c": [("2024-04-03 16:10", 5.4)],
                  "Albumin": [("2024-04-03 16:10", 3.4)]},
         "dx": [("E53.8", "Deficiency of other specified B group vitamins"), ("D51.9", "Vitamin B12 deficiency anemia, unspecified"),
                ("D50.9", "Iron deficiency anemia"), ("Z98.84", "Bariatric surgery status"),
                ("G62.9", "Polyneuropathy, unspecified")],
         "inpatient": [M("Hydroxocobalamin", "1000 mcg", "IM", "every other day for three doses"),
                       M("Ferric carboxymaltose", "1000 mg", "IV", "once")],
         "meds": [M("Hydroxocobalamin", "1000 mcg", "IM", "every 3 months", "changed", "switched from oral because of malabsorption and nonadherence"),
                  M("Bariatric multivitamin", "1 tablet", "PO", "twice daily", "changed", "restarted after 6 months off"),
                  M("Ferrous sulfate", "325 mg", "PO", "daily", "changed", "restarted after 6 months off"),
                  M("Calcium citrate with vitamin D", "500 mg", "PO", "three times daily", "changed", "restarted after 6 months off")],
         "stopped": [("Cyanocobalamin", "oral route replaced by intramuscular hydroxocobalamin")],
         "progress": [PROG("2024-04-05 09:20", 3,
                           "She feels less breathless after the iron infusion. The foot numbness is unchanged, as expected this early.",
                           "Less pale. Reduced vibration sense unchanged. No new findings.",
                           "1. Vitamin B12 deficiency with neuropathy after sleeve gastrectomy: vitamin B12 96 pg/mL; intramuscular loading given, switch to three-monthly injections.\n2. Iron deficiency anemia: ferritin 6 ng/mL; intravenous iron given, hemoglobin 9.4 g/dL from 8.6 g/dL.\n3. Adherence and cost: pharmacy assistance referral placed; the previous supplement regimen was stopped at around 6 months after surgery.",
                           labs=["Vitamin B12", "Hemoglobin"])],
         "followup": "Bariatric clinic in 4 weeks with repeat vitamin B12, iron studies, and a complete blood count. Neurology if the numbness has not improved by 3 months. Supplements are lifelong."},
     ]},
]


# ===========================================================================
# Golden QA — deterministic expected facts (checked against notes below)
# ===========================================================================
def Q(qid, n, query, category, temporal, facts, min_facts, *, kind, level, answer,
      adms=(), notes=("discharge",), avoid=()):
    """One golden question. `facts` must appear verbatim in that patient's notes (asserted
    in build); `answer` is a short reference answer for future answer-level scoring; `adms`
    are the 1-based admission indices that hold the evidence; `avoid` are strings a correct
    answer must not contain (used for the not-documented questions)."""
    return {"id": qid, "n": n, "query": query, "category": category, "temporal": temporal,
            "expected_facts": list(facts), "min_facts": min_facts, "answer_type": kind,
            "difficulty": level, "expected_answer": answer, "unsupported": kind == "unsupported",
            "must_not_contain": list(avoid), "evidence_admissions": list(adms),
            "evidence_note_types": list(notes)}


GOLDEN = [
    Q("demo_q01", 1, "What was the patient's most recent creatinine?", "labs", "latest",
      ["creatinine 1.4 mg/dL"], 1, kind="latest_value", level="easy", adms=[3],
      answer="1.4 mg/dL, measured on 2024-03-18 at discharge from the third heart failure admission."),
    Q("demo_q02", 1, "How did the patient's creatinine change over time?", "labs", "trend",
      ["creatinine 1.8 mg/dL", "creatinine 2.1 mg/dL", "creatinine 1.4 mg/dL"], 2, kind="trend",
      level="hard", adms=[1, 2, 3], notes=["discharge", "progress"],
      answer="It rose during each diuresis (peak 1.8 mg/dL in January 2023, 2.1 mg/dL in August 2023) "
             "and recovered each time, reaching 1.4 mg/dL at the March 2024 discharge."),
    Q("demo_q03", 1, "What medications was the patient discharged on most recently?", "medications", "latest",
      ["torsemide 20 mg", "sacubitril-valsartan 24-26 mg"], 2, kind="medication_history", level="medium", adms=[3],
      answer="Torsemide 20 mg daily, sacubitril-valsartan 24-26 mg twice daily, metoprolol succinate 100 mg daily, "
             "spironolactone, atorvastatin and aspirin."),
    Q("demo_q04", 1, "Does the patient have a history of heart failure?", "diagnosis", "all",
      ["heart failure with reduced ejection fraction"], 1, kind="diagnosis_history", level="easy", adms=[1, 2, 3],
      answer="Yes. Heart failure with reduced ejection fraction (LVEF 30%) from ischemic cardiomyopathy, with three admissions."),
    Q("demo_q05", 1, "Did the chest x-ray show pulmonary edema?", "imaging", "all",
      ["pulmonary edema"], 1, kind="exact_fact", level="easy", adms=[1], notes=["radiology", "discharge"],
      answer="Yes. The January 2023 chest radiograph showed moderate pulmonary edema with small bilateral effusions."),
    Q("demo_q06", 3, "Why was the patient having trouble breathing?", "plain_language", "all",
      ["pneumonia"], 1, kind="exact_fact", level="easy", adms=[1],
      answer="A right lower lobe community-acquired pneumonia on a background of severe COPD."),
    Q("demo_q07", 2, "How has the creatinine changed across admissions?", "labs", "trend",
      ["creatinine 1.9 mg/dL", "creatinine 2.4 mg/dL", "creatinine 3.1 mg/dL"], 2, kind="trend",
      level="hard", adms=[1, 2, 3],
      answer="It has risen steadily across admissions, from 1.9 mg/dL to 2.4 mg/dL and then 3.1 mg/dL."),
    Q("demo_q08", 2, "What was the most recent hemoglobin A1c?", "labs", "latest",
      ["hemoglobin A1c 7.1%"], 1, kind="latest_value", level="easy", adms=[3],
      answer="7.1%."),
    Q("demo_q09", 3, "Did the right lower lobe pneumonia resolve on follow-up chest imaging?", "imaging", "all",
      ["interval resolution"], 1, kind="temporal_reasoning", level="medium", adms=[1, 2], notes=["radiology"],
      answer="Yes. Follow-up imaging showed interval resolution of the right lower lobe consolidation."),
    Q("demo_q10", 3, "How much supplemental oxygen did the patient require for pneumonia?", "diagnosis", "all",
      ["4 liters"], 1, kind="exact_fact", level="medium", adms=[1],
      answer="Up to 4 liters by nasal cannula."),
    Q("demo_q11", 4, "Is the patient on anticoagulation for atrial fibrillation?", "medications", "all",
      ["apixaban 5 mg"], 1, kind="medication_history", level="easy", adms=[1, 2, 3],
      answer="Yes, apixaban 5 mg twice daily."),
    Q("demo_q12", 4, "How was the rate control medication changed?", "medications", "all",
      ["diltiazem", "metoprolol succinate 50 mg"], 2, kind="medication_history", level="medium", adms=[2, 3],
      answer="Diltiazem extended-release was replaced by metoprolol succinate 50 mg daily."),
    Q("demo_q13", 1, "Was furosemide started before or after the previous admission?", "medications", "all",
      ["not previously been prescribed a loop diuretic", "furosemide was started"], 1, kind="temporal_reasoning",
      level="hard", adms=[1],
      answer="Furosemide was started during the January 2023 admission; he had not been on a loop diuretic before it."),
    Q("demo_q14", 5, "What was the most recent hemoglobin?", "labs", "latest",
      ["hemoglobin 11.2 g/dL"], 1, kind="latest_value", level="easy", adms=[2], notes=["discharge", "progress"],
      answer="11.2 g/dL."),
    Q("demo_q15", 1, "What water pill is the patient taking now?", "plain_language", "all",
      ["torsemide 20 mg"], 1, kind="medication_history", level="medium", adms=[3],
      answer="Torsemide 20 mg daily, which replaced furosemide in March 2024."),

    # ---- expanded set over the wider corpus --------------------------------
    Q("demo_q16", 14, "What was the patient's most recent sodium?", "labs", "latest",
      ["sodium 131 mEq/L"], 1, kind="latest_value", level="medium", adms=[], notes=["progress"],
      answer="131 mEq/L, from the outpatient hepatology visit on 2024-08-06."),
    Q("demo_q17", 14, "Why were the diuretics stopped?", "medications", "all",
      ["spironolactone", "held for acute kidney injury"], 2, kind="medication_history", level="medium", adms=[2],
      answer="Spironolactone and furosemide were held during the September 2023 admission because of acute kidney "
             "injury with spontaneous bacterial peritonitis, and were not restarted at discharge."),
    Q("demo_q18", 14, "How has the total bilirubin changed over time?", "labs", "trend",
      ["total bilirubin 2.4 mg/dL", "total bilirubin 3.6 mg/dL", "total bilirubin 4.1 mg/dL"], 2, kind="trend",
      level="hard", adms=[1, 2, 3],
      answer="It has risen across admissions: 2.4 mg/dL in February 2023, 3.6 mg/dL in September 2023 and "
             "4.1 mg/dL in April 2024, indicating worsening liver function."),
    Q("demo_q19", 15, "Why was the infliximab dose increased in 2024?", "medications", "all",
      ["dose intensified from 5 mg/kg", "insurance interruption"], 2, kind="multi_note_synthesis", level="hard", adms=[3],
      answer="Her infusion schedule was interrupted for five weeks by an insurance problem, the trough level was "
             "undetectable without antibodies, so the dose was intensified from 5 mg/kg to 10 mg/kg every 8 weeks."),
    Q("demo_q20", 16, "What medication replaced clopidogrel and why?", "medications", "all",
      ["apixaban 5 mg", "atrial fibrillation"], 2, kind="medication_history", level="medium", adms=[2],
      answer="Apixaban 5 mg twice daily replaced clopidogrel in November 2023 after paroxysmal atrial fibrillation "
             "was found on the ambulatory monitor."),
    Q("demo_q21", 17, "What was the patient's most recent hemoglobin A1c?", "labs", "latest",
      [], 0, kind="unsupported", level="medium", adms=[], notes=[],
      avoid=["hemoglobin a1c", "hba1c"],
      answer="Not documented. No hemoglobin A1c result appears anywhere in this patient's record."),
    Q("demo_q22", 17, "Which antibiotic should not be used for future urinary infections?", "medications", "all",
      ["cephalexin", "resistant"], 2, kind="multi_note_synthesis", level="hard", adms=[2],
      answer="Cephalexin. The January 2024 organism was an extended-spectrum beta-lactamase producing E. coli and "
             "cephalexin was stopped as resistant."),
    Q("demo_q23", 18, "How much oxygen did the patient need during the acute chest syndrome admission?", "diagnosis", "all",
      ["6 liters of oxygen"], 1, kind="exact_fact", level="medium", adms=[3], notes=["discharge", "progress"],
      answer="Up to 6 liters by nasal cannula, weaned to room air by hospital day 6."),
    Q("demo_q24", 19, "When was the gallbladder removed relative to the pancreatitis admission?", "diagnosis", "all",
      ["laparoscopic cholecystectomy", "2023-07-10"], 2, kind="temporal_reasoning", level="hard", adms=[1, 2],
      answer="About four weeks later: gallstone pancreatitis was treated from 2023-06-11 and the elective "
             "laparoscopic cholecystectomy admission began on 2023-07-10."),
    Q("demo_q25", 20, "What did the lung biopsy show?", "diagnosis", "all",
      ["non-small cell lung cancer", "adenocarcinoma"], 2, kind="diagnosis_history", level="medium", adms=[2],
      answer="Non-small cell lung cancer, adenocarcinoma subtype, with subcarinal nodal involvement, clinical stage IIIA."),
    Q("demo_q26", 20, "Why was the patient's sodium low during the January 2024 admission?", "labs", "all",
      ["syndrome of inappropriate antidiuretic hormone"], 1, kind="multi_note_synthesis", level="hard", adms=[3],
      notes=["discharge", "progress"],
      answer="Hyponatremia attributed to the syndrome of inappropriate antidiuretic hormone secretion in the setting "
             "of progressive lung cancer; sodium was 129 mEq/L and improved to 132 mEq/L with fluid restriction."),
    Q("demo_q27", 22, "How did the hemoglobin A1c change between the two admissions?", "labs", "trend",
      ["hemoglobin A1c 9.8%", "hemoglobin A1c 8.1%"], 2, kind="trend", level="medium", adms=[1, 2],
      answer="It improved from 9.8% in September 2023 to 8.1% in March 2024."),
    Q("demo_q28", 23, "Should the patient restart apixaban after discharge?", "medications", "latest",
      ["apixaban", "on hold pending gastroenterology review"], 2, kind="ambiguous", level="hard", adms=[2],
      notes=["discharge", "progress"],
      answer="No, not yet. The record is inconsistent: the hospital day 3 progress note plans to resume apixaban, "
             "while the discharge summary keeps it on hold pending gastroenterology review at 6 weeks. The discharge "
             "instruction governs, and the conflict should be flagged for a clinician."),
    Q("demo_q29", 24, "Has methotrexate ever been stopped, and why?", "medications", "all",
      ["held for active pneumonia", "restarted at discharge after being held for cellulitis"], 2,
      kind="medication_history", level="hard", adms=[1, 2],
      answer="Twice, both times for infection: held in August 2023 for pneumonia, and held again in May 2024 for "
             "cellulitis before being restarted at 15 mg weekly on the day of that discharge."),
    Q("demo_q30", 25, "Why was the patient readmitted in November 2023?", "diagnosis", "all",
      ["alcohol-associated hepatitis"], 1, kind="temporal_reasoning", level="medium", adms=[2],
      answer="He returned nine days after discharge with jaundice from alcohol-associated hepatitis after resuming "
             "drinking, rather than the isolated alcohol withdrawal of the first admission."),
    Q("demo_q31", 26, "What was the most recent hemoglobin?", "labs", "latest",
      ["hemoglobin 11.2 g/dL"], 1, kind="latest_value", level="medium", adms=[], notes=["progress"],
      answer="11.2 g/dL at the geriatrics follow-up on 2024-01-15, recovered from 8.9 g/dL after hip surgery."),
    Q("demo_q32", 27, "How was the inhaler treatment changed after the second admission?", "medications", "all",
      ["stepped up from 200-6 mcg"], 1, kind="medication_history", level="medium", adms=[2],
      answer="Budesonide-formoterol was stepped up from 200-6 mcg to 400-12 mcg twice daily and montelukast was added."),
    Q("demo_q33", 28, "What happened to the calcium after thyroid surgery?", "labs", "all",
      ["calcium 7.6 mg/dL"], 1, kind="temporal_reasoning", level="medium", adms=[2],
      answer="It fell to 7.6 mg/dL on postoperative day 1 from 9.8 mg/dL in May 2023 (transient hypoparathyroidism) "
             "and improved to 8.4 mg/dL on calcium and calcitriol."),
    Q("demo_q34", 31, "What was the aortic valve gradient before and after the procedure?", "labs", "all",
      ["mean gradient of 52 mmHg", "mean gradient of 9 mmHg"], 2, kind="trend", level="hard", adms=[1, 2],
      answer="A mean gradient of 52 mmHg before transcatheter aortic valve replacement and 9 mmHg afterwards."),
    Q("demo_q35", 32, "Is the patient allergic to penicillin?", "medications", "all",
      ["penicillin allergy", "could not be clarified"], 2, kind="ambiguous", level="hard", adms=[1],
      notes=["discharge", "progress"],
      answer="Unclear and unresolved. The allergy list records a penicillin allergy with an unspecified reaction, but "
             "he reports tolerating amoxicillin last year. It was not clarified during the admission, he was treated "
             "with vancomycin, and an allergy clinic referral was made. This needs clinician review before any "
             "beta-lactam is given."),
    Q("demo_q36", 33, "What was the lowest neutrophil count?", "labs", "all",
      ["absolute neutrophil count 0.2 K/uL"], 1, kind="trend", level="medium", adms=[1],
      answer="0.2 K/uL on 2023-04-27, during the first febrile neutropenia admission."),
    Q("demo_q37", 35, "How has the hemoglobin A1c changed across admissions?", "labs", "trend",
      ["hemoglobin A1c 13.2%", "hemoglobin A1c 7.4%", "hemoglobin A1c 6.8%"], 2, kind="trend", level="hard",
      adms=[1, 2, 3],
      answer="It improved from 13.2% during the hyperosmolar hyperglycemic state in March 2023 to 7.4% in December "
             "2023 and 6.8% in August 2024."),
    Q("demo_q38", 36, "Why did the patient become deficient in vitamin B12?", "diagnosis", "all",
      ["stopped taking her bariatric supplements"], 1, kind="multi_note_synthesis", level="hard", adms=[1, 2],
      notes=["discharge", "progress"],
      answer="She stopped her lifelong post-sleeve-gastrectomy supplements about six months after surgery and missed "
             "her monitoring blood tests; vitamin B12 fell from 388 to 96 pg/mL."),
    Q("demo_q39", 21, "What medications was the patient discharged on?", "medications", "latest",
      ["acetaminophen 1 g", "ibuprofen 400 mg"], 2, kind="medication_history", level="easy", adms=[1],
      answer="Acetaminophen 1 g as needed, a short course of ibuprofen 400 mg as needed, and his usual loratadine."),
    Q("demo_q40", 30, "What treatment was started for hepatitis B?", "medications", "all",
      ["tenofovir alafenamide 25 mg"], 1, kind="exact_fact", level="easy", adms=[1],
      answer="Tenofovir alafenamide 25 mg daily, started after the biopsy showed stage 2 fibrosis."),
]


# ===========================================================================
# Rendering
# ===========================================================================
def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def _fmt(label: str, v) -> str:
    dec = LAB_ITEMS[label][4]
    return f"{v:.{dec}f}" if dec else f"{int(round(v))}"


def _lab_str(label: str, v) -> str:
    unit = LAB_ITEMS[label][1]
    if not unit:
        return f"{label} {_fmt(label, v)}"
    return f"{label} {_fmt(label, v)}{unit}" if unit == "%" else f"{label} {_fmt(label, v)} {unit}"


def _med_line(i: int, m: dict) -> str:
    s = f"{i}. {m['drug']} {m['dose']} {m['route']} {m['freq']}"
    if m["status"] == "new":
        s += " - NEW" + (f" ({m['note']})" if m["note"] else "")
    elif m["status"] == "changed":
        s += f" - CHANGED ({m['note']})"
    elif m["note"]:
        s += f" ({m['note']})"
    return s


def _vitals(rng: random.Random) -> str:
    return (f"temperature {36.5 + rng.randint(0, 6) / 10:.1f} C, blood pressure {rng.randint(118, 148)}/{rng.randint(68, 88)}, "
            f"heart rate {rng.randint(68, 96)}, respiratory rate {rng.randint(14, 20)}, "
            f"oxygen saturation {rng.randint(94, 98)}% on room air")


def build(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    patients, admissions, diagnoses, labs, rx, notes = [], [], [], [], [], []
    lab_id, rx_id, note_id = LAB_BASE, RX_BASE, NOTE_BASE

    def add_lab(sid, hadm, label, t, v):
        nonlocal lab_id
        lab_id += 1
        item, unit = LAB_ITEMS[label][0], LAB_ITEMS[label][1]
        labs.append({"labevent_id": lab_id, "subject_id": sid, "hadm_id": hadm, "itemid": item, "label": label,
                     "charttime": t, "value": _fmt(label, v), "valuenum": float(_fmt(label, v)), "valueuom": unit})

    def add_note(sid, hadm, ntype, t, text):
        nonlocal note_id
        note_id += 1
        notes.append({"note_id": note_id, "subject_id": sid, "hadm_id": hadm, "note_type": ntype,
                      "charttime": t, "text": text})

    for p in PATIENTS:
        sid = SUBJECT_BASE + p["n"]
        sexw, pron = ("man", "He") if p["sex"] == "M" else ("woman", "She")
        first_year = _dt(p["admissions"][0]["admit"]).year
        patients.append({"subject_id": sid, "gender": p["sex"], "anchor_age": p["age"], "anchor_year": first_year,
                         "anchor_year_group": "SYNTHETIC"})

        for ai, a in enumerate(p["admissions"], 1):
            hadm = HADM_BASE + p["n"] * 10 + ai
            admit, disch = _dt(a["admit"]), _dt(a["disch"])
            age = p["age"] + (admit.year - first_year)
            admissions.append({"hadm_id": hadm, "subject_id": sid, "admittime": a["admit"], "dischtime": a["disch"],
                               "admission_type": f"SYNTHETIC-{a['atype'].upper()}" if a.get("atype") else "SYNTHETIC-DEMO",
                               "insurance": "SYNTHETIC-DEMO",
                               "discharge_location": a.get("dispo", "HOME"), "hospital_expire_flag": 0})
            for seq, (code, desc) in enumerate(a["dx"], 1):
                diagnoses.append({"subject_id": sid, "hadm_id": hadm, "seq_num": seq, "icd_code": code.replace(".", ""),
                                  "icd_version": 10, "description": desc})

            # labs: authored series, then seeded supporting panels on admission + discharge
            for label, series in a["labs"].items():
                for t, v in series:
                    add_lab(sid, hadm, label, t, v)
            adm_t = (admit + timedelta(minutes=35)).strftime("%Y-%m-%d %H:%M")
            dis_t = disch.strftime("%Y-%m-%d") + " 06:00"
            support = {"admission": {}, "discharge": {}}
            for label in SUPPORT_LABS:
                if label in a["labs"]:
                    continue
                for when, t in (("admission", adm_t), ("discharge", dis_t)):
                    base, sp = p["base"][label], SUPPORT_SPREAD[label]
                    v = base + (rng.random() * 2 - 1) * sp
                    support[when][label] = v
                    add_lab(sid, hadm, label, t, v)

            # prescriptions (inpatient orders + discharge medications)
            for m in a.get("inpatient", []):
                rx_id += 1
                val, unit = (m["dose"].rsplit(" ", 1) + [""])[:2]
                rx.append({"pharmacy_id": rx_id, "subject_id": sid, "hadm_id": hadm, "drug": m["drug"], "drug_type": "INPATIENT",
                           "dose_val_rx": val, "dose_unit_rx": unit, "route": m["route"], "frequency": m["freq"],
                           "starttime": (admit + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
                           "stoptime": (admit + (disch - admit) / 2).strftime("%Y-%m-%d %H:%M"), "status": "inpatient"})
            for m in a["meds"]:
                rx_id += 1
                val, unit = (m["dose"].rsplit(" ", 1) + [""])[:2]
                start = admit + (timedelta(hours=3) if m["status"] == "continued" else (disch - admit) / 2)
                rx.append({"pharmacy_id": rx_id, "subject_id": sid, "hadm_id": hadm, "drug": m["drug"], "drug_type": "DISCHARGE",
                           "dose_val_rx": val, "dose_unit_rx": unit, "route": m["route"], "frequency": m["freq"],
                           "starttime": start.strftime("%Y-%m-%d %H:%M"), "stoptime": a["disch"], "status": m["status"]})

            # --- discharge summary -------------------------------------------------
            lab_lines = []
            for label, series in a["labs"].items():
                for j, (t, v) in enumerate(series):
                    tag = " (admission)" if j == 0 and len(series) > 1 else (" (discharge)" if j == len(series) - 1 and len(series) > 1 else "")
                    lab_lines.append(f"{_lab_str(label, v)} on {t[:10]}{tag}")
            for when in ("admission", "discharge"):
                if support[when]:
                    vals = ", ".join(_lab_str(k, v).lower().replace("mg/dl", "mg/dL").replace("g/dl", "g/dL")
                                     .replace("meq/l", "mEq/L").replace("k/ul", "K/uL") for k, v in support[when].items())
                    lab_lines.append(f"Other {when} labs: {vals}.")
            meds = "\n".join(_med_line(i, m) for i, m in enumerate(a["meds"], 1))
            stopped = "".join(f"\nStopped: {d} - {why}." for d, why in a.get("stopped", []))
            imaging = "\n".join(f"{im['exam']} ({im['time'][:10]}): {im['impression']}" for im in a.get("imaging", []))
            dx_lines = [f"Primary: {a['dx'][0][1]} ({a['dx'][0][0]})"] + [f"- {d} ({c})" for c, d in a["dx"][1:]]
            pmh = "\n".join(f"- {x}" for x in p["pmh"])
            text = (
                f"DISCHARGE SUMMARY\nSYNTHETIC DEMO RECORD - fictional patient\n"
                f"Admission date: {a['admit'][:10]}    Discharge date: {a['disch'][:10]}\nService: {a['service']}\n"
                + (f"Admission type: {a['atype']}\n" if a.get("atype") else "") + "\n"
                f"CHIEF COMPLAINT:\n{a['cc']}\n\n"
                f"HISTORY OF PRESENT ILLNESS:\n{age}-year-old {sexw} with {p['pmh_short']}. {a['hpi']}\n\n"
                f"PAST MEDICAL HISTORY:\n{pmh}\n\nSOCIAL HISTORY:\n{p['social']}\n\n"
                f"PHYSICAL EXAM:\nOn admission: {a.get('vitals') or _vitals(rng)}. {a['exam']}\n\n"
                f"HOSPITAL COURSE:\n{a['course']}\n\n"
                f"LABORATORY DATA:\n" + "\n".join(lab_lines) + "\n\n"
                + (f"IMAGING:\n{imaging}\n\n" if imaging else "")
                + "DISCHARGE DIAGNOSES:\n" + "\n".join(dx_lines) + "\n\n"
                f"DISCHARGE MEDICATIONS:\n{meds}{stopped}\n\n"
                f"DISCHARGE CONDITION:\nStable. {pron} is ambulating and tolerating a diet.\n\n"
                f"FOLLOW UP:\n{a['followup']}\n"
            )
            add_note(sid, hadm, "discharge", a["disch"], text)

            # --- imaging reports -----------------------------------------------
            for im in a.get("imaging", []):
                rad = (f"RADIOLOGY REPORT\nSYNTHETIC DEMO RECORD - fictional patient\n"
                       f"EXAMINATION: {im['exam']}\nDATE: {im['time'][:10]}\n"
                       f"INDICATION: {im['indication']}\nCOMPARISON: {im['comparison']}\n\n"
                       f"FINDINGS:\n{im['findings']}\n\nIMPRESSION:\n{im['impression']}\n")
                add_note(sid, hadm, "radiology", im["time"], rad)

            # --- progress notes with assessment and plan -------------------------
            for pr in a.get("progress", []):
                lab_txt = "; ".join(f"{_lab_str(l, v)} on {t[:10]}" for l in pr["labs"]
                                    for t, v in a["labs"][l] if t <= pr["time"])
                prog = (f"PROGRESS NOTE - HOSPITAL DAY {pr['day']}\nSYNTHETIC DEMO RECORD - fictional patient\nDate: {pr['time'][:10]}\n"
                        f"Interval history: {pr['interval']}\n\nPHYSICAL EXAM:\n{pr['exam']}\n\n"
                        + (f"LABS:\n{lab_txt}\n\n" if lab_txt else "")
                        + f"ASSESSMENT AND PLAN:\n{age}-year-old {sexw} with {p['pmh_short']}.\n{pr['ap']}\n")
                add_note(sid, hadm, "progress", pr["time"], prog)

        # --- outpatient notes (no admission) ------------------------------------
        for pr in p.get("clinic", []):
            for label, t, v in pr["labs"]:
                add_lab(sid, None, label, t, v)
            lab_txt = "; ".join(f"{_lab_str(l, v)} on {t[:10]}" for l, t, v in pr["labs"])
            prog = (f"OUTPATIENT PROGRESS NOTE\nSYNTHETIC DEMO RECORD - fictional patient\nDate: {pr['time'][:10]}\n"
                    f"Interval history: {pr['interval']}\n\nPHYSICAL EXAM:\n{pr['exam']}\n\nLABS:\n{lab_txt}\n\n"
                    f"ASSESSMENT AND PLAN:\n{pr['ap']}\n")
            add_note(sid, None, "progress", pr["time"], prog)

    lab_items = [{"itemid": v[0], "label": k, "fluid": v[2], "category": v[3]} for k, v in LAB_ITEMS.items()]
    n_adm = {p["n"]: len(p["admissions"]) for p in PATIENTS}
    golden = []
    for q in GOLDEN:
        assert all(1 <= i <= n_adm[q["n"]] for i in q["evidence_admissions"]), (q["id"], "bad admission index")
        golden.append({"id": q["id"], "subject_id": SUBJECT_BASE + q["n"], "query": q["query"],
                       "category": q["category"], "temporal": q["temporal"],
                       "expected_facts": q["expected_facts"], "min_facts": q["min_facts"],
                       "answer_type": q["answer_type"], "difficulty": q["difficulty"],
                       "expected_answer": q["expected_answer"], "unsupported": q["unsupported"],
                       "must_not_contain": q["must_not_contain"],
                       "evidence_hadm_ids": [HADM_BASE + q["n"] * 10 + i for i in q["evidence_admissions"]],
                       "evidence_note_types": q["evidence_note_types"]})

    # Every expected fact must exist in that patient's notes, and nothing a "not documented"
    # answer must avoid may exist there, or the QA set is wrong.
    norm = lambda s: " ".join(s.lower().split())
    for g in golden:
        corpus = norm(" ".join(n["text"] for n in notes if n["subject_id"] == g["subject_id"]))
        missing = [f for f in g["expected_facts"] if norm(f) not in corpus]
        assert not missing, (g["id"], missing)
        present = [f for f in g["must_not_contain"] if norm(f) in corpus]
        assert not present, (g["id"], "must_not_contain present", present)
        assert g["min_facts"] <= len(g["expected_facts"]), g["id"]

    return {"synthetic_patients.json": patients, "synthetic_admissions.json": admissions,
            "synthetic_diagnoses.json": diagnoses, "synthetic_lab_items.json": lab_items,
            "synthetic_labs.json": labs, "synthetic_prescriptions.json": rx,
            "synthetic_notes.json": notes, "golden_qa.json": golden}


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


# ===========================================================================
# Corpus validation — relational integrity, temporal sanity, synthetic-only
# ===========================================================================
MIN_PATIENTS = 30
# Anything that looks like a name, address, phone, email, MRN or date of birth.
PHI_PATTERNS = [
    (r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "telephone number"),
    (r"[\w.+-]+@[\w-]+\.[a-z]{2,}", "email address"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "social security number"),
    (r"\b(?:MRN|Medical Record Number)\b", "medical record number"),
    (r"\b(?:Dr|Mr|Mrs|Ms|Miss)\.\s+[A-Z]", "personal name"),
    (r"\b(?:DOB|Date of Birth)\b", "date of birth"),
    (r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr)\b", "street address"),
]


def validate(files: dict) -> list[str]:
    """Return a list of problems; empty means the corpus is internally consistent."""
    bad: list[str] = []
    add = bad.append
    pat = files["synthetic_patients.json"]
    adm = files["synthetic_admissions.json"]
    notes, labs, rx = files["synthetic_notes.json"], files["synthetic_labs.json"], files["synthetic_prescriptions.json"]
    dx, items, golden = files["synthetic_diagnoses.json"], files["synthetic_lab_items.json"], files["golden_qa.json"]

    subjects = {p["subject_id"] for p in pat}
    if len(pat) < MIN_PATIENTS:
        add(f"only {len(pat)} patients, expected at least {MIN_PATIENTS}")

    # identifiers: synthetic ranges and uniqueness
    if any(not 90000000 < s <= 90999999 for s in subjects):
        add("subject_id outside the synthetic 90000001-90999999 range")
    if any(not 91000000 < a["hadm_id"] <= 91999999 for a in adm):
        add("hadm_id outside the synthetic range")
    if any(not 990000 <= i["itemid"] <= 990999 for i in items):
        add("itemid outside the synthetic range")
    for name, rows, key in (("patients", pat, "subject_id"), ("admissions", adm, "hadm_id"),
                            ("notes", notes, "note_id"), ("labs", labs, "labevent_id"),
                            ("prescriptions", rx, "pharmacy_id"), ("lab_items", items, "itemid")):
        ids = [r[key] for r in rows]
        if len(ids) != len(set(ids)):
            add(f"duplicate {key} in {name}")
    if len({(d["hadm_id"], d["seq_num"]) for d in dx}) != len(dx):
        add("duplicate (hadm_id, seq_num) in diagnoses")

    # relational integrity: every reference resolves to a synthetic parent
    window = {a["hadm_id"]: (a["subject_id"], a["admittime"], a["dischtime"]) for a in adm}
    for a in adm:
        if a["subject_id"] not in subjects:
            add(f"orphan admission {a['hadm_id']}")
        if a["dischtime"] <= a["admittime"]:
            add(f"admission {a['hadm_id']} discharges before it admits")
    for name, rows, tkey in (("note", notes, "charttime"), ("lab", labs, "charttime"),
                             ("prescription", rx, "starttime"), ("diagnosis", dx, None)):
        for r in rows:
            if r["subject_id"] not in subjects:
                add(f"orphan {name} for subject {r['subject_id']}")
            h = r.get("hadm_id")
            if h is None:
                if name == "diagnosis":
                    add("diagnosis without an admission")
                continue
            if h not in window:
                add(f"{name} references unknown admission {h}")
                continue
            sid, lo, hi = window[h]
            if r["subject_id"] != sid:
                add(f"{name} on admission {h} belongs to a different subject")
            if tkey and not lo[:10] <= r[tkey][:10] <= hi[:10]:
                add(f"{name} at {r[tkey]} falls outside admission {h} ({lo} to {hi})")
    known_items = {i["itemid"] for i in items}
    if any(l["itemid"] not in known_items for l in labs):
        add("lab event references an unknown lab item")

    # admissions per patient must not overlap and must be ordered
    per = {}
    for a in adm:
        per.setdefault(a["subject_id"], []).append((a["admittime"], a["dischtime"], a["hadm_id"]))
    for sid, rows in per.items():
        rows.sort()
        for (_, prev_out, prev_id), (nxt_in, _, nxt_id) in zip(rows, rows[1:]):
            if nxt_in <= prev_out:
                add(f"admissions {prev_id} and {nxt_id} overlap for subject {sid}")

    # no obvious PHI in free text
    for n in notes:
        for rx_pat, what in PHI_PATTERNS:
            if re.search(rx_pat, n["text"]):
                add(f"note {n['note_id']} looks like it contains a {what}")

    # golden QA points at real synthetic patients and real admissions
    hadms = set(window)
    for q in golden:
        if q["subject_id"] not in subjects:
            add(f"{q['id']} references unknown subject {q['subject_id']}")
        if any(h not in hadms for h in q["evidence_hadm_ids"]):
            add(f"{q['id']} references an unknown admission")
    if len({q["id"] for q in golden}) != len(golden):
        add("duplicate golden question id")
    return bad


def main() -> int:
    files = build()
    problems = validate(files)
    if problems:
        print("VALIDATION FAILED", file=sys.stderr)
        for p in problems[:40]:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if "--validate" in sys.argv:
        print(json.dumps({"validation": "PASS", "checks": "ids, orphans, dates, overlap, phi, golden refs",
                          "counts": {k.replace(".json", ""): len(v) for k, v in files.items()}}))
        return 0
    manifest = {"version": VERSION, "seed": SEED, "synthetic": True, "source": "authored fictional scenarios (src/demo_data/generate.py)",
                "counts": {k.replace(".json", ""): len(v) for k, v in files.items()},
                "sha256": {k: hashlib.sha256(_dump(v).encode()).hexdigest() for k, v in files.items()}}
    files["manifest.json"] = manifest
    if "--check" in sys.argv:
        stale = [k for k, v in files.items() if not (OUT / k).exists() or (OUT / k).read_text() != _dump(v)]
        print("up to date" if not stale else f"stale: {stale}")
        return 1 if stale else 0
    for k, v in files.items():
        (OUT / k).write_text(_dump(v))
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
