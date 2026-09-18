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

    python -m src.demo_data.generate          # rewrite the JSON files
    python -m src.demo_data.generate --check  # verify files are up to date
"""

from __future__ import annotations

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
]


# ===========================================================================
# Golden QA — deterministic expected facts (checked against notes below)
# ===========================================================================
GOLDEN = [
    ("demo_q01", 1, "What was the patient's most recent creatinine?", "labs", "latest",
     ["creatinine 1.4 mg/dL"], 1),
    ("demo_q02", 1, "How did the patient's creatinine change over time?", "labs", "trend",
     ["creatinine 1.8 mg/dL", "creatinine 2.1 mg/dL", "creatinine 1.4 mg/dL"], 2),
    ("demo_q03", 1, "What medications was the patient discharged on most recently?", "medications", "latest",
     ["torsemide 20 mg", "sacubitril-valsartan 24-26 mg"], 2),
    ("demo_q04", 1, "Does the patient have a history of heart failure?", "diagnosis", "all",
     ["heart failure with reduced ejection fraction"], 1),
    ("demo_q05", 1, "Did the chest x-ray show pulmonary edema?", "imaging", "all",
     ["pulmonary edema"], 1),
    ("demo_q06", 3, "Why was the patient having trouble breathing?", "plain_language", "all",
     ["pneumonia"], 1),
    ("demo_q07", 2, "How has the creatinine changed across admissions?", "labs", "trend",
     ["creatinine 1.9 mg/dL", "creatinine 2.4 mg/dL", "creatinine 3.1 mg/dL"], 2),
    ("demo_q08", 2, "What was the most recent hemoglobin A1c?", "labs", "latest",
     ["hemoglobin A1c 7.1%"], 1),
    ("demo_q09", 3, "Did the right lower lobe pneumonia resolve on follow-up chest imaging?", "imaging", "all",
     ["interval resolution"], 1),
    ("demo_q10", 3, "How much supplemental oxygen did the patient require for pneumonia?", "diagnosis", "all",
     ["4 liters"], 1),
    ("demo_q11", 4, "Is the patient on anticoagulation for atrial fibrillation?", "medications", "all",
     ["apixaban 5 mg"], 1),
    ("demo_q12", 4, "How was the rate control medication changed?", "medications", "all",
     ["diltiazem", "metoprolol succinate 50 mg"], 2),
    ("demo_q13", 1, "Was furosemide started before or after the previous admission?", "medications", "all",
     ["not previously been prescribed a loop diuretic", "furosemide was started"], 1),
    ("demo_q14", 5, "What was the most recent hemoglobin?", "labs", "latest",
     ["hemoglobin 11.2 g/dL"], 1),
    ("demo_q15", 1, "What water pill is the patient taking now?", "plain_language", "all",
     ["torsemide 20 mg"], 1),
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
                               "admission_type": "SYNTHETIC-DEMO", "insurance": "SYNTHETIC-DEMO",
                               "discharge_location": "HOME", "hospital_expire_flag": 0})
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
                f"Admission date: {a['admit'][:10]}    Discharge date: {a['disch'][:10]}\nService: {a['service']}\n\n"
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
    golden = [{"id": qid, "subject_id": SUBJECT_BASE + pn, "query": q, "category": cat, "temporal": tmode,
               "expected_facts": facts, "min_facts": mn} for qid, pn, q, cat, tmode, facts, mn in GOLDEN]

    # Every expected fact must exist in that patient's notes, or the QA set is wrong.
    norm = lambda s: " ".join(s.lower().split())
    for g in golden:
        corpus = norm(" ".join(n["text"] for n in notes if n["subject_id"] == g["subject_id"]))
        missing = [f for f in g["expected_facts"] if norm(f) not in corpus]
        assert not missing, (g["id"], missing)

    return {"synthetic_patients.json": patients, "synthetic_admissions.json": admissions,
            "synthetic_diagnoses.json": diagnoses, "synthetic_lab_items.json": lab_items,
            "synthetic_labs.json": labs, "synthetic_prescriptions.json": rx,
            "synthetic_notes.json": notes, "golden_qa.json": golden}


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    files = build()
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
