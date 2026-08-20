"""
Lumen Golden Evaluation Dataset
=================================
30 hand-curated clinical queries with relevance criteria.

Each query has:
  - query: the search query
  - category: query type for stratified analysis
  - relevance_criteria: keywords/phrases that MUST appear in relevant chunks
  - irrelevance_signals: keywords that indicate a WRONG result
  - min_relevant: minimum chunks we expect in top-5 that match criteria
  - notes: why this query tests something specific

These are used by the eval harness to automatically judge whether
retrieved chunks are relevant WITHOUT manually labeling chunk_ids
(which would break if you re-index).

The criteria-based approach is more robust and scalable than
hardcoding chunk_ids.
"""

GOLDEN_QUERIES = [
    # ──────────────────────────────────────────────────────
    # CATEGORY: Medication Queries
    # ──────────────────────────────────────────────────────
    {
        "id": "med_001",
        "query": "heart failure medications on discharge",
        "category": "medications",
        "relevance_criteria": [
            ["lasix", "furosemide", "carvedilol", "lisinopril", "metoprolol",
             "spironolactone", "entresto", "sacubitril", "digoxin", "hydralazine",
             "isosorbide", "torsemide"],
            ["discharge medication", "heart failure", "hf", "chf", "hfref", "hfpef"],
        ],
        "irrelevance_signals": ["deep vein", "venous doppler", "dvt", "appendicitis"],
        "min_relevant": 3,
        "notes": "Tests medication retrieval for CHF — should find discharge med lists with HF drugs",
    },
    {
        "id": "med_002",
        "query": "insulin dosing regimen diabetes",
        "category": "medications",
        "relevance_criteria": [
            ["insulin", "glargine", "lispro", "lantus", "humalog", "novolog",
             "levemir", "sliding scale", "basal bolus", "units"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests retrieval of insulin-specific content",
    },
    {
        "id": "med_003",
        "query": "anticoagulation warfarin heparin dosing",
        "category": "medications",
        "relevance_criteria": [
            ["warfarin", "heparin", "coumadin", "lovenox", "enoxaparin",
             "apixaban", "rivaroxaban", "anticoagul", "inr", "therapeutic"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests anticoagulant medication retrieval",
    },
    {
        "id": "med_004",
        "query": "antibiotics prescribed for pneumonia",
        "category": "medications",
        "relevance_criteria": [
            ["azithromycin", "levofloxacin", "ceftriaxone", "augmentin",
             "amoxicillin", "doxycycline", "moxifloxacin"],
            ["pneumonia", "antibiotic", "antimicrobial"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests infection-specific antibiotic retrieval",
    },
    {
        "id": "med_005",
        "query": "opioid pain management after surgery",
        "category": "medications",
        "relevance_criteria": [
            ["oxycodone", "morphine", "hydromorphone", "dilaudid", "opioid",
             "narcotic", "fentanyl", "tramadol", "percocet"],
            ["post-op", "postoperative", "surgical", "pain management", "pain control"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 2,
        "notes": "Tests post-surgical pain medication retrieval",
    },

    # ──────────────────────────────────────────────────────
    # CATEGORY: Lab Results
    # ──────────────────────────────────────────────────────
    {
        "id": "lab_001",
        "query": "abnormal potassium lab results",
        "category": "labs",
        "relevance_criteria": [
            ["potassium", "k+", "hypokalemia", "hyperkalemia", "meq", "k-4", "k-5",
             "k-3", "k-6", "k-7"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests whether actual potassium values are retrieved, not just notes mentioning potassium",
    },
    {
        "id": "lab_002",
        "query": "creatinine BUN kidney function labs",
        "category": "labs",
        "relevance_criteria": [
            ["creatinine", "bun", "urea nitrogen", "creat", "renal function"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests renal lab retrieval",
    },
    {
        "id": "lab_003",
        "query": "hemoglobin hematocrit anemia workup",
        "category": "labs",
        "relevance_criteria": [
            ["hemoglobin", "hgb", "hematocrit", "hct", "anemia", "rbc",
             "red blood cell", "iron", "ferritin", "reticulocyte"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests anemia-related lab retrieval",
    },
    {
        "id": "lab_004",
        "query": "blood culture results infection",
        "category": "labs",
        "relevance_criteria": [
            ["blood culture", "bacteremia", "gram stain", "staphylococcus",
             "streptococcus", "e. coli", "escherichia", "susceptib", "sensit",
             "resistant", "mrsa", "organism"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests microbiology result retrieval",
    },
    {
        "id": "lab_005",
        "query": "troponin BNP cardiac biomarkers",
        "category": "labs",
        "relevance_criteria": [
            ["troponin", "bnp", "nt-probnp", "ck-mb", "creatine kinase",
             "cardiac enzyme", "pro-bnp"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 2,
        "notes": "Tests cardiac biomarker retrieval",
    },

    # ──────────────────────────────────────────────────────
    # CATEGORY: Clinical Conditions / Diagnoses
    # ──────────────────────────────────────────────────────
    {
        "id": "dx_001",
        "query": "sepsis treatment antibiotics vasopressors",
        "category": "diagnosis",
        "relevance_criteria": [
            ["sepsis", "septic"],
            ["vasopressor", "norepinephrine", "levophed", "dopamine", "vasopressin",
             "antibiotic", "broad spectrum", "vancomycin", "piperacillin", "zosyn"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests sepsis management retrieval",
    },
    {
        "id": "dx_002",
        "query": "stroke neurological exam CT findings",
        "category": "diagnosis",
        "relevance_criteria": [
            ["stroke", "cva", "infarct", "hemorrhage", "ischemic", "tpa",
             "alteplase", "neuro", "ct head", "mri brain", "mca", "nihss"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 2,
        "notes": "Tests stroke-related note retrieval",
    },
    {
        "id": "dx_003",
        "query": "COPD exacerbation treatment nebulizer steroids",
        "category": "diagnosis",
        "relevance_criteria": [
            ["copd", "chronic obstructive", "albuterol", "nebulizer", "ipratropium",
             "prednisone", "solumedrol", "methylprednisolone", "bronchodilator",
             "exacerbation"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 2,
        "notes": "Tests COPD exacerbation retrieval",
    },
    {
        "id": "dx_004",
        "query": "GI bleeding upper lower endoscopy",
        "category": "diagnosis",
        "relevance_criteria": [
            ["gi bleed", "gastrointestinal bleed", "hematemesis", "melena",
             "hematochezia", "rectal bleed", "upper gi", "lower gi"],
            ["endoscopy", "colonoscopy", "egd", "esophagogastro"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests GI bleeding retrieval",
    },
    {
        "id": "dx_005",
        "query": "pneumonia chest xray infiltrate",
        "category": "diagnosis",
        "relevance_criteria": [
            ["pneumonia", "infiltrate", "consolidation", "opacity",
             "chest x-ray", "chest xray", "cxr", "radiograph"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests pneumonia with imaging findings",
    },

    # ──────────────────────────────────────────────────────
    # CATEGORY: Plain Language (tests query expansion)
    # ──────────────────────────────────────────────────────
    {
        "id": "plain_001",
        "query": "fluid overload swollen legs",
        "category": "plain_language",
        "relevance_criteria": [
            ["edema", "swelling", "fluid overload", "volume overload",
             "peripheral", "lower extremity", "bilateral leg", "pedal"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests expansion: swollen legs → edema, volume overload, CHF",
    },
    {
        "id": "plain_002",
        "query": "patient who stopped breathing needed a tube",
        "category": "plain_language",
        "relevance_criteria": [
            ["intubat", "mechanical ventilation", "respiratory failure",
             "endotracheal", "airway", "ventilat", "extubat"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests expansion: stopped breathing → intubation, respiratory failure",
    },
    {
        "id": "plain_003",
        "query": "blood sugar out of control",
        "category": "plain_language",
        "relevance_criteria": [
            ["glucose", "blood sugar", "hyperglycemia", "hypoglycemia",
             "dka", "diabetic ketoacidosis", "a1c", "hba1c"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests expansion: blood sugar → glucose, DKA, hyperglycemia",
    },
    {
        "id": "plain_004",
        "query": "confused elderly patient fell at home",
        "category": "plain_language",
        "relevance_criteria": [
            ["delirium", "altered mental status", "confusion", "fall",
             "syncope", "ams", "disoriented", "agitated"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 2,
        "notes": "Tests expansion: confused + fell → delirium, AMS, fall, syncope",
    },
    {
        "id": "plain_005",
        "query": "kidneys shutting down",
        "category": "plain_language",
        "relevance_criteria": [
            ["acute kidney injury", "aki", "renal failure", "creatinine",
             "oliguria", "oliguric", "dialysis", "esrd"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests expansion: kidneys shutting down → AKI, renal failure, creatinine",
    },

    # ──────────────────────────────────────────────────────
    # CATEGORY: Clinical Sections
    # ──────────────────────────────────────────────────────
    {
        "id": "sec_001",
        "query": "history of present illness admission reason",
        "category": "sections",
        "relevance_criteria": [
            ["history of present illness", "hpi", "presented with",
             "admitted", "chief complaint", "reason for admission"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests HPI section retrieval",
    },
    {
        "id": "sec_002",
        "query": "physical examination findings on admission",
        "category": "sections",
        "relevance_criteria": [
            ["physical exam", "vitals", "temperature", "blood pressure",
             "heart rate", "respiratory rate", "general:", "heent",
             "lungs:", "cardiac:", "abdomen:"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests physical exam section retrieval",
    },
    {
        "id": "sec_003",
        "query": "discharge instructions follow up appointments",
        "category": "sections",
        "relevance_criteria": [
            ["discharge instruction", "follow up", "follow-up",
             "appointment", "return to", "call your doctor"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests discharge instruction retrieval",
    },
    {
        "id": "sec_004",
        "query": "social history smoking alcohol drug use",
        "category": "sections",
        "relevance_criteria": [
            ["social history", "smoking", "tobacco", "alcohol", "etoh",
             "drug use", "substance", "illicit", "cocaine", "heroin"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 2,
        "notes": "Tests social history section retrieval",
    },
    {
        "id": "sec_005",
        "query": "family history of cancer or heart disease",
        "category": "sections",
        "relevance_criteria": [
            ["family history", "mother", "father", "sibling", "brother",
             "sister", "cancer", "coronary", "cardiac", "diabetes",
             "hypertension"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 2,
        "notes": "Tests family history section retrieval",
    },

    # ──────────────────────────────────────────────────────
    # CATEGORY: Imaging / Radiology
    # ──────────────────────────────────────────────────────
    {
        "id": "img_001",
        "query": "chest xray findings pneumonia pleural effusion",
        "category": "imaging",
        "relevance_criteria": [
            ["chest x-ray", "chest xray", "cxr", "radiograph",
             "pleural effusion", "infiltrate", "consolidation",
             "opacity", "pneumonia", "atelectasis"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 3,
        "notes": "Tests radiology report retrieval",
    },
    {
        "id": "img_002",
        "query": "CT abdomen acute findings",
        "category": "imaging",
        "relevance_criteria": [
            ["ct abdomen", "computed tomography", "appendicitis",
             "diverticulitis", "bowel obstruction", "free air",
             "perforation", "abscess", "abdominal"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 2,
        "notes": "Tests CT abdomen report retrieval",
    },
    {
        "id": "img_003",
        "query": "echocardiogram ejection fraction cardiac function",
        "category": "imaging",
        "relevance_criteria": [
            ["echocardiogram", "echo", "ejection fraction", "ef",
             "wall motion", "lvef", "systolic function", "diastolic",
             "mitral", "aortic"],
        ],
        "irrelevance_signals": [],
        "min_relevant": 2,
        "notes": "Tests echocardiogram result retrieval",
    },
]
