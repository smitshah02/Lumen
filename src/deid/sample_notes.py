"""
Sample Clinical Notes for Testing
===================================
Realistic synthetic discharge summaries and clinical notes with known PHI
embedded at known positions. Use these to validate the de-identification
pipeline before MIMIC-IV data arrives.

Each sample includes the raw text and a manifest of expected PHI entities
so you can measure recall / precision of the pipeline.
"""

SAMPLE_NOTES = [
    {
        "id": "test_001",
        "note_type": "discharge_summary",
        "text": """DISCHARGE SUMMARY

Patient Name: Maria Elena Rodriguez
MRN: 00384721
Date of Birth: 03/17/1952
Date of Admission: 01/08/2024
Date of Discharge: 01/14/2024
Attending Physician: Dr. James K. Patterson

CHIEF COMPLAINT: Shortness of breath and bilateral lower extremity edema.

HISTORY OF PRESENT ILLNESS:
Mrs. Rodriguez is a 71-year-old female with a past medical history significant for
congestive heart failure (EF 30%), type 2 diabetes mellitus, and chronic kidney disease
stage 3b, who presented to Springfield Memorial Hospital emergency department on
January 8, 2024 with worsening dyspnea on exertion over the past 5 days and a 12-pound
weight gain over 2 weeks. She reports orthopnea requiring 3 pillows and paroxysmal
nocturnal dyspnea occurring nightly for the past 3 nights. She denies chest pain,
palpitations, or syncope.

SOCIAL HISTORY:
Lives at 1247 Oak Valley Drive, Apt 3B, Springfield, IL 62704 with her daughter
Angela. Former smoker (quit 2011). No alcohol or illicit drug use. Retired school
teacher. Emergency contact: Angela Rodriguez, phone (217) 555-0843.

MEDICATIONS ON DISCHARGE:
1. Furosemide 40mg PO BID
2. Lisinopril 20mg PO daily
3. Carvedilol 12.5mg PO BID
4. Metformin 1000mg PO BID
5. Empagliflozin 10mg PO daily
6. Atorvastatin 40mg PO QHS

LABS ON DISCHARGE:
BNP: 456 pg/mL (down from 2,340 on admission)
Creatinine: 1.8 mg/dL (baseline 1.6)
Potassium: 4.2 mEq/L
HbA1c: 7.8%

FOLLOW-UP:
- Cardiology: Dr. Patterson, January 22, 2024, 10:30 AM
- Primary Care: Dr. Susan M. Chen, February 5, 2024
- Labs: BMP and BNP at Springfield Medical Lab, January 19, 2024

Electronically signed by Dr. James K. Patterson, MD
January 14, 2024 at 4:32 PM""",
        "expected_phi": [
            {"type": "PERSON", "value": "Maria Elena Rodriguez"},
            {"type": "MEDICAL_RECORD_NUMBER", "value": "00384721"},
            {"type": "DATE_TIME", "value": "03/17/1952"},
            {"type": "DATE_TIME", "value": "01/08/2024"},
            {"type": "DATE_TIME", "value": "01/14/2024"},
            {"type": "PERSON", "value": "James K. Patterson"},
            {"type": "LOCATION", "value": "Springfield Memorial Hospital"},
            {"type": "DATE_TIME", "value": "January 8, 2024"},
            {"type": "LOCATION", "value": "1247 Oak Valley Drive, Apt 3B, Springfield, IL 62704"},
            {"type": "PERSON", "value": "Angela"},
            {"type": "PHONE_NUMBER", "value": "(217) 555-0843"},
            {"type": "PERSON", "value": "Dr. Patterson"},
            {"type": "DATE_TIME", "value": "January 22, 2024"},
            {"type": "PERSON", "value": "Susan M. Chen"},
            {"type": "DATE_TIME", "value": "February 5, 2024"},
            {"type": "LOCATION", "value": "Springfield Medical Lab"},
            {"type": "DATE_TIME", "value": "January 19, 2024"},
            {"type": "DATE_TIME", "value": "January 14, 2024"},
        ],
    },
    {
        "id": "test_002",
        "note_type": "radiology_report",
        "text": """RADIOLOGY REPORT

Patient: Thomas J. Whitfield
MRN: 01129843
DOB: 11/22/1948
Exam Date: 03/15/2024
Ordering Physician: Dr. Priya Ramanathan
Radiologist: Dr. Michael S. Torres

EXAM: CT Chest with contrast

CLINICAL INDICATION: 75-year-old male with persistent cough and hemoptysis.
History of 40 pack-year smoking. Rule out malignancy.

TECHNIQUE: Helical CT of the chest was performed from the thoracic inlet to
the adrenal glands following IV administration of 80 mL Omnipaque 350.

FINDINGS:
A 2.3 x 1.8 cm spiculated mass is identified in the right upper lobe, abutting
the mediastinal pleura. Multiple ipsilateral mediastinal lymph nodes are
enlarged, the largest measuring 1.5 cm in the subcarinal station (station 7).

The left lung is clear. No pleural effusion. Heart size is normal. No
pericardial effusion. Thoracic aorta is mildly ectatic. Degenerative changes
of the thoracic spine.

IMPRESSION:
1. Right upper lobe spiculated mass highly suspicious for primary lung
   malignancy. Recommend PET-CT and tissue sampling.
2. Mediastinal lymphadenopathy concerning for nodal metastatic disease.

Results communicated to Dr. Ramanathan by phone at 2:45 PM on March 15, 2024.

Electronically signed: Dr. Michael S. Torres, MD
Mercy General Hospital, Department of Radiology
4500 J Street, Sacramento, CA 95819
Report finalized: 03/15/2024 15:12""",
        "expected_phi": [
            {"type": "PERSON", "value": "Thomas J. Whitfield"},
            {"type": "MEDICAL_RECORD_NUMBER", "value": "01129843"},
            {"type": "DATE_TIME", "value": "11/22/1948"},
            {"type": "DATE_TIME", "value": "03/15/2024"},
            {"type": "PERSON", "value": "Priya Ramanathan"},
            {"type": "PERSON", "value": "Michael S. Torres"},
            {"type": "PERSON", "value": "Dr. Ramanathan"},
            {"type": "DATE_TIME", "value": "March 15, 2024"},
            {"type": "LOCATION", "value": "Mercy General Hospital"},
            {"type": "LOCATION", "value": "4500 J Street, Sacramento, CA 95819"},
        ],
    },
    {
        "id": "test_003",
        "note_type": "progress_note",
        "text": """PROGRESS NOTE

Date: 2024-02-20
Patient: Barbara Ann Kowalski
MRN: 00567234
Insurance ID: BCBS-IL-98234571
SSN: 341-22-8876

Seen today in the Diabetes Management Clinic at Northwestern Memorial Hospital
by Dr. Aisha Johnson, NP.

SUBJECTIVE:
Mrs. Kowalski is a 93-year-old female with longstanding type 2 diabetes,
hypertension, and mild cognitive impairment. She is accompanied by her son
David Kowalski. She reports occasional dizziness when standing. Home glucose
readings range 140-220 mg/dL fasting. She says she sometimes forgets her
evening metformin dose.

Her husband Walter passed away last March and she has been living with David
at 892 Lakeshore Boulevard, Unit 14A, Chicago, IL 60611 since then.

OBJECTIVE:
Vitals: BP 152/88, HR 72, Temp 97.8°F, Weight 148 lbs
A1c today: 8.4% (up from 7.6% in August 2023)
eGFR: 38 mL/min (CKD stage 3b)
Monofilament exam: diminished sensation bilateral feet

ASSESSMENT AND PLAN:
1. Diabetes - suboptimal control, A1c rising. Reduce metformin to 500mg BID
   given declining renal function. Add semaglutide 0.25mg SQ weekly, titrate
   to 0.5mg after 4 weeks if tolerated. Referral to diabetes educator.
2. Hypertension - above target. Increase amlodipine from 5mg to 10mg daily.
3. Diabetic neuropathy - start gabapentin 100mg TID, titrate as tolerated.
4. Fall risk - given orthostatic symptoms + neuropathy + age, refer to PT.
   Home safety evaluation ordered.

Follow-up in 6 weeks. Call clinic at (312) 555-1920 if glucose >300 or
symptoms worsen. Email: diabetes.clinic@nm.org

Signed: Aisha Johnson, NP
Supervising Physician: Dr. Robert W. Gallagher, MD""",
        "expected_phi": [
            {"type": "DATE_TIME", "value": "2024-02-20"},
            {"type": "PERSON", "value": "Barbara Ann Kowalski"},
            {"type": "MEDICAL_RECORD_NUMBER", "value": "00567234"},
            {"type": "INSURANCE_NUMBER", "value": "BCBS-IL-98234571"},
            {"type": "US_SSN", "value": "341-22-8876"},
            {"type": "LOCATION", "value": "Northwestern Memorial Hospital"},
            {"type": "PERSON", "value": "Aisha Johnson"},
            {"type": "AGE_OVER_89", "value": "93-year-old"},
            {"type": "PERSON", "value": "David Kowalski"},
            {"type": "PERSON", "value": "Walter"},
            {"type": "LOCATION", "value": "892 Lakeshore Boulevard, Unit 14A, Chicago, IL 60611"},
            {"type": "DATE_TIME", "value": "August 2023"},
            {"type": "PHONE_NUMBER", "value": "(312) 555-1920"},
            {"type": "EMAIL_ADDRESS", "value": "diabetes.clinic@nm.org"},
            {"type": "PERSON", "value": "Robert W. Gallagher"},
        ],
    },
    {
        "id": "test_004",
        "note_type": "operative_note",
        "text": """OPERATIVE NOTE

Patient: Johnathan Lee Park
MRN: 00891203
DOB: 06/30/1961
Date of Surgery: 04/02/2024
Surgeon: Dr. Elena Vasquez-Morales
Assistant: Dr. Kevin O'Brien
Anesthesiologist: Dr. Yuki Tanaka

PREOPERATIVE DIAGNOSIS: Severe aortic stenosis with NYHA Class III symptoms.
POSTOPERATIVE DIAGNOSIS: Same.

PROCEDURE PERFORMED: Transcatheter Aortic Valve Replacement (TAVR) with
Edwards SAPIEN 3 Ultra valve, 26mm. Device serial number: SPN3-2024-AK7842.

DESCRIPTION OF PROCEDURE:
The patient is a 62-year-old male with severe calcific aortic stenosis
(valve area 0.7 cm2, mean gradient 52 mmHg) deemed intermediate surgical
risk by the heart team. After informed consent was obtained, the patient was
brought to the hybrid OR at University of California San Francisco Medical
Center.

Under general anesthesia with TEE guidance, right femoral artery access was
obtained. A 14-French Edwards eSheath was advanced. The native valve was
crossed and balloon valvuloplasty performed. The 26mm SAPIEN 3 Ultra valve
was deployed under rapid pacing at 180 bpm. Post-deployment TEE showed no
paravalvular leak and mean gradient of 8 mmHg.

Access site closed with two ProGlide devices. Total procedure time: 62 minutes.
Estimated blood loss: 50 mL. The patient tolerated the procedure well and was
transferred to the CVICU in stable condition.

Signed: Dr. Elena Vasquez-Morales, MD, FACC
UCSF Medical Center, Division of Cardiology
505 Parnassus Avenue, San Francisco, CA 94143
April 2, 2024""",
        "expected_phi": [
            {"type": "PERSON", "value": "Johnathan Lee Park"},
            {"type": "MEDICAL_RECORD_NUMBER", "value": "00891203"},
            {"type": "DATE_TIME", "value": "06/30/1961"},
            {"type": "DATE_TIME", "value": "04/02/2024"},
            {"type": "PERSON", "value": "Elena Vasquez-Morales"},
            {"type": "PERSON", "value": "Kevin O'Brien"},
            {"type": "PERSON", "value": "Yuki Tanaka"},
            {"type": "DEVICE_ID", "value": "SPN3-2024-AK7842"},
            {"type": "LOCATION", "value": "University of California San Francisco Medical Center"},
            {"type": "PERSON", "value": "Dr. Elena Vasquez-Morales"},
            {"type": "LOCATION", "value": "UCSF Medical Center"},
            {"type": "LOCATION", "value": "505 Parnassus Avenue, San Francisco, CA 94143"},
            {"type": "DATE_TIME", "value": "April 2, 2024"},
        ],
    },
    {
        "id": "test_005",
        "note_type": "emergency_note",
        "text": """EMERGENCY DEPARTMENT NOTE

Arrival: 11/28/2024 02:17 AM
Patient: Deshawn Marcus Williams
MRN: 00234876
DOB: 08/14/1989
Mode of Arrival: EMS from 3847 West Cermak Road, Chicago, IL 60623

TRIAGE: ESI Level 2
Chief Complaint: "My chest feels tight and I can't breathe"

HISTORY:
35-year-old male with history of asthma (moderate persistent) presents with
acute dyspnea, chest tightness, and audible wheezing that began approximately
2 hours prior to arrival. Patient states he ran out of his albuterol inhaler
3 days ago. He was seen at Stroger Hospital last month for a similar episode.
Denies fever, cough, hemoptysis. No recent travel or sick contacts.

ALLERGIES: Penicillin (rash), Sulfa drugs (hives)

EMERGENCY CONTACTS:
Mother: Patricia Williams (773) 555-4291
Spouse: Keisha Williams keisha.w88@gmail.com

EXAM:
General: Alert, oriented, moderate respiratory distress, speaking in phrases.
Lungs: Diffuse bilateral expiratory wheezes, poor air movement.
O2 sat 89% on RA, improved to 95% on 4L NC.

ED COURSE:
- Continuous albuterol nebulization x3
- Ipratropium 0.5mg nebulized x1
- Methylprednisolone 125mg IV
- Magnesium sulfate 2g IV

Patient improved significantly. O2 sat 97% on RA. Speaking in full sentences.
Wheezing resolved. Discharged home with:
- Prednisone 40mg PO daily x5 days
- Albuterol MDI with spacer
- Fluticasone/salmeterol 250/50 BID
- PCP follow-up within 5 days

Attending: Dr. Marcus A. Freeman, MD
Rush University Medical Center ED
1653 West Congress Parkway, Chicago, IL 60612
Note finalized: 11/28/2024 05:48 AM""",
        "expected_phi": [
            {"type": "DATE_TIME", "value": "11/28/2024"},
            {"type": "PERSON", "value": "Deshawn Marcus Williams"},
            {"type": "MEDICAL_RECORD_NUMBER", "value": "00234876"},
            {"type": "DATE_TIME", "value": "08/14/1989"},
            {"type": "LOCATION", "value": "3847 West Cermak Road, Chicago, IL 60623"},
            {"type": "LOCATION", "value": "Stroger Hospital"},
            {"type": "PERSON", "value": "Patricia Williams"},
            {"type": "PHONE_NUMBER", "value": "(773) 555-4291"},
            {"type": "PERSON", "value": "Keisha Williams"},
            {"type": "EMAIL_ADDRESS", "value": "keisha.w88@gmail.com"},
            {"type": "PERSON", "value": "Marcus A. Freeman"},
            {"type": "LOCATION", "value": "Rush University Medical Center"},
            {"type": "LOCATION", "value": "1653 West Congress Parkway, Chicago, IL 60612"},
        ],
    },
]
