"""
Lumen De-Identification Pipeline
=================================
Presidio + spaCy (en_core_web_lg) with custom recognizers for clinical PHI.

Detects and redacts:
  - Person names (patient, physician, nurse, family)
  - Dates (DOB, admission, discharge, procedure dates)
  - Locations (addresses, hospitals, cities, states, zip codes)
  - Phone / fax numbers
  - Medical record numbers (MRN)
  - Social Security Numbers (SSN)
  - Email addresses
  - Ages over 89 (HIPAA Safe Harbor rule)
  - Device / serial identifiers
  - Account / insurance numbers

Usage:
    from src.deid.pipeline import DeidentificationPipeline

    pipeline = DeidentificationPipeline()
    result = pipeline.deidentify(clinical_note_text)
    print(result.text)           # redacted text
    print(result.entities)       # list of detected PHI entities
    print(result.entity_counts)  # summary dict
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from presidio_analyzer import (
    AnalyzerEngine,
    PatternRecognizer,
    Pattern,
    RecognizerResult,
    RecognizerRegistry,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)

# Suppress noisy Presidio warnings about unmapped spaCy entity types
# (CARDINAL, PERCENT, QUANTITY, etc. are not PHI — safe to ignore)
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class DeidResult:
    """Container for de-identification output."""

    text: str
    original_text: str
    entities: list[dict] = field(default_factory=list)

    @property
    def entity_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ent in self.entities:
            counts[ent["type"]] = counts.get(ent["type"], 0) + 1
        return counts

    @property
    def phi_found(self) -> bool:
        return len(self.entities) > 0


# ---------------------------------------------------------------------------
# Custom clinical recognizers
# ---------------------------------------------------------------------------

class MRNRecognizer(PatternRecognizer):
    """Detects Medical Record Numbers in common clinical formats."""

    def __init__(self):
        patterns = [
            Pattern("mrn_labeled", r"(?i)\bMRN[\s:#]*(\d{5,12})\b", 0.95),
            Pattern("medical_record", r"(?i)\bmedical\s+record\s*(?:number|no|#)?[\s:#]*(\d{5,12})\b", 0.90),
            Pattern("patient_id_labeled", r"(?i)\bpatient\s*(?:id|ID|#)[\s:#]*(\d{5,12})\b", 0.85),
            Pattern("acct_number", r"(?i)\baccount\s*(?:number|no|#)?[\s:#]*(\d{5,12})\b", 0.80),
        ]
        super().__init__(
            supported_entity="MEDICAL_RECORD_NUMBER",
            supported_language="en",
            patterns=patterns,
            context=["mrn", "medical record", "patient id", "chart number", "account"],
        )


class ClinicalDateRecognizer(PatternRecognizer):
    """Catches date formats common in clinical notes that Presidio sometimes misses."""

    def __init__(self):
        patterns = [
            # 01/15/2024, 1/5/2024
            Pattern("date_slash", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", 0.70),
            # 01-15-2024
            Pattern("date_dash", r"\b\d{1,2}-\d{1,2}-\d{2,4}\b", 0.70),
            # January 15, 2024 / Jan 15, 2024
            Pattern(
                "date_written",
                r"(?i)\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s*\d{2,4}\b",
                0.85,
            ),
            # 15 January 2024
            Pattern(
                "date_euro",
                r"(?i)\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{2,4}\b",
                0.85,
            ),
            # 2024-01-15 (ISO)
            Pattern("date_iso", r"\b\d{4}-\d{2}-\d{2}\b", 0.75),
        ]
        super().__init__(
            supported_entity="DATE_TIME",
            supported_language="en",
            patterns=patterns,
            context=[
                "date", "dob", "born", "birth", "admitted", "admission",
                "discharged", "discharge", "procedure", "surgery", "visit",
                "encounter", "onset", "diagnosed",
            ],
        )


class AgeOver89Recognizer(PatternRecognizer):
    """
    HIPAA Safe Harbor: ages over 89 must be redacted.
    Catches patterns like '92-year-old', 'age 94', '91 y/o', 'aged 103'.
    """

    def __init__(self):
        patterns = [
            Pattern(
                "age_year_old",
                r"\b(9[0-9]|[1-9]\d{2,})\s*[-–]?\s*(?:year|yr)s?\s*[-–]?\s*old\b",
                0.92,
            ),
            Pattern(
                "age_yo",
                r"\b(9[0-9]|[1-9]\d{2,})\s*(?:y/?o|yo)\b",
                0.90,
            ),
            Pattern(
                "age_labeled",
                r"(?i)\b(?:age|aged)\s*:?\s*(9[0-9]|[1-9]\d{2,})\b",
                0.90,
            ),
        ]
        super().__init__(
            supported_entity="AGE_OVER_89",
            supported_language="en",
            patterns=patterns,
            context=["age", "old", "elderly", "year", "yr"],
        )


class DeviceIdentifierRecognizer(PatternRecognizer):
    """Catches device serial numbers and UDIs common in clinical notes."""

    def __init__(self):
        patterns = [
            # "serial number: SPN3-2024-AK7842" — label + value (whole match)
            Pattern(
                "serial_labeled_with_value",
                r"(?i)\b(?:serial|device|UDI|implant)\s*(?:number|no|#|ID)?[\s:#]*[A-Z0-9][A-Z0-9\-]{4,24}\b",
                0.90,
            ),
            # Standalone alphanumeric-dash codes near context words
            # e.g., "SPN3-2024-AK7842" when preceded by device context
            Pattern(
                "serial_standalone",
                r"\b[A-Z]{2,5}\d[\w\-]{4,20}\b",
                0.45,
            ),
        ]
        super().__init__(
            supported_entity="DEVICE_ID",
            supported_language="en",
            patterns=patterns,
            context=[
                "serial", "device", "implant", "pacemaker", "stent", "UDI",
                "valve", "prosthesis", "catheter", "lead", "generator",
                "number", "model",
            ],
        )


class InsuranceRecognizer(PatternRecognizer):
    """Catches insurance/policy/group numbers."""

    def __init__(self):
        patterns = [
            Pattern(
                "insurance_id",
                r"(?i)\b(?:insurance|policy|group|member|subscriber)\s*(?:number|no|#|ID)?[\s:#]*([A-Z0-9]{5,20})\b",
                0.80,
            ),
        ]
        super().__init__(
            supported_entity="INSURANCE_NUMBER",
            supported_language="en",
            patterns=patterns,
            context=["insurance", "policy", "group", "member", "subscriber", "coverage"],
        )


class HospitalFacilityRecognizer(PatternRecognizer):
    """
    Catches hospital and medical facility names that spaCy often tags as ORG
    instead of LOCATION, or misses entirely.

    Matches patterns like:
      - Springfield Memorial Hospital
      - UCSF Medical Center
      - Rush University Medical Center
      - Northwestern Memorial Hospital
      - Stroger Hospital
      - Springfield Medical Lab
    """

    def __init__(self):
        # Facility suffix words that indicate a healthcare institution
        facility_suffixes = (
            r"(?:Hospital|Medical\s+Center|Medical\s+Lab(?:oratory)?|"
            r"Health\s+Center|Health\s+System|Clinic|"
            r"Medical\s+Group|Medical\s+Campus|"
            r"Children(?:'s)?\s+Hospital|"
            r"Veterans?\s+(?:Affairs\s+)?(?:Medical\s+Center|Hospital)|"
            r"Rehabilitation\s+Center|Surgical\s+Center|"
            r"Cancer\s+Center|Heart\s+(?:Center|Institute)|"
            r"Community\s+Hospital|Regional\s+Medical\s+Center|"
            r"University\s+Hospital)"
        )
        patterns = [
            # "Name Name Hospital/Medical Center/etc" — 1-5 preceding words
            Pattern(
                "named_facility",
                r"(?i)\b(?:[A-Z][a-zA-Z']+\s+){1,5}" + facility_suffixes + r"\b",
                0.88,
            ),
            # Abbreviation + Medical Center (e.g., "UCSF Medical Center")
            Pattern(
                "abbrev_facility",
                r"(?i)\b[A-Z]{2,6}\s+" + facility_suffixes + r"\b",
                0.85,
            ),
        ]
        super().__init__(
            supported_entity="LOCATION",
            supported_language="en",
            patterns=patterns,
            context=[
                "hospital", "admitted", "presented", "seen at", "transferred",
                "department", "emergency", "clinic", "facility", "center",
                "discharged from", "referred to",
            ],
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Entities Presidio handles natively + our custom ones
DEFAULT_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "LOCATION",
    "DATE_TIME",
    "IP_ADDRESS",
    "URL",
    # Custom
    "MEDICAL_RECORD_NUMBER",
    "AGE_OVER_89",
    "DEVICE_ID",
    "INSURANCE_NUMBER",
]

# Map entity types to the placeholder tags used in redacted text
ENTITY_TAG_MAP = {
    "PERSON": "[PERSON]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "US_SSN": "[SSN]",
    "LOCATION": "[LOCATION]",
    "DATE_TIME": "[DATE]",
    "IP_ADDRESS": "[IP_ADDRESS]",
    "URL": "[URL]",
    "MEDICAL_RECORD_NUMBER": "[MRN]",
    "AGE_OVER_89": "[AGE_OVER_89]",
    "DEVICE_ID": "[DEVICE_ID]",
    "INSURANCE_NUMBER": "[INSURANCE]",
}


class DeidentificationPipeline:
    """
    Production-grade de-identification pipeline for clinical text.

    Wraps Presidio Analyzer + Anonymizer with:
      - spaCy en_core_web_lg NER backbone
      - Custom clinical recognizers (MRN, ages >89, devices, insurance)
      - Configurable confidence threshold
      - Structured output with entity audit trail
    """

    def __init__(
        self,
        score_threshold: float = 0.35,
        entities: Optional[list[str]] = None,
        spacy_model: str = "en_core_web_lg",
    ):
        self.score_threshold = score_threshold
        self.entities = entities or DEFAULT_ENTITIES

        # --- Build NLP engine with spaCy ---
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()

        # --- Register custom recognizers ---
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)
        registry.add_recognizer(MRNRecognizer())
        registry.add_recognizer(ClinicalDateRecognizer())
        registry.add_recognizer(AgeOver89Recognizer())
        registry.add_recognizer(DeviceIdentifierRecognizer())
        registry.add_recognizer(InsuranceRecognizer())
        registry.add_recognizer(HospitalFacilityRecognizer())

        # --- Analyzer & Anonymizer ---
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry,
            supported_languages=["en"],
        )
        self.anonymizer = AnonymizerEngine()

        logger.info(
            "DeidentificationPipeline initialized "
            f"(model={spacy_model}, threshold={score_threshold}, "
            f"entities={len(self.entities)})"
        )

    def analyze(self, text: str) -> list[RecognizerResult]:
        """Run analysis only — returns raw Presidio results."""
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language="en",
            score_threshold=self.score_threshold,
        )
        # Sort by position for consistent output
        return sorted(results, key=lambda r: r.start)

    def deidentify(self, text: str) -> DeidResult:
        """
        Full de-identification: analyze → anonymize → return structured result.

        Returns a DeidResult with:
          .text           - the redacted text
          .original_text  - the input text (for auditing)
          .entities       - list of dicts with type, start, end, score, original_value
          .entity_counts  - summary dict {entity_type: count}
          .phi_found      - bool, True if any PHI detected
        """
        # Analyze
        analyzer_results = self.analyze(text)

        # Build entity audit trail before anonymization (positions shift after)
        entities = []
        for result in analyzer_results:
            entities.append(
                {
                    "type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": round(result.score, 3),
                    "original_value": text[result.start : result.end],
                }
            )

        # Build operator config — replace each entity type with its tag
        operators = {}
        for entity_type in self.entities:
            tag = ENTITY_TAG_MAP.get(entity_type, f"[{entity_type}]")
            operators[entity_type] = OperatorConfig("replace", {"new_value": tag})

        # Anonymize
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators,
        )

        return DeidResult(
            text=anonymized.text,
            original_text=text,
            entities=entities,
        )

    def deidentify_batch(
        self, texts: list[str], show_progress: bool = True
    ) -> list[DeidResult]:
        """De-identify a list of clinical notes."""
        results = []
        total = len(texts)
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 100 == 0:
                logger.info(f"De-identified {i + 1}/{total} notes")
            results.append(self.deidentify(text))
        if show_progress:
            logger.info(f"De-identification complete: {total} notes processed")
        return results