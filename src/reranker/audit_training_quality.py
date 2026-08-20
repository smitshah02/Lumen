"""
Reranker Training Data — Quality Audit via Groq API
====================================================

Samples positive pairs from your training data, sends them to Groq
(running Llama 3.3 70B) for relevance scoring, and produces a quality
report telling you what percentage of your data is clean vs noisy.

Groq API is OpenAI-compatible, so we use the openai SDK.

Prerequisites:
    pip install openai tqdm

Usage:
    export GROQ_API_KEY="your-key-here"
    python audit_training_quality.py

Get your API key at: https://console.groq.com/keys
"""

import json
import os
import random
import time
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Groq API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"  # fast and capable; alternatives: "llama-3.1-8b-instant" (faster), "mixtral-8x7b-32768"

# Input
TRAIN_PAIRS_FILE = Path("/Users/smitshah/Lumen/src/reranker/reranker_training_data/train_full.jsonl")

# Audit settings
SAMPLE_SIZE = 400           # how many positive pairs to audit
ALSO_AUDIT_NEGATIVES = True # also score a sample of negatives (sanity check)
NEGATIVE_SAMPLE_SIZE = 100  # how many negatives to audit
MAX_WORKERS = 3             # Groq free tier: ~30 req/min. Keep low to avoid 429s
RETRY_ATTEMPTS = 3
RETRY_DELAY = 3             # seconds between retries (Groq rate limits are strict)

# Output
AUDIT_OUTPUT = Path("/Users/smitshah/Lumen/src/reranker/reranker_training_data/quality_audit.jsonl")
AUDIT_REPORT = Path("/Users/smitshah/Lumen/src/reranker/reranker_training_data/quality_report.txt")

random.seed(42)


# =============================================================================
# GROQ CLIENT
# =============================================================================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
)

SCORING_PROMPT = """You are a clinical relevance judge. You will be given a search query and a clinical document chunk from a patient's medical record. Rate how well the document answers or is relevant to the query.

Scoring rubric:
  0 = IRRELEVANT — The document has nothing to do with the query. No meaningful connection.
  1 = MARGINALLY RELATED — The document touches on a related topic but does NOT answer the query. For example, a medication list when the query asks about workup/management reasoning.
  2 = PARTIALLY RELEVANT — The document contains some relevant information but doesn't fully address the query. Missing key details.
  3 = HIGHLY RELEVANT — The document directly and substantively addresses the query. This is what a clinician searching for this query would want to read.

Important guidelines:
- A medication list that merely CONTAINS a drug name is NOT relevant to a query about "why was [drug] prescribed" — it needs to discuss the clinical reasoning.
- A note that MENTIONS a lab value in passing is NOT relevant to a query about that lab's "workup and management" — it needs to discuss the clinical response.
- Section headers or boilerplate (demographics, formatting) don't count as relevant content.

Respond with ONLY a JSON object, no other text:
{{"score": <0-3>, "reason": "<one sentence explanation>"}}

Query: {query}

Document:
{document}"""


def score_pair(query: str, document: str, pair_type: str = "positive",
               attempt: int = 0) -> dict:
    """Send a query-document pair to Groq for relevance scoring."""
    # Truncate very long documents to save tokens
    doc_truncated = document[:2000] if len(document) > 2000 else document

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": SCORING_PROMPT.format(
                    query=query, document=doc_truncated)
            }],
            temperature=0.0,
            max_tokens=150,
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON response — handle markdown fences if present
        cleaned = raw
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        result = json.loads(cleaned)
        return {
            "query": query,
            "document_preview": document[:200],
            "pair_type": pair_type,
            "score": int(result["score"]),
            "reason": result.get("reason", ""),
            "raw_response": raw,
        }

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY)
            return score_pair(query, document, pair_type, attempt + 1)
        return {
            "query": query,
            "document_preview": document[:200],
            "pair_type": pair_type,
            "score": -1,
            "reason": f"PARSE_ERROR: {str(e)}",
            "raw_response": raw if 'raw' in dir() else "",
        }

    except Exception as e:
        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY * (attempt + 1))
            return score_pair(query, document, pair_type, attempt + 1)
        return {
            "query": query,
            "document_preview": document[:200],
            "pair_type": pair_type,
            "score": -1,
            "reason": f"API_ERROR: {str(e)}",
            "raw_response": "",
        }


# =============================================================================
# SAMPLING
# =============================================================================

def load_samples() -> tuple[list[dict], list[dict]]:
    """Load and sample positive and negative pairs from training data."""

    print(f"Loading training data from {TRAIN_PAIRS_FILE}...")
    all_examples = []
    with open(TRAIN_PAIRS_FILE) as f:
        for line in f:
            all_examples.append(json.loads(line))

    print(f"  Total examples: {len(all_examples)}")

    # Sample positives (query + positive document)
    positive_samples = []
    sampled = random.sample(all_examples, min(SAMPLE_SIZE, len(all_examples)))
    for ex in sampled:
        positive_samples.append({
            "query": ex["query"],
            "document": ex["positive"],
            "query_type": ex["query_type"],
            "pair_type": "positive",
        })

    # Sample negatives
    negative_samples = []
    if ALSO_AUDIT_NEGATIVES:
        neg_sampled = random.sample(all_examples,
                                     min(NEGATIVE_SAMPLE_SIZE, len(all_examples)))
        for ex in neg_sampled:
            if ex["negatives"]:
                neg = random.choice(ex["negatives"])
                negative_samples.append({
                    "query": ex["query"],
                    "document": neg["text"],
                    "query_type": ex["query_type"],
                    "pair_type": f"negative_{neg['neg_type']}",
                    "neg_type": neg["neg_type"],
                })

    print(f"  Sampled {len(positive_samples)} positives, {len(negative_samples)} negatives")
    return positive_samples, negative_samples


# =============================================================================
# AUDIT EXECUTION
# =============================================================================

def run_audit(samples: list[dict]) -> list[dict]:
    """Score all samples using Groq API with parallel requests."""
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for i, sample in enumerate(samples):
            future = executor.submit(
                score_pair,
                sample["query"],
                sample["document"],
                sample["pair_type"]
            )
            futures[future] = {**sample, "index": i}

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Scoring"):
            sample_info = futures[future]
            result = future.result()
            result["query_type"] = sample_info["query_type"]
            if "neg_type" in sample_info:
                result["neg_type"] = sample_info["neg_type"]
            results.append(result)

    return results


# =============================================================================
# ANALYSIS & REPORTING
# =============================================================================

def generate_report(pos_results: list[dict], neg_results: list[dict]) -> str:
    """Analyze scores and produce a human-readable quality report."""
    lines = []
    lines.append("=" * 70)
    lines.append("RERANKER TRAINING DATA — QUALITY AUDIT REPORT")
    lines.append("=" * 70)

    # --- Positive pairs analysis ---
    lines.append("\n\n>>> POSITIVE PAIRS (should score 2-3)")
    lines.append("-" * 50)

    valid_pos = [r for r in pos_results if r["score"] >= 0]
    if not valid_pos:
        lines.append("No valid scores!")
        return "\n".join(lines)

    scores = [r["score"] for r in valid_pos]
    avg = sum(scores) / len(scores)
    lines.append(f"  Samples scored:    {len(valid_pos)}")
    lines.append(f"  Average score:     {avg:.2f} / 3.00")
    lines.append(f"  Parse errors:      {len(pos_results) - len(valid_pos)}")

    # Distribution
    dist = defaultdict(int)
    for s in scores:
        dist[s] += 1
    lines.append(f"\n  Score distribution:")
    for s in [0, 1, 2, 3]:
        count = dist[s]
        pct = count / len(valid_pos) * 100
        bar = "█" * int(pct / 2)
        lines.append(f"    {s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Quality verdict
    good_pct = (dist[2] + dist[3]) / len(valid_pos) * 100
    bad_pct = (dist[0] + dist[1]) / len(valid_pos) * 100
    lines.append(f"\n  GOOD (score 2-3):  {good_pct:.1f}%")
    lines.append(f"  BAD  (score 0-1):  {bad_pct:.1f}%")

    if good_pct >= 80:
        lines.append(f"\n  ✓ VERDICT: Data quality is GOOD. Safe to train.")
    elif good_pct >= 60:
        lines.append(f"\n  ⚠ VERDICT: Data quality is MODERATE. Consider filtering score 0-1 pairs before training.")
    else:
        lines.append(f"\n  ✗ VERDICT: Data quality is LOW. Needs improvement before training.")

    # Breakdown by query type
    lines.append(f"\n  Breakdown by query type:")
    by_type = defaultdict(list)
    for r in valid_pos:
        by_type[r["query_type"]].append(r["score"])

    for qtype in sorted(by_type.keys()):
        type_scores = by_type[qtype]
        type_avg = sum(type_scores) / len(type_scores)
        type_good = sum(1 for s in type_scores if s >= 2) / len(type_scores) * 100
        lines.append(f"    {qtype:15s}: avg={type_avg:.2f}  good={type_good:.0f}%  n={len(type_scores)}")

    # Worst examples (for debugging)
    lines.append(f"\n  Worst positive pairs (score 0):")
    worst = [r for r in valid_pos if r["score"] == 0][:5]
    for r in worst:
        lines.append(f"    Query: {r['query'][:80]}")
        lines.append(f"    Doc:   {r['document_preview'][:80]}...")
        lines.append(f"    Why:   {r['reason']}")
        lines.append("")

    # --- Negative pairs analysis ---
    if neg_results:
        lines.append("\n\n>>> NEGATIVE PAIRS (should score 0-1)")
        lines.append("-" * 50)

        valid_neg = [r for r in neg_results if r["score"] >= 0]
        neg_scores = [r["score"] for r in valid_neg]
        neg_avg = sum(neg_scores) / len(neg_scores) if neg_scores else 0

        lines.append(f"  Samples scored:    {len(valid_neg)}")
        lines.append(f"  Average score:     {neg_avg:.2f} / 3.00")

        neg_dist = defaultdict(int)
        for s in neg_scores:
            neg_dist[s] += 1
        lines.append(f"\n  Score distribution:")
        for s in [0, 1, 2, 3]:
            count = neg_dist[s]
            pct = count / len(valid_neg) * 100 if valid_neg else 0
            bar = "█" * int(pct / 2)
            lines.append(f"    {s}: {count:4d} ({pct:5.1f}%) {bar}")

        good_neg_pct = (neg_dist[0] + neg_dist[1]) / len(valid_neg) * 100 if valid_neg else 0
        bad_neg_pct = (neg_dist[2] + neg_dist[3]) / len(valid_neg) * 100 if valid_neg else 0
        lines.append(f"\n  GOOD negatives (score 0-1): {good_neg_pct:.1f}%")
        lines.append(f"  FALSE NEGATIVES (score 2-3): {bad_neg_pct:.1f}%  ← these hurt training")

        # Breakdown by negative type
        lines.append(f"\n  Breakdown by negative type:")
        by_neg_type = defaultdict(list)
        for r in valid_neg:
            by_neg_type[r.get("neg_type", "unknown")].append(r["score"])
        for ntype in sorted(by_neg_type.keys()):
            type_scores = by_neg_type[ntype]
            type_avg = sum(type_scores) / len(type_scores)
            false_neg_pct = sum(1 for s in type_scores if s >= 2) / len(type_scores) * 100
            lines.append(f"    {ntype:35s}: avg={type_avg:.2f}  false_neg={false_neg_pct:.0f}%  n={len(type_scores)}")

    # --- Recommendations ---
    lines.append("\n\n>>> RECOMMENDATIONS")
    lines.append("-" * 50)

    recs = []
    if good_pct < 80:
        recs.append("1. Filter out training pairs where the positive scores 0-1 (use this audit's "
                     "approach at scale, or improve the keyword matcher).")
    if by_type.get("medication") and sum(by_type["medication"]) / len(by_type["medication"]) < 1.5:
        recs.append("2. Medication queries have low relevance — medication lists match on drug names "
                     "but don't discuss clinical reasoning. Consider filtering or using LLM-based "
                     "positive selection for medication pairs.")
    if by_type.get("lab") and sum(by_type["lab"]) / len(by_type["lab"]) < 1.5:
        recs.append("3. Lab queries have low relevance — notes often mention lab values in passing "
                     "without discussing workup. Consider requiring the lab name to appear in a "
                     "'Hospital Course' or 'Assessment' section specifically.")
    if neg_results and bad_neg_pct > 15:
        recs.append("4. High false-negative rate — some 'negatives' are actually relevant. "
                     "This confuses the reranker. Filter negatives scoring 2-3.")
    if not recs:
        recs.append("Data quality looks solid. Proceed to training!")

    for rec in recs:
        lines.append(f"  {rec}")

    # --- Cost estimate for full-data filtering ---
    lines.append(f"\n\n>>> COST TO AUDIT FULL DATASET")
    lines.append("-" * 50)
    total_positives = len(pos_results) / SAMPLE_SIZE * 307621 if SAMPLE_SIZE else 307621
    tokens_per_call = 800  # rough avg (prompt + response)
    total_tokens = total_positives * tokens_per_call
    # Groq pricing is approximate
    lines.append(f"  To score all ~307K positives at ~{tokens_per_call} tokens/call:")
    lines.append(f"  Estimated tokens: ~{total_tokens/1e6:.0f}M")
    lines.append(f"  At Groq rates: check console.groq.com for current pricing")
    lines.append(f"  Note: Groq free tier has rate limits (~30 req/min). Upgrade if needed.")
    lines.append(f"  Tip: score only the 'bad' query types (lab, medication) to save cost")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("RERANKER TRAINING DATA — QUALITY AUDIT")
    print("=" * 60)

    # Ensure output directory exists
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # 1. Sample pairs
    positive_samples, negative_samples = load_samples()

    # 2. Score positives
    print(f"\nScoring {len(positive_samples)} positive pairs via Groq...")
    pos_results = run_audit(positive_samples)

    # Save immediately so we don't lose results if something crashes later
    with open(AUDIT_OUTPUT, "w") as f:
        for r in pos_results:
            f.write(json.dumps(r) + "\n")
    print(f"  Checkpoint: {len(pos_results)} positive scores saved to {AUDIT_OUTPUT}")

    # 3. Score negatives
    neg_results = []
    if negative_samples:
        print(f"\nScoring {len(negative_samples)} negative pairs via Groq...")
        neg_results = run_audit(negative_samples)

        # Append negative results to the same file
        with open(AUDIT_OUTPUT, "a") as f:
            for r in neg_results:
                f.write(json.dumps(r) + "\n")
        print(f"  Checkpoint: {len(neg_results)} negative scores appended to {AUDIT_OUTPUT}")

    # 4. Generate and save report
    report = generate_report(pos_results, neg_results)
    print(report)

    with open(AUDIT_REPORT, "w") as f:
        f.write(report)
    print(f"\nReport saved to {AUDIT_REPORT}")


if __name__ == "__main__":
    main()