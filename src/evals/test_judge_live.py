from src.evals.llm_judge import LLMJudge
from src.llm.local_client import FAST_MODEL, build_call_fn

judge = LLMJudge(
    model=FAST_MODEL,
    call_fn=build_call_fn(tier="fast", json_mode=True, max_tokens=200),
    max_workers=2,   # local single-GPU: 8 parallel calls will thrash
)
tests = [
    ("abnormal potassium lab results", "Discharge labs: Potassium 6.1 mEq/L, critical high. Given kayexalate."),
    ("abnormal potassium lab results", "CBC: WBC 7.3 RBC 3.72 Hgb 11.2 Hct 33.8 Plt 210"),
    ("swollen legs fluid overload",    "CERVICAL SPINE: vertebral body heights are preserved."),
]
for q, chunk in tests:
    r = judge.judge(q, chunk)
    print(f"score={r.score}  relevant={r.is_relevant()}  err={r.error}  reason={r.reason!r}")