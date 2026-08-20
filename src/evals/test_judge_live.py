from src.evals.ollama_backend import make_ollama_judge

judge = make_ollama_judge()  # qwen2.5:14b via local Ollama, no API key

tests = [
    ("abnormal potassium lab results", "Discharge labs: Potassium 6.1 mEq/L, critical high. Given kayexalate."),
    ("abnormal potassium lab results", "CBC: WBC 7.3 RBC 3.72 Hgb 11.2 Hct 33.8 Plt 210"),
    ("swollen legs fluid overload",    "CERVICAL SPINE: vertebral body heights are preserved."),
]
for q, chunk in tests:
    r = judge.judge(q, chunk)
    print(f"score={r.score}  relevant={r.is_relevant()}  err={r.error}  reason={r.reason!r}")