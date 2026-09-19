"""
Fetch the retrieval models into LUMEN_MODELS_DIR
================================================
Downloads exactly the weights Lumen already uses, pinned to the Hugging Face
revisions recorded in results/interview_baseline/run_manifest.json, into the
directory layout src/retrieval expects. Used by the demo/cloud container, whose
/models is a persistent volume — weights are never baked into the image.

    python scripts/fetch_models.py            # skips models already present
    python scripts/fetch_models.py --verify   # print sha256 of each safetensors file

Only config/tokenizer files and model.safetensors are fetched (the repos also
carry duplicate pytorch_model.bin weights that Lumen does not need).
"""

from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = {
    # local dir name      repo id                          pinned revision
    "medcpt-query":   ("ncbi/MedCPT-Query-Encoder",   "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc"),
    "medcpt-article": ("ncbi/MedCPT-Article-Encoder", "d05a736da4bb84ee4057b7f7999485be6ed85465"),
    "bge-reranker":   ("BAAI/bge-reranker-v2-m3",     "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
}
PATTERNS = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.txt", "added_tokens.json", "sentencepiece.bpe.model", "model.safetensors"]


def _models_dir() -> Path:
    # Same resolution as src.retrieval.embeddings.MODELS_DIR, without importing torch.
    return Path(os.environ.get("LUMEN_MODELS_DIR") or Path.home() / "Lumen" / "models")


def main() -> int:
    target_root = _models_dir()
    if "--verify" in sys.argv:
        for name in MODELS:
            f = target_root / name / "model.safetensors"
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            print(f"{name}: {h.hexdigest()}")
        return 0

    from huggingface_hub import snapshot_download
    target_root.mkdir(parents=True, exist_ok=True)
    for name, (repo, rev) in MODELS.items():
        target = target_root / name
        if (target / "config.json").exists() and (target / "model.safetensors").exists():
            print(f"{name}: present, skipping")
            continue
        print(f"{name}: downloading {repo}@{rev[:10]}", flush=True)
        snapshot_download(repo_id=repo, revision=rev, local_dir=target, allow_patterns=PATTERNS)
    print(f"models ready in {target_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
