"""Download or verify Lumen's pinned Hugging Face model snapshots.

The authoritative repositories, revisions, and runtime/evaluation profiles are
in ``configs/models.json``. A directory is ready only when its local provenance
manifest exists and every recorded file hash still matches.

    python scripts/fetch_models.py
    python scripts/fetch_models.py --verify
    python scripts/fetch_models.py --adopt-local
    python scripts/fetch_models.py --profile legacy-eval

``--adopt-local`` is for weights already on disk without a manifest. It
downloads nothing: it asks the Hub for the pinned revision's published
``model.safetensors`` hash (repository and revision identifiers are the only
thing sent) and writes the manifest only when the local file matches.

The runtime profile contains MedCPT query/article encoders and the BGE
reranker. The retired MedCPT cross-encoder comparison is fetched only through
``legacy-eval`` or ``all``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODELS_CONFIG, MODELS_CONFIG_PATH, MODELS_DIR

PATTERNS = [
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "added_tokens.json",
    "sentencepiece.bpe.model", "model.safetensors",
]
LOCAL_MANIFEST = ".lumen-model.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_models(profile: str) -> dict[str, dict]:
    models = MODELS_CONFIG["hugging_face"]
    if profile == "all":
        return dict(models)
    return {name: spec for name, spec in models.items() if spec["profile"] == profile}


def _snapshot_files(target: Path) -> list[Path]:
    return sorted(
        path for path in target.rglob("*")
        if path.is_file() and path.name != LOCAL_MANIFEST and ".cache" not in path.parts
    )


def build_local_manifest(name: str, spec: dict, target: Path) -> dict:
    files = _snapshot_files(target)
    if not (target / "config.json").is_file() or not (target / "model.safetensors").is_file():
        raise FileNotFoundError(f"{name}: required config.json/model.safetensors is missing")
    return {
        "schema_version": 1,
        "name": name,
        "profile": spec["profile"],
        "repo": spec["repo"],
        "revision": spec["revision"],
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            str(path.relative_to(target)): {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }


def write_local_manifest(name: str, spec: dict, target: Path) -> None:
    payload = build_local_manifest(name, spec, target)
    destination = target / LOCAL_MANIFEST
    temporary = target / f"{LOCAL_MANIFEST}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def verify_local_model(name: str, spec: dict, target: Path) -> list[str]:
    manifest_path = target / LOCAL_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"missing or invalid {LOCAL_MANIFEST}: {exc}"]
    errors = []
    for key in ("name", "profile", "repo", "revision"):
        expected = name if key == "name" else spec[key]
        if manifest.get(key) != expected:
            errors.append(f"{key} mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        errors.append("manifest contains no file hashes")
        return errors
    current_files = {
        str(path.relative_to(target)) for path in _snapshot_files(target)
    }
    unrecorded = sorted(current_files - set(files))
    if unrecorded:
        errors.append(f"unrecorded files: {', '.join(unrecorded)}")
    for relative, recorded in files.items():
        path = target / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        if path.stat().st_size != recorded.get("size"):
            errors.append(f"size mismatch: {relative}")
            continue
        if sha256_file(path) != recorded.get("sha256"):
            errors.append(f"sha256 mismatch: {relative}")
    for required in ("config.json", "model.safetensors"):
        if required not in files:
            errors.append(f"required file not recorded: {required}")
    return errors


def published_weights_sha256(spec: dict) -> str | None:
    """SHA-256 the Hub publishes for model.safetensors at the pinned revision."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
    for sibling in info.siblings or []:
        if sibling.rfilename == "model.safetensors" and sibling.lfs:
            lfs = sibling.lfs
            return lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
    return None


def adopt_local_model(name: str, spec: dict, target: Path, published=None) -> list[str]:
    """Write the manifest for weights already on disk, if they are the pinned ones."""
    weights = target / "model.safetensors"
    if not weights.is_file() or not (target / "config.json").is_file():
        return ["required config.json/model.safetensors is missing"]
    expected = (published or published_weights_sha256)(spec)
    if not expected:
        return ["the Hub publishes no model.safetensors hash for the pinned revision"]
    if sha256_file(weights) != expected:
        return ["local model.safetensors does not match the pinned revision"]
    write_local_manifest(name, spec, target)
    return verify_local_model(name, spec, target)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("runtime", "legacy-eval", "all"), default="runtime")
    parser.add_argument("--verify", action="store_true", help="verify locally; never download")
    parser.add_argument("--adopt-local", action="store_true",
                        help="record existing local weights after checking their hash against "
                             "the pinned revision; never downloads weights")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    chosen = selected_models(args.profile)
    if not chosen:
        raise RuntimeError(f"no models configured for profile {args.profile!r}")

    if args.verify:
        failed = False
        for name, spec in chosen.items():
            errors = verify_local_model(name, spec, MODELS_DIR / name)
            print(f"{name}: {'FAILED — ' + '; '.join(errors) if errors else 'verified'}")
            failed |= bool(errors)
        return 1 if failed else 0

    if args.adopt_local:
        failed = False
        for name, spec in chosen.items():
            target = MODELS_DIR / name
            if not verify_local_model(name, spec, target):
                print(f"{name}: already verified, nothing to adopt")
                continue
            errors = adopt_local_model(name, spec, target)
            print(f"{name}: {'REFUSED — ' + '; '.join(errors) if errors else 'adopted, matches ' + spec['revision'][:12]}")
            failed |= bool(errors)
        return 1 if failed else 0

    from huggingface_hub import snapshot_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in chosen.items():
        target = MODELS_DIR / name
        errors = verify_local_model(name, spec, target)
        if not errors:
            print(f"{name}: verified, skipping")
            continue
        print(f"{name}: local verification failed ({'; '.join(errors)}); refreshing pinned snapshot")
        snapshot_download(
            repo_id=spec["repo"], revision=spec["revision"],
            local_dir=target, allow_patterns=PATTERNS,
        )
        write_local_manifest(name, spec, target)
        remaining = verify_local_model(name, spec, target)
        if remaining:
            raise RuntimeError(f"{name}: post-download verification failed: {'; '.join(remaining)}")
        print(f"{name}: verified {spec['repo']}@{spec['revision'][:12]}")
    print(f"models ready in {MODELS_DIR} (source: {MODELS_CONFIG_PATH})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
