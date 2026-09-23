#!/usr/bin/env python3
"""Generate a pinned, CSV-only Synthea cohort and provenance manifest.

The official release JAR and, when needed on Apple Silicon, a portable Java 17
toolchain are cached under ignored ``.cache/synthea``. Generated data is kept
under ignored ``data/synthea/<profile>`` by default.

    .venv/bin/python scripts/generate_synthea.py dev
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "synthea.json"
CACHE_ROOT = ROOT / ".cache" / "synthea"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(url: str, expected_sha256: str, destination: Path) -> Path:
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    print(f"downloading {url}")
    try:
        try:
            import certifi
            tls_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            tls_context = ssl.create_default_context()
        request = urllib.request.Request(url, headers={"User-Agent": "Lumen-Synthea/1"})
        with urllib.request.urlopen(request, context=tls_context) as response, \
                temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"checksum mismatch for {destination.name}: {actual} != {expected_sha256}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def platform_key() -> str:
    system = {"darwin": "darwin", "linux": "linux"}.get(platform.system().lower())
    machine = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x86_64"}.get(
        platform.machine().lower()
    )
    return f"{system}-{machine}" if system and machine else "unsupported"


def java_major(java: Path) -> tuple[int, str]:
    result = subprocess.run(
        [str(java), "-version"], text=True, capture_output=True, check=True
    )
    detail = (result.stderr or result.stdout).strip().splitlines()[0]
    match = re.search(r'version "(\d+)', detail)
    if not match:
        raise RuntimeError(f"could not parse Java version from: {detail}")
    return int(match.group(1)), detail


def resolve_java(config: dict) -> tuple[Path, str]:
    configured = os.environ.get("SYNTHEA_JAVA")
    candidate = Path(configured).expanduser() if configured else None
    if candidate is None:
        found = shutil.which("java")
        candidate = Path(found) if found else None
    minimum = int(config["java"]["minimum_major"])
    if candidate is not None:
        try:
            major, detail = java_major(candidate)
            if major >= minimum:
                return candidate, detail
        except (OSError, RuntimeError, subprocess.CalledProcessError):
            pass

    key = platform_key()
    spec = config["java"].get("portable", {}).get(key)
    if not spec:
        raise RuntimeError(
            f"Java {minimum}+ is required; set SYNTHEA_JAVA to its java executable "
            f"(no pinned portable toolchain is configured for {key})"
        )
    toolchain = CACHE_ROOT / "java" / key / spec["version"]
    java = toolchain / spec["java_relative_path"]
    if not java.is_file():
        archive = download_verified(
            spec["archive_url"], spec["archive_sha256"],
            CACHE_ROOT / "downloads" / Path(spec["archive_url"]).name,
        )
        toolchain.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(toolchain, filter="data")
    major, detail = java_major(java)
    if major < minimum:
        raise RuntimeError(f"pinned Java is {major}; Synthea requires {minimum}+")
    return java, detail


def generator_jar(config: dict) -> Path:
    spec = config["generator"]
    destination = CACHE_ROOT / spec["version"] / spec["artifact"]
    return download_verified(spec["artifact_url"], spec["artifact_sha256"], destination)


def invocation(java: Path, jar: Path, profile: dict, output: Path, exporter: dict) -> list[str]:
    settings = {
        "exporter.baseDirectory": str(output),
        "exporter.csv.export": str(exporter["csv"]).lower(),
        "exporter.csv.append_mode": str(exporter["csv_append_mode"]).lower(),
        "exporter.csv.folder_per_run": str(exporter["csv_folder_per_run"]).lower(),
        "exporter.years_of_history": str(exporter["years_of_history"]),
        "exporter.fhir.export": str(exporter["fhir"]).lower(),
        "exporter.fhir.transaction_bundle": str(exporter["fhir_transaction_bundle"]).lower(),
        "exporter.fhir_stu3.export": str(exporter["fhir_stu3"]).lower(),
        "exporter.fhir_dstu2.export": str(exporter["fhir_dstu2"]).lower(),
        "exporter.hospital.fhir.export": str(exporter["hospital_fhir"]).lower(),
        "exporter.practitioner.fhir.export": str(exporter["practitioner_fhir"]).lower(),
        "exporter.ccda.export": str(exporter["ccda"]).lower(),
        "exporter.json.export": str(exporter["json"]).lower(),
        "exporter.text.export": str(exporter["text"]).lower(),
        "exporter.clinical_note.export": str(exporter["clinical_note"]).lower(),
        "exporter.metadata.export": str(exporter["metadata"]).lower(),
        "exporter.enable_custom_exporters": str(exporter["custom_exporters"]).lower(),
        "generate.thread_pool_size": str(profile["thread_pool_size"]),
        "generate.log_patients.detail": "none",
    }
    command = [
        str(java), "-jar", str(jar),
        "-s", str(profile["population_seed"]),
        "-cs", str(profile["clinician_seed"]),
        "-p", str(profile["population"]),
        "-r", profile["reference_date"],
        "-e", profile["end_date"],
        "-o", str(profile["overflow_population"]).lower(),
    ]
    command.extend(f"--{key}={value}" for key, value in settings.items())
    command.append(profile["state"])
    if profile.get("city"):
        command.append(profile["city"])
    return command


def csv_metadata(csv_dir: Path) -> dict[str, dict]:
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"Synthea produced no CSV files in {csv_dir}")
    metadata = {}
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            rows = sum(1 for _ in reader)
        metadata[path.name] = {
            "rows": rows,
            "sha256": sha256_file(path),
            "columns": header or [],
        }
    return metadata


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def dataset_identity(manifest: dict) -> dict:
    """Deterministic dataset identity; intentionally excludes timestamps."""
    return {
        "schema_version": manifest["schema_version"],
        "synthetic": manifest["synthetic"],
        "profile": manifest["profile"],
        "synthea": manifest["synthea"],
        "population_seed": manifest["population_seed"],
        "clinician_seed": manifest["clinician_seed"],
        "requested_population": manifest["requested_population"],
        "actual_patient_count": manifest["actual_patient_count"],
        "generation_parameters": manifest["generation_parameters"],
        "config_sha256": manifest["config_sha256"],
        "csv_files": manifest["csv_files"],
    }


def identity_sha256(identity: dict) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", default="dev")
    parser.add_argument(
        "--output", type=Path,
        help="output directory (default: data/synthea/<profile>)",
    )
    parser.add_argument("--verify-only", action="store_true",
                        help="validate an existing frozen profile without generating it")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != 1:
        raise RuntimeError("unsupported configs/synthea.json schema_version")
    try:
        profile = config["profiles"][args.profile]
    except KeyError:
        raise SystemExit(
            f"unknown profile {args.profile!r}; choose: {', '.join(sorted(config['profiles']))}"
        ) from None
    if not profile.get("enabled"):
        raise SystemExit(f"profile {args.profile!r} is defined but generation is disabled")

    output = (args.output or ROOT / "data" / "synthea" / args.profile).expanduser().resolve()
    if args.verify_only:
        manifest_path = output / "generation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = csv_metadata(output / "csv")
        if manifest.get("profile") != args.profile or manifest.get("synthetic") is not True:
            raise RuntimeError("generation manifest profile/synthetic identity mismatch")
        if files != manifest.get("csv_files"):
            raise RuntimeError("generation manifest CSV counts, columns, or hashes mismatch")
        identity = manifest.get("dataset_identity") or dataset_identity(manifest)
        fingerprint = identity_sha256(identity)
        if manifest.get("dataset_fingerprint") not in (None, fingerprint):
            raise RuntimeError("generation manifest dataset fingerprint mismatch")
        print(json.dumps({
            "profile": args.profile,
            "actual_patients": files.get("patients.csv", {}).get("rows"),
            "csv_files": {name: {"rows": meta["rows"], "sha256": meta["sha256"]}
                          for name, meta in files.items()},
            "dataset_fingerprint": fingerprint,
            "manifest": display_path(manifest_path),
        }, indent=2, sort_keys=True))
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    java, java_detail = resolve_java(config)
    jar = generator_jar(config)

    staging = Path(tempfile.mkdtemp(prefix=f".{args.profile}-", dir=output.parent))
    try:
        command = invocation(java, jar, profile, staging, config["exporter"])
        printable = ["<java>" if i == 0 else "<synthea-jar>" if i == 2 else value
                     for i, value in enumerate(command)]
        print("running:", " ".join(printable))
        subprocess.run(command, cwd=ROOT, check=True)
        files = csv_metadata(staging / "csv")
        actual = files.get("patients.csv", {}).get("rows")
        if actual != profile["population"]:
            raise RuntimeError(
                f"requested {profile['population']} patients but patients.csv contains {actual}"
            )
        unexpected = sorted(
            path.name for path in staging.iterdir() if path.name != "csv"
        )
        if unexpected:
            raise RuntimeError(f"unexpected non-CSV output directories/files: {unexpected}")

        manifest = {
            "schema_version": 1,
            "synthetic": True,
            "profile": args.profile,
            "synthea": {
                "project": config["generator"]["project"],
                "version": config["generator"]["version"],
                "artifact": config["generator"]["artifact"],
                "artifact_sha256": sha256_file(jar),
                "generator_checksum": sha256_file(jar),
            },
            "generator_checksum": sha256_file(jar),
            "java": java_detail,
            "population_seed": profile["population_seed"],
            "clinician_seed": profile["clinician_seed"],
            "requested_population": profile["population"],
            "actual_patient_count": actual,
            "generation_parameters": {
                **profile,
                "exporter": config["exporter"],
            },
            "invocation": printable,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "csv_files": files,
        }
        manifest["dataset_identity"] = dataset_identity(manifest)
        manifest["dataset_fingerprint"] = identity_sha256(manifest["dataset_identity"])
        (staging / "generation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps({
        "output": display_path(output),
        "manifest": display_path(output / "generation_manifest.json"),
        "patients": actual,
        "csv_files": {name: details["rows"] for name, details in files.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
