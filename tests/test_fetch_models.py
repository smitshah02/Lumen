"""Model provenance manifests use tiny synthetic files; no downloads."""

import json

from scripts import fetch_models as models


SPEC = {"profile": "runtime", "repo": "example/model", "revision": "a" * 40}


def make_model(path):
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"tiny synthetic weight")


def test_runtime_profile_excludes_legacy_cross_encoder():
    assert "medcpt-cross-encoder" not in models.selected_models("runtime")
    assert models.selected_models("legacy-eval") == {
        "medcpt-cross-encoder": models.MODELS_CONFIG["hugging_face"]["medcpt-cross-encoder"]
    }


def test_local_manifest_detects_content_and_revision_drift(tmp_path):
    target = tmp_path / "example"
    make_model(target)
    models.write_local_manifest("example", SPEC, target)
    assert models.verify_local_model("example", SPEC, target) == []

    (target / "model.safetensors").write_bytes(b"tampered")
    assert any("size mismatch" in error for error in models.verify_local_model("example", SPEC, target))

    changed = {**SPEC, "revision": "b" * 40}
    assert "revision mismatch" in models.verify_local_model("example", changed, target)


def test_directory_without_provenance_manifest_is_not_trusted(tmp_path):
    target = tmp_path / "example"
    make_model(target)
    errors = models.verify_local_model("example", SPEC, target)
    assert errors and "missing or invalid" in errors[0]


def test_unrecorded_snapshot_file_is_not_trusted(tmp_path):
    target = tmp_path / "example"
    make_model(target)
    models.write_local_manifest("example", SPEC, target)
    (target / "unexpected.bin").write_bytes(b"stale")
    assert any("unrecorded files" in error for error in
               models.verify_local_model("example", SPEC, target))


def test_written_manifest_records_hashes_not_absolute_paths(tmp_path):
    target = tmp_path / "example"
    make_model(target)
    models.write_local_manifest("example", SPEC, target)
    payload = json.loads((target / models.LOCAL_MANIFEST).read_text())
    assert set(payload["files"]) == {"config.json", "model.safetensors"}
    assert len(payload["files"]["model.safetensors"]["sha256"]) == 64
