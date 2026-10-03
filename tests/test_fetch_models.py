"""Model provenance manifests use tiny synthetic files; no downloads."""

import json

import pytest

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


def test_adopt_local_records_weights_that_match_the_pinned_revision(tmp_path):
    target = tmp_path / "example"
    make_model(target)
    published = lambda spec: models.sha256_file(target / "model.safetensors")
    assert models.adopt_local_model("example", SPEC, target, published=published) == []
    manifest = json.loads((target / models.LOCAL_MANIFEST).read_text())
    assert manifest["revision"] == SPEC["revision"]
    assert models.verify_local_model("example", SPEC, target) == []


def test_adopt_local_refuses_weights_that_are_not_the_pinned_revision(tmp_path):
    target = tmp_path / "example"
    make_model(target)
    errors = models.adopt_local_model("example", SPEC, target, published=lambda spec: "0" * 64)
    assert errors == ["local model.safetensors does not match the pinned revision"]
    assert not (target / models.LOCAL_MANIFEST).exists()      # nothing recorded
    assert models.adopt_local_model("example", SPEC, target, published=lambda spec: None)


def test_adopt_local_cli_refuses_weights_that_do_not_match(tmp_path, monkeypatch, capsys):
    make_model(tmp_path / "example")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(models, "selected_models", lambda profile: {"example": SPEC})
    monkeypatch.setattr(models, "published_weights_sha256", lambda spec: "0" * 64)
    assert models.main(["--adopt-local"]) == 1
    assert not (tmp_path / "example" / models.LOCAL_MANIFEST).exists()


def test_adopt_local_cli_reports_an_already_verified_model_as_such(tmp_path, monkeypatch, capsys):
    make_model(tmp_path / "example")
    models.write_local_manifest("example", SPEC, tmp_path / "example")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(models, "selected_models", lambda profile: {"example": SPEC})
    monkeypatch.setattr(models, "published_weights_sha256", lambda spec: pytest.fail("no Hub call needed"))
    assert models.main(["--adopt-local"]) == 0
    assert "already verified" in capsys.readouterr().out
