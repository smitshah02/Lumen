"""Static deployment guards; no SSH, Docker, cloud, or process mutation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_process_stop_uses_validated_pid_files_not_broad_matching():
    script = (ROOT / "scripts/start_cloud_demo.sh").read_text()
    assert "pkill" not in script
    assert "pid_matches" in script and "kill -TERM" in script
    assert "api.pid" in script and "ollama.pid" in script
    assert "ps -p" in script


def test_runpod_sync_allowlist_excludes_restricted_roots():
    sync = (ROOT / "scripts/sync_to_pod.sh").read_text()
    files_block = sync.split("FILES=(", 1)[1].split(")", 1)[0]
    allowlist = files_block.split()
    forbidden_roots = {"data", "models", "backups", ".env", "results", "API_keys"}
    assert not {path.split("/", 1)[0] for path in allowlist} & forbidden_roots
    assert "configs/models.json" in files_block


def test_bootstrap_remains_demo_only_with_restricted_path_guard():
    script = (ROOT / "scripts/bootstrap_pod.sh").read_text()
    assert 'export LUMEN_DATA_PLANE=demo' in script
    assert "src/reranker/reranker_training_data" in script
    assert "research-era path" in script
