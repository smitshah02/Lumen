from src.evals.final_eval import providers


def test_demo_provider_remains_the_safe_default():
    assert providers.provider_names() == ("demo", "synthea")
    provider = providers.get_provider()
    assert provider.name == "demo"
    assert provider.data_plane == "demo"
    assert len(provider.load_cases()) == 40


def test_synthea_provider_declares_profile_plane_and_database():
    dev = providers.get_provider("synthea", "dev")
    evaluation = providers.get_provider("synthea", "eval")
    assert dev.profile == "dev" and evaluation.profile == "eval"
    assert dev.data_plane == evaluation.data_plane == "synthea"
    assert dev.expected_database == evaluation.expected_database == "lumen_synthea"


def test_provider_fingerprint_records_source_boundary():
    fp = providers.get_provider("demo").fingerprint()
    assert fp["provider"] == "demo"
    assert fp["provider_data_plane"] == "demo"
    assert fp["manifest_sha256_matches"] is True
