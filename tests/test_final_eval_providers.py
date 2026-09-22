from src.evals.final_eval import providers


def test_demo_provider_is_the_only_registered_safe_default():
    assert providers.provider_names() == ("demo",)
    provider = providers.get_provider()
    assert provider.name == "demo"
    assert provider.data_plane == "demo"
    assert len(provider.load_cases()) == 40


def test_provider_fingerprint_records_source_boundary():
    fp = providers.get_provider("demo").fingerprint()
    assert fp["provider"] == "demo"
    assert fp["provider_data_plane"] == "demo"
    assert fp["manifest_sha256_matches"] is True
