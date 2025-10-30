import importlib

def test_ensemble_has_public_api():
    ens_mod = importlib.import_module("probly.transformation.ensemble")

    assert hasattr(ens_mod, "ensemble")
    assert hasattr(ens_mod, "register")

    assert callable(ens_mod.ensemble)
