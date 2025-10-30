import importlib
import pytest

@pytest.mark.parametrize("mod", [
    "probly.transformation.ensemble",
    "probly.transformation.ensemble.common",
    "probly.transformation.ensemble.torch",
])
def test_modules_import(mod):
    """Lässt sich das Ensemble-Modul importieren?"""
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as e:
        pytest.skip(f"Optionales Modul nicht vorhanden: {mod} ({e})")
