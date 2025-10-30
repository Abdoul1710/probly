import importlib
import pytest

@pytest.mark.parametrize("mod", [
    "probly.transformation.ensemble",
    "probly.transformation.ensemble.common",
    "probly.transformation.ensemble.torch",
])
def test_modules_import(mod):
    """
     Lässt sich das Ensemble-Modul importieren?
    Falls ein Teilmodul (z. B. .torch) optional/nicht installiert ist,
    wird der Test sauber 'skipped' statt zu fehlschlagen.
    """
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as e:
        pytest.skip(f"Optionales Modul nicht vorhanden: {mod} ({e})")
