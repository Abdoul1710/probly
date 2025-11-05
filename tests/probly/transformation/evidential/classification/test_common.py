from __future__ import annotations

import pytest

# Importiere dein Modul
# Falls dein Code in einer Datei wie evidential.py liegt:
from probly.transformation.evidential.classification.common import (
    evidential_classification,
    register,
)


# === Test 1: Kein Appender registriert -> Fehler ===
def test_evidential_classification_raises_not_implemented_error():
    class DummyPredictor:
        pass

    dummy = DummyPredictor()

    with pytest.raises(NotImplementedError) as excinfo:
        evidential_classification(dummy)

    assert "No evidential classification appender registered" in str(excinfo.value)


# === Test 2: Ein registrierter Appender wird korrekt aufgerufen ===
def test_registered_appender_is_called():
    class DummyPredictor:
        def __init__(self, name) -> None:
            self.name = name

    def dummy_appender(base):
        # Wird aufgerufen, wenn Registrierung funktioniert
        return f"Evidential({base.name})"

    # Registrierung
    register(DummyPredictor, dummy_appender)

    dummy = DummyPredictor("ModelX")
    result = evidential_classification(dummy)

    assert result == "Evidential(ModelX)"


# === Test 3: Mehrere Typen können unabhängig registriert werden ===
def test_multiple_types_are_handled_independently():
    class ModelA:
        def __init__(self, id) -> None:
            self.id = id

    class ModelB:
        def __init__(self, id) -> None:
            self.id = id

    def appender_a(base):
        return f"A_Enhanced({base.id})"

    def appender_b(base):
        return f"B_Enhanced({base.id})"

    register(ModelA, appender_a)
    register(ModelB, appender_b)

    a = ModelA("001")
    b = ModelB("002")

    result_a = evidential_classification(a)
    result_b = evidential_classification(b)

    assert result_a == "A_Enhanced(001)"
    assert result_b == "B_Enhanced(002)"
