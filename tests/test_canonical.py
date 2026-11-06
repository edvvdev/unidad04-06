"""El párrafo canónico de la entrega debe coincidir con las métricas publicadas.

Si el modelo se reentrena, las cifras del párrafo canónico cambian. Este test
detecta esa desincronización en `pytest -q` (paso 2 de `scripts/run_all.sh`), en
lugar de dejarla para el final, cuando el generador del PDF aborta.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inf8239_u03_gen.metrics import (
    CLAVES_CANONICAS_DIVERSIDAD,
    extraer_parrafo_canonico,
    numero_es,
    parrafo_canonico,
)

ROOT = Path(__file__).resolve().parents[1]
RUTA_METRICAS = ROOT / "reports" / "ejercicio06_metricas.json"
RUTA_MODEL_CARD = ROOT / "MODEL_CARD.md"
RUTA_CIERRE = ROOT / "docs" / "CIERRE_INTERPRETATIVO_LAB10.md"


@pytest.fixture(scope="module")
def metricas() -> dict[str, object]:
    if not RUTA_METRICAS.exists():
        pytest.skip("No hay métricas publicadas: ejecuta el entrenamiento o el notebook")
    return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))


def test_el_parrafo_canonico_coincide_con_el_json(metricas):
    canonico = extraer_parrafo_canonico(RUTA_MODEL_CARD.read_text(encoding="utf-8"))
    assert canonico == parrafo_canonico(metricas)


def test_el_cierre_cita_literalmente_el_parrafo_canonico(metricas):
    canonico = extraer_parrafo_canonico(RUTA_MODEL_CARD.read_text(encoding="utf-8"))
    assert canonico in RUTA_CIERRE.read_text(encoding="utf-8")


def test_las_cifras_publicadas_son_las_del_json(metricas):
    canonico = extraer_parrafo_canonico(RUTA_MODEL_CARD.read_text(encoding="utf-8"))
    for clave in CLAVES_CANONICAS_DIVERSIDAD:
        assert numero_es(float(metricas[clave])) in canonico


def test_el_json_declara_su_origen_y_su_fecha(metricas):
    assert metricas["origen"].endswith(".ipynb") or metricas["origen"].endswith(".py")
    assert str(metricas["fecha_ejecucion"]).startswith("20")
