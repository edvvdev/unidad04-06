import numpy as np
import pytest

from inf8239_u03_gen.metrics import (
    CLAVES_CANONICAS,
    build_report,
    flatten,
    mean_nearest_neighbor_distance,
    mean_pairwise_distance,
)


def test_flatten_contract():
    images = np.zeros((3, 28, 28, 1), dtype="float32")
    assert flatten(images).shape == (3, 784)


def test_mean_pairwise_distance_matches_brute_force():
    points = np.random.default_rng(0).random((25, 12))
    esperado = np.mean([np.linalg.norm(points[i] - points[j]) for i in range(25) for j in range(i + 1, 25)])
    assert mean_pairwise_distance(points) == pytest.approx(esperado, rel=1e-9)


def test_mean_nearest_neighbor_distance_matches_brute_force():
    reference = np.random.default_rng(1).random((40, 12))
    samples = np.random.default_rng(2).random((7, 12))
    esperado = np.mean([np.min(np.linalg.norm(reference - sample, axis=1)) for sample in samples])
    assert mean_nearest_neighbor_distance(samples, reference) == pytest.approx(esperado, rel=1e-9)


def test_build_report_keeps_the_canonical_keys():
    valores = {clave: 0 for clave in CLAVES_CANONICAS if clave not in {"origen", "fecha_ejecucion"}}
    reporte = build_report(valores, origen="tests")
    assert list(reporte) == list(CLAVES_CANONICAS)
    assert reporte["origen"] == "tests"
    assert reporte["fecha_ejecucion"]


def test_build_report_rejects_missing_keys():
    with pytest.raises(ValueError, match="Faltan claves"):
        build_report({}, origen="tests")
