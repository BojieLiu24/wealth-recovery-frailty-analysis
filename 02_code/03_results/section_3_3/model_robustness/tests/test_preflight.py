"""尺度、基函数、历史异常和只读重构的数学验收。"""
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import pytest

from common import repo_path, write_json, read_json
from prepare_inputs import classify_history, reconstruct_primary
from specification import KNOTS
from time_basis import rcs_basis, scale_equivalence_checks, covariance_to_years, covariance_to_scaled


def test_scale_equivalence_without_fit():
    result = scale_equivalence_checks()
    assert result["status"] == "PASS" and result["fit_calls"] == 0
    assert len(result["cases"]) == 6
    assert {r["observations"] for r in result["cases"]} == {2, 3, 5}
    assert max(r["loglik_absolute_error"] for r in result["cases"]) < 1e-8


@pytest.mark.parametrize("correlation", [-.8, 0., .8])
def test_scale_variance_covariance_and_correlation(correlation):
    g = np.array([[.02, correlation*np.sqrt(.02*.007)], [correlation*np.sqrt(.02*.007), .007]])
    a = covariance_to_years(g)
    assert a[1, 1] == g[1, 1]/64
    assert a[0, 1] == g[0, 1]/8
    np.testing.assert_allclose(covariance_to_scaled(a), g, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a[0, 1]/np.sqrt(a[0, 0]*a[1, 1]), correlation, atol=1e-12)


@pytest.mark.parametrize("knots", KNOTS.values())
def test_rcs_natural_basis_and_derivative(knots):
    t = np.unique(np.r_[np.arange(0, 26, .25), knots])
    ours = np.column_stack([np.ones(len(t)), t, rcs_basis(t, knots)])
    native = np.asarray(patsy.dmatrix("cr(t, knots=(k,), lower_bound=0, upper_bound=b)-1",
                                     {"t": t, "k": knots[1], "b": knots[2]}))
    q, _ = np.linalg.qr(ours)
    np.testing.assert_allclose(native, q @ (q.T @ native), rtol=1e-10, atol=1e-12)
    assert rcs_basis(0., knots) == 0
    locations = np.array([.5, 2., 6.1, 10.7, 20.])
    eps = 1e-5
    numeric = (rcs_basis(locations+eps, knots)-rcs_basis(locations-eps, knots))/(2*eps)
    np.testing.assert_allclose(rcs_basis(locations, knots, 1), numeric, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(rcs_basis([knots[2]+1, knots[2]+4], knots, 2), 0, atol=1e-12)
    for knot in knots[1:]:
        for derivative in (0, 1, 2):
            assert abs(float(rcs_basis(knot+1e-7, knots, derivative)-rcs_basis(knot-1e-7, knots, derivative))) < 1e-5


def test_rcs_rejects_invalid_knots():
    for knots in ([1., 2., 3.], [0., 2., 2.], [0., np.nan, 8.]):
        with pytest.raises(ValueError):
            rcs_basis([0., 1.], knots)


def test_history_presentations_do_not_waive_statistical_exception(tmp_path):
    manuscript = tmp_path / "07_manuscript/supplementary_materials/manuscript_v1.docx"
    manuscript.parent.mkdir(parents=True)
    manuscript.write_bytes(b"new document")
    from common import sha
    old_path = "07_manuscript/manuscript_v1.docx"
    stat = "00_admin/model_manifests/primary_lmm/primary_lmm_v1_contract.json"
    rows = classify_history({old_path: "old", stat: "old"}, tmp_path, {manuscript.relative_to(tmp_path).as_posix(): sha(manuscript)})
    assert rows[0]["classification"] == "NONBLOCKING_HISTORICAL_EXCEPTION"
    assert rows[0]["later_evidence_verified"] and not rows[0]["current_candidates"][0]["matches_old_sha"]
    assert rows[1]["classification"] == "BLOCKING_STATISTICAL_SOURCE_EXCEPTION"


def test_path_escape_and_overwrite_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        repo_path("../outside", tmp_path)
    path = tmp_path / "report.json"
    write_json(path, {"中文": "保留"})
    assert read_json(path)["中文"] == "保留"
    with pytest.raises(FileExistsError):
        write_json(path, {})


def test_primary_reconstruction_requires_exact_anchor_summary():
    frame = pd.DataFrame({"person_id": ["a", "a", "b", "b", "b"], "time_years": [0., 2., 0., 1.7, 4.2],
                          "fi_primary": [.15, .18, .3, .28, .34]})
    x = np.column_stack([np.ones(5), frame.time_years])
    beta = np.array([.2, .005])
    residual = frame.fi_primary.to_numpy()-x @ beta
    bhat = np.array([.008*residual[:2].sum()/(.005+2*.008)]*2 + [.008*residual[2:].sum()/(.005+3*.008)]*3)
    fitted = x @ beta + bhat
    r = frame.fi_primary.to_numpy()-fitted
    d = {"random_intercept_variance": .008, "residual_variance": .005,
         "residual_mean": r.mean(), "residual_sd": r.std(ddof=1), "residual_rmse": np.sqrt((r*r).mean()),
         "residual_time_correlation": np.corrcoef(r, frame.time_years)[0, 1], "residual_fitted_correlation": np.corrcoef(r, fitted)[0, 1]}
    assert reconstruct_primary(frame, pd.DataFrame(x), beta, d)["status"] == "PASS"
    d["residual_sd"] += 1e-7
    with pytest.raises(ValueError):
        reconstruct_primary(frame, pd.DataFrame(x), beta, d)
