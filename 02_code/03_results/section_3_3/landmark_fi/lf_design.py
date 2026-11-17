"""KLoSA等价基底与FI变化量等价检查；只处理设计，不拟合结局。"""
import numpy as np
import pandas as pd
from lf_common import TERM, require
from lf_spec import Q1, Q1_TIME, Q1_COMBINED, LM_MAIN, LM_TIME, POLICY


def spectrum(x):
    a = np.asarray(x, dtype=float)
    require(np.isfinite(a).all(), '设计包含非有限值')
    s = np.linalg.svd(a, compute_uv=False)
    return {'columns': a.shape[1], 'rank': int(np.linalg.matrix_rank(a)),
            'singular_values': s.tolist(), 'condition_number': float(s[0]/s[-1])}


def parameterize(original, time, cohort, spec):
    original = original.astype(float)
    names = list(original.columns)
    p = 42 if spec == 'PO01' else 44
    require(spec in ('PO01', 'LM01') and len(names) == p and len(set(names)) == p, '原设计身份或列数错误')
    before = spectrum(original)
    core = names.index(TERM)
    if cohort == 'KLoSA':
        require(before['rank'] == p-1, 'KLoSA额外秩亏或预定依赖消失，停止')
        require(Q1 in names and Q1_TIME in names, 'KLoSA缺少预定Q1列')
        active = original[Q1].to_numpy(float) != 0
        require(int(active.sum()) == 3 and np.asarray(time)[active].tolist() == [23/12]*3, 'Q1支持事实变化')
        np.testing.assert_array_equal(original[Q1_TIME], original[Q1]*(23/12))
        keep = [j for j, n in enumerate(names) if n != Q1_TIME]
        basis = original.iloc[:, keep].copy().rename(columns={Q1: Q1_COMBINED})
        mapping = np.eye(p)[keep]
        mapping[keep.index(names.index(Q1)), names.index(Q1_TIME)] = 23/12
        null = np.zeros(p); null[names.index(Q1)] = -23/12; null[names.index(Q1_TIME)] = 1
        dependency = {'relation': 'Q1*time_years=(23/12)*Q1', 'rational_time': [23, 12],
            'combined_parameter': 'kappa_Q1=gamma_Q1+(23/12)*eta_Q1', 'null_vector': null.tolist(),
            'nonidentifiable_original_terms': [Q1, Q1_TIME], 'Q1_post_rows': 3,
            'independent_Q1_CI_allowed': False, 'Q1_unsupported_extrapolation_allowed': False}
    else:
        require(cohort in ('HRS', 'ELSA') and before['rank'] == p, '原设计降秩，禁止自动删列')
        basis = original.copy(); mapping = np.eye(p); null = np.zeros(p)
        dependency = None
    after = spectrum(basis)
    require(after['rank'] == after['columns'], '基底仍降秩')
    reconstructed = basis.to_numpy(float) @ mapping
    np.testing.assert_allclose(reconstructed, original.to_numpy(float), rtol=POLICY['equivalence_rtol'], atol=POLICY['equivalence_atol'])
    contrast = np.zeros(len(basis.columns)); contrast[list(basis).index(TERM)] = 1
    target = np.zeros(p); target[core] = 1
    np.testing.assert_array_equal(contrast @ mapping, target)
    np.testing.assert_allclose(original.to_numpy(float) @ null, 0, rtol=0, atol=1e-12)
    return basis, {'schema': 'landmark_fi_design_parameterization_v1', 'cohort': cohort, 'spec_id': spec,
        'original_columns': names, 'basis_columns': list(basis), 'original_spectrum': before, 'basis_spectrum': after,
        'matrix_relation': 'X_original=X_basis@A; beta_basis=A@beta_original', 'A': mapping.tolist(),
        'dependency': dependency, 'factorization_max_absolute_error': float(np.max(abs(reconstructed-original.to_numpy(float)))),
        'core_estimability': {'term': TERM, 'basis_contrast': contrast.tolist(), 'original_contrast': target.tolist(),
            'core_null_coefficient': float(null[core]), 'status': 'PASS'},
        'all_covariate_information_retained': True, 'sample_rows_changed': False}


def delta_equivalence(original, frame, fi_t_mean, fi_t_sd, lm_mean, lm_sd):
    delta = frame.fi_landmark.to_numpy(float)-frame.fi_t.to_numpy(float)
    # 常数来自每人一次，不让后续访视次数改变中心和尺度。
    person = frame.drop_duplicates('person_id')
    d = person.fi_landmark.to_numpy(float)-person.fi_t.to_numpy(float)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    require(sd > 0 and fi_t_sd > 0 and lm_sd > 0, '变化量或FI尺度退化')
    xd = original.iloc[:, :-2].copy()
    xd['delta_fi_z'] = (delta-md)/sd
    xd['delta_fi_z:time_years'] = xd.delta_fi_z*frame.time_years
    names = list(original); mapping = np.eye(len(names))
    a, b, c = fi_t_sd/lm_sd, sd/lm_sd, (fi_t_mean+md-lm_mean)/lm_sd
    mapping[:, -2] = 0; mapping[names.index('Intercept'), -2] = c
    mapping[names.index('fi_t_z'), -2] = a; mapping[-2, -2] = b
    mapping[:, -1] = 0; mapping[names.index('time_years'), -1] = c
    mapping[names.index('fi_t_z:time_years'), -1] = a; mapping[-1, -1] = b
    predicted = xd.to_numpy(float) @ mapping
    np.testing.assert_allclose(predicted, original.to_numpy(float), rtol=1e-10, atol=1e-12)
    inverse = np.linalg.solve(mapping, np.eye(len(names)))
    np.testing.assert_allclose(original.to_numpy(float) @ inverse, xd.to_numpy(float), rtol=1e-10, atol=1e-12)
    return {'status': 'PASS_SAME_COLUMN_SPACE', 'original_columns': names, 'delta_columns': list(xd),
            'mapping_delta_to_landmark': mapping.tolist(), 'delta_unique_person_mean': md,
            'delta_unique_person_sample_sd': sd, 'forward_max_abs': float(np.max(abs(predicted-original.to_numpy(float)))),
            'reverse_max_abs': float(np.max(abs(original.to_numpy(float) @ inverse-xd.to_numpy(float)))),
            'math_identity': 'a*F+b*L=(a+b)*F+b*(L-F); likewise for time interactions',
            'additional_real_delta_model': False}


def synthetic_equivalence():
    rng = np.random.default_rng(20261014)
    counts = [1, 2, 3, 1, 4, 2, 1, 3]
    ids = np.repeat(np.arange(len(counts)), counts)
    f = rng.normal(.2, .1, len(counts)); l = f+rng.normal(.02, .08, len(counts)); d = l-f
    t = np.concatenate([np.sort(rng.uniform(.3, 9, n)) for n in counts])
    x = np.column_stack([np.ones(len(ids)), t, f[ids], f[ids]*t, l[ids], l[ids]*t])
    z = np.column_stack([np.ones(len(ids)), t, f[ids], f[ids]*t, d[ids], d[ids]*t])
    beta = rng.normal(0, .1, 6); gamma = beta.copy(); gamma[2] += beta[4]; gamma[3] += beta[5]
    v = .01*(ids[:, None] == ids[None, :])+.04*np.eye(len(ids))
    factor = np.linalg.cholesky(v)
    y = x@beta+factor@rng.normal(size=len(ids))
    def ll(pred):
        r = np.linalg.solve(factor, y-pred)
        return float(-.5*(len(ids)*np.log(2*np.pi)+2*np.log(np.diag(factor)).sum()+r@r))
    np.testing.assert_allclose(x@beta, z@gamma, rtol=1e-10, atol=1e-12)
    error = abs(ll(x@beta)-ll(z@gamma)); require(error <= 1e-8, '合成等价似然检查失败')
    return {'status': 'PASS', 'seed': 20261014, 'counts': counts, 'irregular_times': True,
            'prediction_max_abs': float(np.max(abs(x@beta-z@gamma))), 'loglik_abs_difference': error,
            'marginal_covariance_difference': 0., 'model_fit_calls': 0}
