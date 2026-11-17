"""Stage2基础数值分类；不提前实现独立Stage3科学QA。"""
import numpy as np
from lf_common import TERM, Z95, require
from lf_spec import POLICY
from basic_qa import extract_numerical


def core_estimate(beta, covariance, names):
    i = names.index(TERM)
    value, variance = float(beta[i]), float(covariance[i, i])
    require(np.isfinite([value, variance]).all() and variance > 0, '核心估计或方差不可用，禁止裁剪')
    se = float(np.sqrt(variance))
    return {'term': TERM, 'estimate': value, 'se': se,
            'ci_lower': value-Z95*se, 'ci_upper': value+Z95*se,
            'units': 'FI units/year', 'ci_method': 'model_based_normal_Wald_95'}


def optimizer_checks(first, final, names):
    a = core_estimate(np.asarray(first.fe_params), np.asarray(first.cov_params().loc[names,names]), names)
    b = core_estimate(np.asarray(final.fe_params), np.asarray(final.cov_params().loc[names,names]), names)
    return {'loglik_absolute_difference': abs(float(first.llf-final.llf)),
            'core_beta_absolute_difference': abs(a['estimate']-b['estimate']),
            'core_se_relative_difference': abs(a['se']-b['se'])/a['se'],
            'bfgs_converged': bool(first.converged), 'lbfgs_converged': bool(final.converged)}


def classify(d, checks):
    reasons, caveats = [], []
    if not d['converged'] or not checks['bfgs_converged'] or not checks['lbfgs_converged']:
        reasons.append('优化器未收敛')
    if not np.isfinite(d['log_likelihood']):
        reasons.append('似然非有限')
    # Primary结局窗口不同，不能要求post-only似然高于Primary。
    sigma = d['residual_variance']
    if not np.isfinite(sigma) or sigma <= 0:
        reasons.append('残差方差非法')
    g = d['random_spectrum']
    if not g['symmetric'] or g['eigenvalues'] is None or min(g['eigenvalues']) < 0:
        reasons.append('随机截距方差非法；不裁剪')
    fixed = d['fixed_covariance_spectrum']
    if not fixed['symmetric'] or fixed['eigenvalues'] is None or min(fixed['eigenvalues']) <= 0:
        reasons.append('固定效应协方差不支持CI')
    if not d.get('score_finite', False):
        reasons.append('梯度非有限')
    for field, threshold, exclusive in [
        ('loglik_absolute_difference', 1e-4, True),
        ('core_beta_absolute_difference', 1e-6, True),
        ('core_se_relative_difference', 1e-3, False)]:
        value = checks[field]
        if not np.isfinite(value) or (value >= threshold if exclusive else value > threshold):
            reasons.append('优化器一致性失败：'+field)
    if d['boundary_candidate']:
        caveats.append('合法边界或弱识别候选，须Stage3审阅；不自动改变结构')
        if d.get('boundary_kkt', {}).get('status') != 'PASS_SCREEN':
            caveats.append('可行方向/KKT尚未充分支持')
        if not d['balanced_information_cholesky']:
            caveats.append('完整信息非内点正定，需区分方差识别及核心推断')
    else:
        gradient = d['scaled_gradient_max']
        if not d['balanced_information_cholesky'] or gradient is None or not np.isfinite(gradient) or gradient > 1e-3:
            reasons.append('内点信息或标准化梯度失败')
    return {'status': 'HOLD_NOT_ESTIMABLE' if reasons else ('BOUNDARY_REVIEW' if d['boundary_candidate'] else 'PASS'),
            'failure_reasons': reasons, 'caveats': caveats, 'stage3_complete': False,
            'release_eligible': False, 'author_scientific_review': 'PENDING'}


def validate_saved(folder):
    from lf_common import read_json, sha
    from lf_spec import interpretation, Q1, Q1_TIME, Q1_COMBINED
    import pandas as pd
    o = read_json(folder/'result_object.json')
    require(o['status'] in ('PASS','HOLD_NOT_ESTIMABLE','BOUNDARY_REVIEW'), '未知Stage2状态')
    require(not o['release_eligible'] and not o['stage3_complete'], 'Stage2不能宣称发布完成')
    require(o['interpretation'] == interpretation(o['spec_id']), '估计对象或外推限制被改写')
    require((folder/'optimization_log.json').exists() and (folder/'diagnostics.json').exists(), '缺少运行证据')
    if o['status'] != 'HOLD_NOT_ESTIMABLE':
        with np.load(folder/'model_parameters.npz', allow_pickle=False) as p:
            require(p['fixed_effect_names'].tolist() == o['fixed_effect_names'], '参数标签不一致')
            np.testing.assert_array_equal(p['fixed_effects'], o['fixed_effects'])
            np.testing.assert_array_equal(p['fixed_effect_covariance'], o['fixed_effect_covariance'])
            expected = core_estimate(p['fixed_effects'],p['fixed_effect_covariance'],o['fixed_effect_names'])
        for key in ('estimate','se','ci_lower','ci_upper'):
            np.testing.assert_allclose(expected[key],o['core_estimate'][key],rtol=1e-12,atol=1e-14)
        mapping = read_json(folder/'design_parameterization.json')
        require(mapping['basis_columns'] == o['fixed_effect_names'], '执行基底身份变化')
        table = pd.read_csv(folder/'fixed_effects.csv',encoding='utf-8',float_precision='round_trip')
        require(table.term.tolist() == o['fixed_effect_names'], 'CSV系数标签变化')
        np.testing.assert_array_equal(table.estimate.to_numpy(float),np.asarray(o['fixed_effects']))
        require(not table.publication_eligible.any(), 'Stage2参数不能提前发布')
        if o['cohort'] == 'KLoSA':
            require(Q1 not in table.term.values and Q1_TIME not in table.term.values and Q1_COMBINED in table.term.values,
                    '不得报告独立Q1系数')
    else:
        require(bool(o.get('failure_reasons')), 'HOLD必须保存原因')
    return {'status':'PASS_FILE_INTEGRITY','numerical_status':o['status'],
            'outputs':{p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
