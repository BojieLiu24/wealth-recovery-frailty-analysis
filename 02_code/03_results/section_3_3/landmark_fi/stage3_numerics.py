"""Round1C独立数值QA；仅评价保存点，不调用fit或优化器。"""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import norm
from lf_common import TERM, Z95, require
from lf_spec import Q1, Q1_TIME, Q1_COMBINED
from exception_checks import independent_derivatives, inverse_information

sys.path.insert(0, str(Path(__file__).parent.parent/'model_robustness/stage3'))
from independent_checks import block_checks


def validate_anchor(actual, expected):
    require(actual == expected, '样本人数、行数或有序键漂移')


def validate_basis(original, basis, names, mapping, cohort, time):
    """独立核对保存映射，不调用Stage2参数化函数。"""
    original = np.asarray(original, float); basis = np.asarray(basis, float)
    A = np.asarray(mapping['A'], float)
    require(mapping['basis_columns'] == names, '执行基底列名不一致')
    p = original.shape[1]; rank = np.linalg.matrix_rank(original)
    require(np.linalg.matrix_rank(basis) == basis.shape[1], '执行基底降秩')
    np.testing.assert_allclose(basis@A, original, rtol=1e-10, atol=1e-12)
    target = np.zeros(p); target[mapping['original_columns'].index(TERM)] = 1
    contrast = np.zeros(len(names)); contrast[names.index(TERM)] = 1
    np.testing.assert_array_equal(contrast@A, target)
    _, singular, vh = np.linalg.svd(original, full_matrices=False)
    null_error = 0.
    if cohort == 'KLoSA':
        require(rank == p-1 and basis.shape[1] == p-1, 'KLoSA额外秩亏')
        columns = mapping['original_columns']; j,k = columns.index(Q1),columns.index(Q1_TIME)
        active = original[:,j] != 0
        require(active.sum() == 3 and np.array_equal(np.asarray(time)[active], [23/12]*3), 'Q1支持变化')
        np.testing.assert_array_equal(original[:,k], (23/12)*original[:,j])
        null = np.zeros(p); null[j] = -23/12; null[k] = 1
        np.testing.assert_allclose(A@null, 0, rtol=0, atol=1e-12)
        null_error = abs(float(vh[-1]@target))
        require(null_error <= 1e-10 and Q1_COMBINED in names and Q1 not in names and Q1_TIME not in names,
                '恢复×时间不可估计或生成了独立Q1系数')
    else:
        require(rank == p and basis.shape[1] == p, '原42/44列非满秩')
        np.testing.assert_array_equal(A, np.eye(p))
    return {'status':'PASS','original_rank':int(rank),'basis_rank':int(np.linalg.matrix_rank(basis)),
            'original_columns':mapping['original_columns'],'basis_columns':names,'A':A.tolist(),
            'factorization_max_abs':float(np.max(abs(basis@A-original))),
            'original_singular_values':singular.tolist(),'core_null_projection_abs':null_error,
            'independent_Q1_coefficients_reported':False}


def wald(beta, covariance, names):
    j = names.index(TERM); b = float(beta[j]); v = float(covariance[j,j])
    require(np.isfinite([b,v]).all() and v>0, '核心SE非法，不裁剪')
    se = float(np.sqrt(v))
    return {'estimate':b,'se':se,'ci_lower':b-Z95*se,'ci_upper':b+Z95*se,
            'p_value':float(2*norm.sf(abs(b/se))),'units':'FI units/year'}


def technical_status(numerical_ok, original_status, converged, exception=None, boundary=False):
    """作者数值例外只进入限定推荐；任何新失败仍HOLD。"""
    if not numerical_ok:
        return 'HOLD'
    if original_status != 'PASS' or not all(converged):
        valid = exception and exception.get('disposition')=='AUTHOR_ACCEPTED_NUMERICAL_EXCEPTION' and \
                exception.get('cohort')=='HRS' and exception.get('specification')=='PO01' and \
                exception.get('accepted_for_stage3') is True
        return 'QA_PASS_WITH_CAVEAT_RECOMMENDED' if valid else 'HOLD'
    return 'QA_PASS_WITH_CAVEAT_RECOMMENDED' if boundary else 'PASS'


def validate_candidate(y, x, ids, names, saved, diagnostic, logs, original_status, exception=None):
    beta=saved['fixed_effects']; cov=saved['fixed_effect_covariance']; p=len(beta)
    sigma=float(saved['residual_variance']); g=saved['random_covariance_scaled']
    q=float(saved['random_covariance_unscaled'][0,0]); j=names.index(TERM)
    require(g.shape==(1,1) and np.isfinite(g).all() and g[0,0]>=0 and sigma>0, '方差非法')
    require(np.isfinite(beta).all() and np.isfinite(cov).all(), '参数或协方差非有限')
    np.testing.assert_allclose(cov,cov.T,rtol=1e-10,atol=1e-12)
    require(np.linalg.eigvalsh(cov).min()>0,'固定协方差非法')
    np.testing.assert_allclose(g[0,0],q*sigma,rtol=1e-12,atol=1e-14)
    d=independent_derivatives(y,x,ids,beta,q,sigma)
    profile,pi=inverse_information(d['hessian_profile'])
    full,fi=inverse_information(d['hessian_full'])
    block=block_checks(y,x,ids,np.ones((len(y),1)),g,sigma,beta)
    np.testing.assert_allclose(d['score_profile'],diagnostic['score_direct_unscaled_covariance'],rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(d['hessian_profile'],saved['hessian'],rtol=1e-9,atol=1e-6)
    np.testing.assert_allclose(profile[:p,:p],cov,rtol=1e-8,atol=1e-12)
    np.testing.assert_allclose(full[:p,:p],cov,rtol=1e-8,atol=1e-12)
    require(abs(d['loglik_profile']-diagnostic['log_likelihood'])<=1e-8 and
            abs(block['loglik']-diagnostic['log_likelihood'])<=1e-8, '独立Gaussian似然不一致')
    require(abs(d['sigma_profile']-sigma)<=1e-10, '残差方差剖面复核失败')
    require(block['saved_qr_beta_max_abs']<=1e-10 and block['qr_svd_beta_max_abs']<=1e-10, 'GLS点估计不一致')
    core=wald(beta,cov,names); pc=wald(beta,profile[:p,:p],names); fc=wald(beta,full[:p,:p],names)
    require(abs(pc['se']-core['se'])/core['se']<=1e-8 and abs(fc['se']-core['se'])/core['se']<=1e-8,'完整信息SE复核失败')
    scaled=float(np.max(abs(d['score_profile'])/np.sqrt(np.diag(-d['hessian_profile']))))
    boundary=q<=1e-6
    if boundary:
        kkt={'feasible_direction_score_q':float(d['score_profile'][-1]),
             'status':'PASS_SCREEN' if d['score_profile'][-1]<=1e-3*np.sqrt(-d['hessian_profile'][-1,-1]) else 'HOLD'}
        require(kkt['status']=='PASS_SCREEN','边界可行方向不支持')
    else:
        require(scaled<=1e-3,'内点标准化梯度超过锁定门槛')
        kkt=None
    require(len(logs)==2 and [r['optimizer'] for r in logs]==['bfgs','lbfgs'],'优化器谱系不符')
    first=logs[0]; final=logs[1]
    for row in logs:
        require(row['maxiter']==2000,'优化次数规格变化')
        expected={'gtol':1e-8} if row['optimizer']=='bfgs' else {'pgtol':1e-10,'factr':100.}
        require(len(row['effective_low_level_calls'])==1 and row['effective_low_level_calls'][0]['effective_kwargs']==expected,'优化器有效设置变化')
    np.testing.assert_array_equal(final['parameters'],saved['parameters'])
    checks=diagnostic['optimizer_checks']
    first_params=np.asarray(first['parameters'],float)
    first_derivatives=independent_derivatives(y,x,ids,first_params[:-1],first_params[-1],sigma)
    first_cov,_=inverse_information(first_derivatives['hessian_profile'])
    first_core=wald(first_params[:-1],first_cov[:p,:p],names)
    independent_optimizer_checks={'loglik_absolute_difference':abs(first_derivatives['loglik_profile']-d['loglik_profile']),
        'core_beta_absolute_difference':abs(first_core['estimate']-core['estimate']),
        'core_se_relative_difference':abs(first_core['se']-core['se'])/first_core['se']}
    require(independent_optimizer_checks['loglik_absolute_difference']<1e-4 and
            independent_optimizer_checks['core_beta_absolute_difference']<1e-6 and
            independent_optimizer_checks['core_se_relative_difference']<=1e-3,'独立优化一致性失败')
    require(checks['loglik_absolute_difference']<1e-4 and checks['core_beta_absolute_difference']<1e-6 and
            checks['core_se_relative_difference']<=1e-3,'锁定优化一致性未通过')
    if exception:
        require(first['converged'] and not final['converged'],'HRS原异常收敛身份不符')
        np.testing.assert_array_equal(first['parameters'],final['parameters'])
    status=technical_status(True,original_status,[first['converged'],final['converged']],exception,boundary)
    require(status!='HOLD','未收敛候选无有效作者例外')
    result={'numerical_status':status,'core':core,'full_physical_information_core':fc,
       'loglik_saved':diagnostic['log_likelihood'],'loglik_profile':d['loglik_profile'],'loglik_Cholesky':block['loglik'],
       'profile_loglik_abs_error':abs(d['loglik_profile']-diagnostic['log_likelihood']),
       'Cholesky_loglik_abs_error':abs(block['loglik']-diagnostic['log_likelihood']),
       'score_max_abs_error':float(np.max(abs(d['score_profile']-diagnostic['score_direct_unscaled_covariance']))),
       'hessian_max_abs_error':float(np.max(abs(d['hessian_profile']-saved['hessian']))),
       'fixed_covariance_max_abs_error':float(np.max(abs(profile[:p,:p]-cov))),
       'full_physical_covariance_max_abs_error':float(np.max(abs(full[:p,:p]-cov))),
       'core_beta_QR_abs_error':float(abs(block['qr_beta'][j]-beta[j])),
       'core_SE_full_physical_relative_error':abs(fc['se']-core['se'])/core['se'],
       'scaled_gradient_max':scaled,'profile_information':pi,'full_physical_information':fi,
       'fixed_covariance_eigenvalues':np.linalg.eigvalsh(cov).tolist(),
       'hessian_eigenvalues':np.linalg.eigvalsh(d['hessian_profile']).tolist(),
       'random_variance':float(g[0,0]),'residual_variance':sigma,'unscaled_variance_ratio':q,
       'marginal_covariance_min_eigenvalue':sigma,'marginal_covariance_max_eigenvalue':float(sigma+g[0,0]*max(np.unique(ids,return_counts=True)[1])),
       'boundary_candidate':boundary,'boundary_kkt':kkt,'optimizer_checks':checks,
       'independently_recalculated_optimizer_checks':independent_optimizer_checks,
       'optimizer_variance_stability':{'BFGS_residual_variance':first_derivatives['sigma_profile'],
          'final_residual_variance':sigma,'residual_variance_absolute_difference':abs(first_derivatives['sigma_profile']-sigma),
          'BFGS_random_intercept_variance':float(first_params[-1]*first_derivatives['sigma_profile']),
          'final_random_intercept_variance':float(g[0,0]),
          'random_variance_absolute_difference':abs(float(first_params[-1]*first_derivatives['sigma_profile']-g[0,0])),
          'not_external_sampling_stability_evidence':True},
       'whitened_design_condition':block['whitened_design_condition'],'qr_svd_beta_max_abs':block['qr_svd_beta_max_abs'],
       'optimizer_warnings':[w for r in logs for w in r['warnings']],
       'pseudoinverse_or_clipping_used':False,'conditional_GLS_covariance_used_for_final_inference':False,
       'original_status':original_status,'ordinary_double_convergence_requirement_met':all(r['converged'] for r in logs)}
    arrays={'fixed_effect_names':np.array(names),'profile_score':d['score_profile'],'profile_hessian':d['hessian_profile'],
            'profile_parameter_covariance':profile,'physical_score':d['score_full'],'physical_hessian':d['hessian_full'],
            'physical_parameter_covariance':full,'independent_QR_beta':block['qr_beta'],'independent_SVD_beta':block['svd_beta']}
    return result,arrays,block
