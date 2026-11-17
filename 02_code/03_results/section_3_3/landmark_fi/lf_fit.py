"""仅在完整授权验证后执行Stage2；本轮不调用真实拟合。"""
from contextlib import contextmanager
from dataclasses import dataclass
import time
import traceback
import warnings
import numpy as np
import statsmodels.api as sm
import statsmodels.base.optimizer as backend
from statsmodels.regression.mixed_linear_model import MixedLMParams
from lf_common import (CONTRACT, UNITS, TERM, Z95, require, sha, clean, write_json, write_csv)
from lf_spec import POLICY, interpretation, Q1_COMBINED
from lf_qa import extract_numerical, optimizer_checks, core_estimate, classify


@dataclass(frozen=True)
class ExecutionPermit:
    authorization: dict
    contract_sha256: str


@contextmanager
def strict_lbfgs(evidence):
    original = backend._fit_lbfgs
    def wrapped(f, score, start_params, fargs, kwargs, *args, **kw):
        effective = {**kwargs, 'pgtol': POLICY['lbfgs_pgtol'], 'factr': POLICY['lbfgs_factr']}
        evidence.append({'optimizer':'lbfgs','effective_kwargs':clean(effective),
                         'backend':'original_statsmodels_low_level_lbfgs'})
        return original(f, score, start_params, fargs, effective, *args, **kw)
    backend._fit_lbfgs = wrapped
    try:
        yield
    finally:
        backend._fit_lbfgs = original


@contextmanager
def strict_bfgs(evidence):
    original = backend._fit_bfgs
    def wrapped(f, score, start_params, fargs, kwargs, *args, **kw):
        require(kwargs.get('gtol') == POLICY['bfgs_gtol'], 'BFGS底层精度未生效')
        evidence.append({'optimizer':'bfgs','effective_kwargs':clean(kwargs),
                         'backend':'original_statsmodels_low_level_bfgs'})
        return original(f, score, start_params, fargs, kwargs, *args, **kw)
    backend._fit_bfgs = wrapped
    try:
        yield
    finally:
        backend._fit_bfgs = original


def optimize(model, method, start):
    began = time.perf_counter(); result, error, evidence = None, None, []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        try:
            adapter = strict_bfgs if method == 'bfgs' else strict_lbfgs
            with adapter(evidence):
                result = model.fit(start_params=start, reml=False, method=method, full_output=True,
                    maxiter=POLICY['maxiter'], disp=False, **({'gtol':POLICY['bfgs_gtol']} if method == 'bfgs' else {}))
            require(len(evidence) == 1, '底层优化器调用数异常，禁止隐藏重试')
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
            result = None
    log = {'optimizer':method,'maxiter':2000,'effective_low_level_calls':evidence,
           'elapsed_seconds':time.perf_counter()-began,'error':error,
           'warnings':[{'category':w.category.__name__,'message':str(w.message)} for w in records]}
    if result is not None:
        log.update(converged=bool(result.converged),log_likelihood=float(result.llf),
                   parameters=np.asarray(result.params).tolist(),parameter_names=list(result.params.index),
                   optimization_history=getattr(result,'hist',None))
    return result, clean(log)


def fit_unit(post, model_spec, report, primary_diag, cohort, spec, folder, permit):
    require(isinstance(permit, ExecutionPermit) and permit.contract_sha256 == sha(CONTRACT), '缺少已核验执行许可')
    auth = permit.authorization
    require(auth.get('authorization') == 'GRANTED_FOR_SIX_ROUND1C_MODELS' and auth.get('scope') == UNITS,
            '未获全部六单元授权')
    require(f'{cohort}/{spec}' in UNITS, '超出模型范围')
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'design_parameterization.json',model_spec['mapping'])
    write_json(folder/'sample_anchor.json',report['anchor'])
    logs, diagnostics = [], {}
    o = {'cohort':cohort,'spec_id':spec,'source_kind':'NEW_LMM_FIT','sample':report['anchor'],
         'interpretation':interpretation(spec),'stage3_complete':False,'release_eligible':False,
         'author_scientific_review':'PENDING','status':'HOLD_NOT_ESTIMABLE','participant_records_saved':False,
         'core_estimate':None,'automatic_specification_change':False}
    try:
        x = model_spec['x'].astype(float)
        model = sm.MixedLM(post.fi_primary.astype(float), x, post.person_id.to_numpy(),
                           exog_re=np.ones((len(post),1)), use_sqrt=True)
        q = np.array([[primary_diag['random_intercept_variance']/primary_diag['residual_variance']]])
        start = MixedLMParams.from_components(fe_params=model_spec['start_beta'], cov_re=q)
        first, log = optimize(model,'bfgs',start); logs.append(log)
        write_json(folder/'optimization_log.json',logs)
        require(first is not None,'BFGS无结果，不自动替换')
        final, log = optimize(model,'lbfgs',first.params_object.copy()); logs.append(log)
        write_json(folder/'optimization_log.json',logs,replace_owned=True)
        require(final is not None,'L-BFGS无结果，不自动重试')
        names = list(x)
        beta, covariance, diagnostics = extract_numerical(model,final)
        require(np.isfinite(beta).all(),'固定效应非有限')
        # 参数与优化证据先保存；后续分类失败也保留原始数值，不生成替代结果。
        np.savez_compressed(folder/'model_parameters.npz',fixed_effect_names=np.array(names),fixed_effects=beta,
            fixed_effect_covariance=covariance,parameters=np.asarray(final.params),parameter_names=np.array(final.params.index),
            random_covariance_scaled=np.asarray(final.cov_re),random_covariance_unscaled=np.asarray(final.cov_re_unscaled),
            residual_variance=np.asarray(final.scale),hessian=np.asarray(diagnostics['hessian']))
        checks = optimizer_checks(first,final,names)
        state = classify(diagnostics,checks)
        diagnostics.update(optimizer_checks=checks,**state)
        o.update(state,fixed_effect_names=names,fixed_effects=beta.tolist(),fixed_effect_covariance=covariance.tolist(),
                 random_intercept_variance=float(np.asarray(final.cov_re)[0,0]),residual_variance=float(final.scale))
        if state['status'] != 'HOLD_NOT_ESTIMABLE':
            o['core_estimate'] = core_estimate(beta,covariance,names)
        ci_supported = diagnostics['fixed_covariance_spectrum']['symmetric'] and min(diagnostics['fixed_covariance_spectrum']['eigenvalues']) > 0
        fixed = []
        for i,name in enumerate(names):
            se = float(np.sqrt(covariance[i,i])) if ci_supported else None
            role = 'audit_nuisance_parameter'
            if name == TERM: role = 'post_landmark_progression_core'
            if name in ('Intercept','recovery'): role = 'extrapolated_nuisance_not_observed_landmark_level'
            if name == Q1_COMBINED: role = 'combined_Q1_effect_at_supported_time_only'
            fixed.append({'term':name,'estimate':float(beta[i]),'se':se,
                'ci_lower':float(beta[i])-Z95*se if se is not None else None,
                'ci_upper':float(beta[i])+Z95*se if se is not None else None,'presentation_role':role,
                'status':o['status'],'publication_eligible':False})
        write_csv(folder/'fixed_effects.csv',fixed)
        write_csv(folder/'fixed_effect_covariance.csv',[{'term_i':a,'term_j':b,'covariance':float(covariance[i,j])}
                    for i,a in enumerate(names) for j,b in enumerate(names)])
        write_json(folder/'random_effects.json',{'random_design':'[1]','random_intercept_variance':o['random_intercept_variance'],
                    'residual_variance':o['residual_variance'],'status':o['status']})
    except Exception as exc:
        reason = type(exc).__name__+': '+str(exc)
        o.update(status='HOLD_NOT_ESTIMABLE',failure_reasons=[reason],core_estimate=None)
        diagnostics.update(status='HOLD_NOT_ESTIMABLE',failure_reasons=[reason])
        (folder/'failure_trace.txt').write_text(traceback.format_exc(),encoding='utf-8')
    if not (folder/'optimization_log.json').exists():
        write_json(folder/'optimization_log.json',logs)
    write_json(folder/'diagnostics.json',diagnostics)
    write_json(folder/'result_object.json',o)
    return o
