"""只读复核HRS/PO01原候选解；保存新异常审查记录，不重试优化或改HOLD。"""
from pathlib import Path
from contextlib import contextmanager
import ast
import inspect
import sys
import numpy as np
import scipy.optimize as optimization
import statsmodels.api as sm
import statsmodels.base.optimizer as backend
import scipy.optimize._lbfgsb_py as scipy_lbfgs
from statsmodels.regression.mixed_linear_model import MixedLMParams
from lf_common import ROOT,RUNS,PROGRESS,TERM,CONTRACT,read_json,write_json,sha,rel,require,assert_hashes,provenance
from run_landmark_fi import verify_lock,same_parameterization
from lf_inputs import prepare_cohort
from exception_checks import independent_derivatives,inverse_information

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'model_robustness/stage3'))
from independent_checks import block_checks


@contextmanager
def prohibit_optimization(counter):
    original_fit=sm.MixedLM.fit
    names=['minimize','fmin_bfgs','fmin_l_bfgs_b','fmin','least_squares','root','minimize_scalar','brent','fminbound']
    originals={name:getattr(optimization,name) for name in names}
    def blocked(*a,**k):
        counter['forbidden_calls']+=1
        raise RuntimeError('EXCEPTION_CHECK_FORBIDS_OPTIMIZATION')
    sm.MixedLM.fit=blocked
    for name in names:setattr(optimization,name,blocked)
    try:
        yield
    finally:
        sm.MixedLM.fit=original_fit
        for name,value in originals.items():setattr(optimization,name,value)


def main():
    contract=verify_lock()
    run=RUNS/'round1c_stage2_20261014_01';folder=run/'PO01/HRS'
    original_manifest=read_json(run/'execution_manifest.json')
    assert_hashes(original_manifest['outputs']);assert_hashes(original_manifest['process_files'])
    before={rel(p):sha(p) for p in run.rglob('*') if p.is_file()}
    logs=read_json(folder/'optimization_log.json');diag=read_json(folder/'diagnostics.json');obj=read_json(folder/'result_object.json')
    require(obj['status']=='HOLD_NOT_ESTIMABLE' and logs[0]['converged'] and not logs[1]['converged'], '原异常身份不符')
    require(len(logs)==2 and [v['optimizer'] for v in logs]==['bfgs','lbfgs'], '原尝试谱系异常')
    counter={'forbidden_calls':0}
    with prohibit_optimization(counter):
        post,models,report,*_=prepare_cohort('HRS')
        same_parameterization(models['PO01']['mapping'],read_json(folder/'design_parameterization.json'))
        require(report['anchor']==obj['sample']==contract['cohort_anchors']['HRS'], '样本或有序行集变化')
        x=models['PO01']['x'].to_numpy(float);names=list(models['PO01']['x'])
        y=post.fi_primary.to_numpy(float);ids=post.person_id.to_numpy()
        with np.load(folder/'model_parameters.npz',allow_pickle=False) as p:
            beta=p['fixed_effects'].copy();cov=p['fixed_effect_covariance'].copy();oldH=p['hessian'].copy()
            sigma=float(p['residual_variance']);g=p['random_covariance_scaled'].copy()
            q=float(p['random_covariance_unscaled'][0,0]);saved_params=p['parameters'].copy()
        require(logs[0]['parameter_names'][:-1]==names, 'BFGS固定参数名称变化')
        np.testing.assert_array_equal(logs[0]['parameters'],logs[1]['parameters'])
        np.testing.assert_array_equal(logs[0]['parameters'],saved_params)
        np.testing.assert_array_equal(np.asarray(logs[0]['parameters'][:-1]),beta)
        require(float(logs[0]['parameters'][-1])==q, 'BFGS方差参数身份错误')
        check=independent_derivatives(y,x,ids,beta,q,sigma)
        profile_cov,profile_info=inverse_information(check['hessian_profile'])
        full_cov,full_info=inverse_information(check['hessian_full'])
        independent=block_checks(y,x,ids,np.ones((len(y),1)),g,sigma,beta)
        # 库API只在固定点评价导数，与独立解析实现交叉核对，不调用fit。
        model=sm.MixedLM(y,models['PO01']['x'],ids,exog_re=np.ones((len(y),1)),use_sqrt=True)
        model.reml=False;model.cov_pen=None;model.fe_pen=None;model._freepat=None;model._cov_sing=0
        params=MixedLMParams.from_components(fe_params=beta,cov_re=np.array([[q]]))
        evaluatedH,singular=model.hessian(params)
        evaluatedScore=np.concatenate(model.score_full(params,calc_fe=True))
        backendLL=float(model.loglike(params,profile_fe=False))
    require(counter['forbidden_calls']==0, '复核越界调用了优化')
    core=names.index(TERM);score=check['score_profile'];info=-check['hessian_profile']
    scaled=float(np.max(abs(score)/np.sqrt(np.diag(info))))
    savedScore=np.asarray(diag['score_direct_unscaled_covariance'])
    np.testing.assert_allclose(score,savedScore,rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(check['hessian_profile'],oldH,rtol=1e-9,atol=1e-6)
    np.testing.assert_allclose(evaluatedH,oldH,rtol=1e-9,atol=1e-6)
    np.testing.assert_allclose(evaluatedScore,savedScore,rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(profile_cov[:len(beta),:len(beta)],cov,rtol=1e-8,atol=1e-12)
    savedSE=float(np.sqrt(cov[core,core]));independentSE=float(np.sqrt(profile_cov[core,core]))
    fullSE=float(np.sqrt(full_cov[core,core]))
    require(abs(check['loglik_profile']-diag['log_likelihood'])<=1e-8 and
            abs(independent['loglik']-diag['log_likelihood'])<=1e-8, '独立似然不一致')
    require(abs(independent['qr_beta'][core]-beta[core])<=1e-10 and
            abs(independentSE-savedSE)/savedSE<=1e-8 and abs(fullSE-savedSE)/savedSE<=1e-8,'核心估计或SE不一致')
    require(scaled<=1e-3 and not singular,'候选解数值检查失败')
    # 仅计算局部二次近似的剩余增益，不应用任何参数步长或接受新解。
    diagnostic_step=profile_cov@score
    gain=float(.5*score@diagnostic_step);ulp=float(np.spacing(diag['log_likelihood']))
    initialGradient=float(logs[1]['optimization_history'][0]['gopt'][0])
    reconstructedGradient=float(-2*np.sqrt(q)*score[-1]/len(y))
    backendSource=inspect.getsource(backend._fit_lbfgs)
    parsed=ast.parse(backendSource)
    returnedKeys=[]
    for node in ast.walk(parsed):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Dict) and any(isinstance(t,ast.Name) and t.id=='retvals' for t in node.targets):
            returnedKeys=[k.value for k in node.value.keys if isinstance(k,ast.Constant)]
    require('task' not in returnedKeys and 'message' not in returnedKeys, '终止信息可能仍可读取，须重新查证')
    hist=logs[1]['optimization_history'][0]
    out=PROGRESS/'Stage2_Exception_Resolution/HRS_PO01/check_v1'
    require(not out.exists(), '拒绝覆盖异常复核记录')
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'independent_numeric_evidence.npz',fixed_effect_names=np.array(names),
        saved_bfgs_beta=beta,independent_gls_beta=independent['qr_beta'],
        independent_profile_score=score,independent_profile_hessian=check['hessian_profile'],
        independent_profile_full_parameter_covariance=profile_cov,
        independent_physical_score=check['score_full'],independent_physical_hessian=check['hessian_full'],
        independent_physical_full_parameter_covariance=full_cov)
    result={'status':'INDEPENDENT_CANDIDATE_CHECK_PASS_AUTHOR_EXCEPTION_DECISION_PENDING',
        'scope':'HRS/PO01 Stage2 exception only; no full Stage3 scientific review',
        'original_status':'HOLD_NOT_ESTIMABLE','new_accepted_status':None,'author_exception_approval':'NOT_GRANTED',
        'original_run':rel(run),'original_result_sha256':sha(folder/'result_object.json'),
        'contract_sha256':sha(CONTRACT),'source_execution_manifest_sha256':sha(run/'execution_manifest.json'),
        'sample':obj['sample'],'BFGS_and_final_parameters_identical':True,
        'termination_forensics':{'warnflag':hist['warnflag'],'iterations':hist['iterations'],'function_calls':hist['fcalls'],
            'original_scipy_task_available':False,'statsmodels_returned_keys':returnedKeys,
            'accurate_original_stop_reason':'UNRECOVERABLE_FROM_RETAINED_LOGS',
            'initial_normalized_Cholesky_gradient_abs':abs(initialGradient),
            'independently_reconstructed_gradient_abs':abs(reconstructedGradient),
            'BFGS_gtol':1e-8,'LBFGS_pgtol':1e-10,'LBFGS_initial_gradient_criterion_met':abs(initialGradient)<=1e-10,
            'line_search_or_rounding_cause':'CONSISTENT_WITH_FLOATING_POINT_STAGNATION_BUT_NOT_CONFIRMED'},
        'independent_numerics':{'loglik_saved':diag['log_likelihood'],'loglik_profile':check['loglik_profile'],
            'loglik_Cholesky':independent['loglik'],'loglik_backend_reevaluation':backendLL,
            'profile_loglik_abs_error':abs(check['loglik_profile']-diag['log_likelihood']),
            'Cholesky_loglik_abs_error':abs(independent['loglik']-diag['log_likelihood']),
            'score_max_abs_error':float(np.max(abs(score-savedScore))),
            'hessian_max_abs_error':float(np.max(abs(check['hessian_profile']-oldH))),
            'hessian_relative_norm_error':float(np.linalg.norm(check['hessian_profile']-oldH)/np.linalg.norm(oldH)),
            'fixed_covariance_max_abs_error':float(np.max(abs(profile_cov[:len(beta),:len(beta)]-cov))),
            'core_beta_saved':float(beta[core]),'core_beta_independent_GLS':float(independent['qr_beta'][core]),
            'core_beta_abs_error':float(abs(independent['qr_beta'][core]-beta[core])),
            'core_SE_saved':savedSE,'core_SE_profile_observed_information':independentSE,
            'core_SE_full_physical_observed_information':fullSE,
            'core_SE_relative_error':abs(independentSE-savedSE)/savedSE,
            'core_SE_full_physical_relative_error':abs(fullSE-savedSE)/savedSE,
            'scaled_gradient_max':scaled,'profile_information':profile_info,'physical_information':full_info,
            'random_variance_positive':True,'residual_variance_positive':True,
            'profile_sigma_abs_error':abs(check['sigma_profile']-sigma),
            'local_quadratic_gain_DIAGNOSTIC_ONLY_NOT_APPLIED':gain,'loglik_float64_ULP':ulp,
            'gain_over_ULP':gain/ulp,'diagnostic_predicted_core_beta_change_NOT_APPLIED':float(diagnostic_step[core]),
            'pseudoinverse_or_clipping_used':False,'conditional_GLS_SE_used_for_inference':False},
        'synthetic_derivative_tests':'2_PASSED_AGAINST_DENSE_CHOLESKY_FINITE_DIFFERENCES',
        'new_model_fits':0,'new_optimizer_calls':0,'optimization_guard':counter,
        'model_spec_sample_thresholds_changed':False,'full_stage3_5_executed':False,
        'recommendation':'AUTHOR_APPROVED_NUMERICAL_EXCEPTION_WITH_CAVEAT; original HOLD retained; science review pending',
        'source_hashes':{rel(Path(p)):sha(p) for p in [__file__,Path(__file__).with_name('exception_checks.py'),
            Path(__file__).with_name('test_exception_checks.py'),inspect.getfile(backend),inspect.getfile(scipy_lbfgs),
            Path(__file__).parent.parent/'model_robustness/stage3/independent_checks.py']},**provenance()}
    write_json(out/'exception_check.json',result)
    assert_hashes(before)
    write_json(out/'manifest.json',{'status':result['status'],'inputs':before,'source_hashes':result['source_hashes'],
        'outputs':{rel(p):sha(p) for p in out.iterdir() if p.is_file()},'original_run_unchanged':True,
        'new_model_fits':0,'new_optimizer_calls':0,'original_status':'HOLD_NOT_ESTIMABLE',**provenance()})
    print({k:result['independent_numerics'][k] for k in ['profile_loglik_abs_error','core_beta_abs_error','core_SE_relative_error','scaled_gradient_max','gain_over_ULP']})
    print(rel(out))


if __name__=='__main__':main()
