"""Stage2结果只读整理与P值补充；不拟合、不改锁定九文件或原结果。"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from lf_common import (ROOT, CODE, PROGRESS, RUNS, CONTRACT, COHORTS, SPECS, TERM, UNITS,
    read_json, write_json, write_csv, sha, rel, require, provenance, assert_hashes)
from lf_qa import core_estimate, validate_saved
from run_landmark_fi import verify_lock, safe_id


def summarize(run_id):
    safe_id(run_id)
    contract = verify_lock()
    run = RUNS/run_id
    progress = PROGRESS/'Stage2'/run_id
    summary = run/'Stage2 Summary'
    require(not summary.exists(), '拒绝覆盖执行汇总')
    original_manifest = read_json(run/'run_manifest.json')
    require(original_manifest['status'] == 'STAGE2_COMPLETE_AWAITING_STAGE3', '六单元执行未完整结束')
    assert_hashes(original_manifest['outputs'])
    require([f"{u['cohort']}/{u['spec_id']}" for u in original_manifest['units']] == UNITS, '执行范围不符')
    require(sha(CONTRACT) == '57fabb54d896cd0ea6ca693f0e5d8396cce58eaead3a095bdd45491691e46c5d', '合同变化')
    approval = read_json(progress/'execution_authorization.json')
    require(sha(progress/'execution_authorization.json') == sha(run/'execution_authorization.json'), '授权副本变化')
    require(approval['stage3_5_authorization'] == 'NOT_GRANTED', '阶段授权边界不符')
    summary.mkdir(exist_ok=False)
    rows, diagnostics = [], []
    references = pd.read_csv(run/'cohort_estimates.csv',encoding='utf-8',float_precision='round_trip')
    primary_folder = ROOT/'03_output/3.2 Wealth Recovery and Subsequent Frailty Progression/Primary Models'
    primary_diag = read_json(primary_folder/'primary_lmm_diagnostics.json')
    primary_lineage = {rel(p):sha(p) for p in [primary_folder/'primary_lmm_result_object.json',
        primary_folder/'primary_lmm_fixed_effects.csv',primary_folder/'primary_lmm_fixed_effect_covariance.csv',
        ROOT/'03_output/3.2 Wealth Recovery and Subsequent Frailty Progression/section_3_2_freeze.json']}
    comparisons = []
    fit_calls = 0
    for cohort in COHORTS:
        primary = references.loc[references.cohort.eq(cohort)&references.spec_id.eq('Primary')].iloc[0]
        primary_row = {'cohort':cohort,'specification':'Primary','participants':int(primary.participants),
            'fi_observations':int(primary.fi_observations),'estimate':float(primary.estimate),'se':float(primary.se),
            'ci_lower':float(primary.ci_lower),'ci_upper':float(primary.ci_upper),
            'p_value':float(2*norm.sf(abs(primary.estimate/primary.se))),
            'convergence':'FROZEN_REFERENCE_NOT_REFITTED','bfgs_converged':None,'lbfgs_converged':None,
            'basic_qa_status':'FROZEN_REFERENCE','core_inference_validated':True,'release_eligible':False,
            'stage3_complete':False,'units':'FI units/year','source_kind':'FROZEN_PRIMARY_REFERENCE'}
        rows.append(primary_row)
        anchor = contract['cohort_anchors'][cohort]
        estimates = {}
        for spec in SPECS:
            folder = run/spec/cohort
            validation = validate_saved(folder)
            o = read_json(folder/'result_object.json'); d = read_json(folder/'diagnostics.json')
            logs = read_json(folder/'optimization_log.json')
            require(len(logs) == 2 and [x['optimizer'] for x in logs] == ['bfgs','lbfgs'], '不允许隐含重试或额外模型')
            fit_calls += len(logs)
            require(o['sample'] == anchor, '样本/有序行集锚点变化')
            for log in logs:
                calls = log['effective_low_level_calls']
                require(len(calls) == 1 and log['maxiter'] == 2000, '低层调用数量或上限变化')
                effective = calls[0]['effective_kwargs']
                require(effective.get('gtol') == 1e-8 if log['optimizer']=='bfgs' else
                    effective.get('pgtol') == 1e-10 and effective.get('factr') == 100., '实际精度设置不符')
            raw = None
            if (folder/'model_parameters.npz').is_file():
                with np.load(folder/'model_parameters.npz',allow_pickle=False) as parameters:
                    raw = core_estimate(parameters['fixed_effects'],parameters['fixed_effect_covariance'],parameters['fixed_effect_names'].tolist())
            accepted = o['status'] == 'PASS'
            candidate = o['core_estimate'] if o['status'] != 'HOLD_NOT_ESTIMABLE' else None
            p_value = float(2*norm.sf(abs(candidate['estimate']/candidate['se']))) if candidate else None
            p_raw = float(2*norm.sf(abs(raw['estimate']/raw['se']))) if raw else None
            estimates[spec] = {'raw':raw,'accepted':accepted,'status':o['status']}
            row = {'cohort':cohort,'specification':spec,'participants':anchor['participants'],
                'recovery_participants':anchor['recovery'],'reference_participants':anchor['reference'],
                'fi_observations':anchor['rows'],'ordered_person_wave_sha256':anchor['ordered_person_wave_sha256'],
                **{key:candidate[key] if candidate else None for key in ['estimate','se','ci_lower','ci_upper']},
                'p_value':p_value,'convergence':'BOTH_CONVERGED' if all(v.get('converged',False) for v in logs) else 'NOT_ALL_CONVERGED',
                'bfgs_converged':logs[0].get('converged',False),'lbfgs_converged':logs[1].get('converged',False),
                'basic_qa_status':o['status'],'core_inference_validated':accepted,'release_eligible':False,'stage3_complete':False,
                'random_intercept_variance':o.get('random_intercept_variance'),'residual_variance':o.get('residual_variance'),
                'scaled_gradient_max':d.get('scaled_gradient_max'),'balanced_information_cholesky':d.get('balanced_information_cholesky'),
                'failure_reasons':' | '.join(o.get('failure_reasons',[])),
                'raw_candidate_estimate':raw['estimate'] if raw else None,'raw_candidate_se':raw['se'] if raw else None,
                'raw_candidate_ci_lower':raw['ci_lower'] if raw else None,'raw_candidate_ci_upper':raw['ci_upper'] if raw else None,
                'raw_candidate_p_value':p_raw,'raw_candidate_inference_role':'AUDIT_ONLY_NONPASS' if not accepted else 'MATCHES_PASS_RESULT',
                'units':'FI units/year','source_kind':'NEW_LMM_FIT','result_folder':rel(folder)}
            rows.append(row)
            write_json(folder/'stage2_core_inference.json',{'schema':'round1c_stage2_core_inference_v1',
                'cohort':cohort,'spec_id':spec,'status':o['status'],'validated_core':candidate,'p_value':p_value,
                'raw_candidate':raw,'raw_candidate_p_value':p_raw,'raw_candidate_role':row['raw_candidate_inference_role'],
                'P_method':'two_sided_normal_Wald_from_saved_beta_and_model_based_SE',
                'original_result_object_sha256':sha(folder/'result_object.json'),
                'parameter_file_sha256':sha(folder/'model_parameters.npz'),'reporting_code_sha256':sha(__file__),
                'model_fit_calls':0,'scientific_approval':'PENDING','release_eligible':False})
            diagnostics.append({'cohort':cohort,'specification':spec,'status':o['status'],
                'diagnostics':d,'optimization_log':logs,'file_integrity_validation':validation,
                'design_parameterization_sha256':sha(folder/'design_parameterization.json'),
                'sample_anchor_sha256':sha(folder/'sample_anchor.json'),
                'original_result_object_sha256':sha(folder/'result_object.json'),
                'reporting_code_fit_calls':0})
        po,lm = estimates['PO01'],estimates['LM01']
        comparisons.append({'cohort':cohort,
            'Primary_to_PO01_difference':po['raw']['estimate']-primary.estimate if po['raw'] else None,
            'PO01_SE_ratio_to_Primary':po['raw']['se']/primary.se if po['raw'] else None,
            'PO01_to_LM01_difference':lm['raw']['estimate']-po['raw']['estimate'] if lm['raw'] and po['raw'] else None,
            'LM01_SE_ratio_to_Primary':lm['raw']['se']/primary.se if lm['raw'] else None,
            'LM01_SE_ratio_to_PO01':lm['raw']['se']/po['raw']['se'] if lm['raw'] and po['raw'] else None,
            'comparison_role':'VALIDATED_BASIC_NUMERICS_DESCRIPTIVE_ONLY' if po['accepted'] and lm['accepted'] else 'RAW_CANDIDATE_AUDIT_ONLY_COMPARISON_PENDING',
            'same_window_difference_significance_test':False,'causal_or_mediation_interpretation':False})
    require(fit_calls == 12,'优化调用范围变化')
    write_csv(summary/'cohort_estimates.csv',rows)
    write_csv(summary/'sequential_comparisons.csv',comparisons)
    write_json(summary/'model_diagnostics.json',{'stage':2,'models':diagnostics,
        'primary_reference_diagnostics':primary_diag,'primary_source_hashes':primary_lineage,
        'independent_stage3_checks_executed':False,'core_inference_pvalues_added_without_refit':True})
    for name in ['cohort_estimates.csv','sequential_comparisons.csv','model_diagnostics.json']:
        target=progress/name
        require(not target.exists(),'过程汇总拒绝覆盖')
        target.write_bytes((summary/name).read_bytes())
    counts = {state:sum(u['status']==state for u in original_manifest['units'])
              for state in ['PASS','BOUNDARY_REVIEW','HOLD_NOT_ESTIMABLE']}
    write_json(summary/'readback_validation.json',{'status':'PASS_STAGE2_OUTPUT_READBACK','models':6,'model_fit_calls':12,
        'scientific_model_identities':6,'unit_attempts':6,'automatic_retries':0,'status_counts':counts,
        'original_runner_outputs_unchanged':True,'contract_sha256':sha(CONTRACT),'locked_code_files_unchanged':9,
        'PO01_LM01_sample_and_key_hashes_equal':True,'P_values':'derived_from_saved_normal_Wald_estimates',
        'HOLD_validated_columns':'EMPTY; raw candidate columns only for audit','primary_refitted':False,
        'stage3_5_executed':False,'new_meta_runs':0})
    # 完整执行manifest在报告写完后生成，避免自引用及漏记最终报告。
    print(json.dumps({'status':'STAGE2_REPORT_DATA_READY','counts':counts,'optimizer_fit_calls':fit_calls,
                      'summary':rel(summary),'original_run_manifest_sha256':sha(run/'run_manifest.json')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='只读取已有Stage2结果，不拟合、不冻结')
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args();summarize(args.run_id)
