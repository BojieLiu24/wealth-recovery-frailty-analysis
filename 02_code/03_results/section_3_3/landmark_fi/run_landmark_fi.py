"""Round1C Stage0–2分阶段入口；完整授权前禁止真实拟合。"""
import argparse
import json
from pathlib import Path
import re
from contextlib import contextmanager
import statsmodels.api as sm
from lf_common import (ROOT, CODE, PROGRESS, CONTRACT, RUNS, COHORTS, SPECS, UNITS, TERM,
    read_json, write_json, write_csv, sha, rel, require, clean, provenance, code_hashes,
    assert_hashes, check_protected, repo_path)
from lf_spec import scientific_spec, interpretation
from lf_inputs import verify_sources, prepare_cohort
from lf_design import synthetic_equivalence
from lf_qa import core_estimate, validate_saved


@contextmanager
def no_fitting():
    original = sm.MixedLM.fit
    def blocked(*args, **kwargs):
        raise RuntimeError('Stage0–1禁止MixedLM.fit')
    sm.MixedLM.fit = blocked
    try:
        yield
    finally:
        sm.MixedLM.fit = original


def safe_id(value):
    require(isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',value), '身份必须为安全单层名称')
    return value


def preflight_lock(audit_id):
    require(not CONTRACT.exists(), '合同已锁定，禁止覆盖或重跑Stage0–1')
    output = PROGRESS/'Stage01'/safe_id(audit_id)
    require(not output.exists(), '审计目录已存在，禁止覆盖')
    output.mkdir(parents=True,exist_ok=False)
    result = {'status':'HOLD_AUDIT','model_fit_calls':0,'meta_calls':0,'scientific_specification':scientific_spec()}
    try:
        with no_fitting():
            source = verify_sources()
            source['inputs'][rel(PROGRESS/'implementation_before.json')] = sha(PROGRESS/'implementation_before.json')
            cohorts = []
            for cohort in COHORTS:
                post, models, report, *_ = prepare_cohort(cohort)
                cohorts.append(report)
                if cohort == 'KLoSA':
                    for spec in SPECS:
                        write_json(output/'KLoSA'/spec/'design_parameterization.json',models[spec]['mapping'])
            synthetic = synthetic_equivalence()
        result.update(status='VALIDATED',source_audit=source,cohorts=cohorts,
                      synthetic_equivalence=synthetic,**provenance())
    except Exception as exc:
        result['failure'] = type(exc).__name__+': '+str(exc)
        write_json(output/'audit_results.json',result)
        raise
    write_json(output/'audit_results.json',result)
    manifest = {'status':'VALIDATED','schema':'landmark_fi_preflight_v1','inputs':source['inputs'],
                'code':code_hashes(),'outputs':{rel(p):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
                'model_fit_calls':0,**provenance()}
    write_json(output/'manifest.json',manifest)
    contract = {'schema':'landmark_fi_round1c_v1','status':'LOCKED',
        'specification_approval':'AUTHOR_APPROVED_IMPLEMENTATION_PLAN',
        'approval_basis':'作者PLEASE IMPLEMENT修订计划；已明确选定KLoSA等价基底及数值精度策略',
        'real_execution_authorization':'NOT_GRANTED_BY_IMPLEMENTATION_REQUEST',
        'scientific_specification':scientific_spec(),'inputs':manifest['inputs'],'code':manifest['code'],
        'audit':{'path':rel(output),'manifest_sha256':sha(output/'manifest.json'),'results_sha256':sha(output/'audit_results.json')},
        'runtime':provenance()['runtime'],'scope':UNITS,
        'cohort_anchors':{r['cohort']:r['anchor'] for r in cohorts},
        'author_result_approval':'PENDING','stages3_5':'NOT_IMPLEMENTED_NOT_STARTED',**provenance()}
    write_json(CONTRACT,contract)
    return {'status':'VALIDATED_AND_LOCKED','contract':rel(CONTRACT),'audit':rel(output),'fit_calls':0,'real_execution':'NOT_AUTHORIZED'}


def verify_lock():
    contract = read_json(CONTRACT)
    require(contract['status'] == 'LOCKED' and contract['scope'] == UNITS, '合同身份错误')
    require(contract['scientific_specification'] == clean(scientific_spec()), '科学规格漂移')
    require(contract['runtime'] == provenance()['runtime'], '锁定环境发生变化')
    assert_hashes(contract['code']); assert_hashes(contract['inputs'])
    folder = repo_path(contract['audit']['path'])
    require(sha(folder/'manifest.json') == contract['audit']['manifest_sha256'], '审计manifest身份变化')
    require(sha(folder/'audit_results.json') == contract['audit']['results_sha256'], '审计结果身份变化')
    manifest = read_json(folder/'manifest.json'); assert_hashes(manifest['outputs'])
    require(manifest['status'] == 'VALIDATED' and manifest['model_fit_calls'] == 0, 'Preflight未通过或越界')
    return contract


def verify_authorization(contract, path):
    require(path is not None, '真实执行须另行明确授权六模型，当前不拟合')
    a = read_json(path)
    require(a.get('schema') == 'landmark_fi_real_execution_authorization_v1', '授权schema错误')
    require(a.get('authorization') == 'GRANTED_FOR_SIX_ROUND1C_MODELS', '缺少本轮真实执行授权')
    require(a.get('scope') == UNITS, '授权必须准确限定全部六单元')
    require(a.get('contract_sha256') == sha(CONTRACT), '授权未绑定当前合同')
    require(a.get('preflight_manifest_sha256') == contract['audit']['manifest_sha256'], '授权未绑定活动Preflight')
    require(isinstance(a.get('human_instruction'),str) and bool(a['human_instruction'].strip()) and
            isinstance(a.get('evidence_source'),str) and bool(a['evidence_source'].strip()), '缺少作者指令及证据，程序不能代作者批准')
    return a


def same_parameterization(a,b):
    for key in ('original_columns','basis_columns','A','dependency','core_estimability'):
        require(a[key] == b[key], '执行参数化与审计不同：'+key)
    require(a['basis_spectrum']['rank'] == b['basis_spectrum']['rank'], '执行设计秩变化')


def fit(run_id, authorization_path):
    # 授权、身份及全部输入预检均在创建运行目录和任何模型调用之前。
    require(authorization_path is not None, '真实执行未授权；不创建运行目录')
    safe_id(run_id)
    contract = verify_lock()
    authorization = verify_authorization(contract,authorization_path)
    source = verify_sources()
    source['inputs'][rel(PROGRESS/'implementation_before.json')] = sha(PROGRESS/'implementation_before.json')
    require(source['inputs'] == contract['inputs'], '执行来源集合变化')
    reference = read_json(repo_path(contract['audit']['path'])/'audit_results.json')
    cached = {}
    with no_fitting():
        for cohort in COHORTS:
            prepared = prepare_cohort(cohort)
            report = prepared[2]
            require(report['anchor'] == contract['cohort_anchors'][cohort], '后续行集身份变化')
            previous = next(r for r in reference['cohorts'] if r['cohort'] == cohort)
            for spec in SPECS:
                same_parameterization(report['models'][spec],previous['models'][spec])
            cached[cohort] = prepared
    from lf_fit import ExecutionPermit, fit_unit
    permit = ExecutionPermit(authorization,sha(CONTRACT))
    before = read_json(PROGRESS/'implementation_before.json')
    # 实际运行保护基线采用运行开始时状态，不将作者期间的非统计编辑误作本轮变更。
    before.update(git_head=provenance()['git_commit'],git_diff_sha256=provenance()['git_diff_sha256'],
                  protected_hashes={n:sha(repo_path(n)) for n in before['protected_hashes'] if repo_path(n).is_file()})
    output = RUNS/run_id
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'execution_authorization.json',authorization)
    write_json(output/'protected_assets_before.json',before)
    rows, units, diagnostics = [], [], []
    def checkpoint(status):
        write_json(output/'run_manifest.json',{'stage':2,'status':status,'units':units,
            'contract_sha256':sha(CONTRACT),'authorization_sha256':sha(authorization_path),
            'inputs':contract['inputs'],'code':contract['code'],'model_identities':6,'automatic_retry':False,
            'meta_calls':0,'stage3_complete':False,'outputs':{rel(p):sha(p) for p in sorted(output.rglob('*'))
                if p.is_file() and p.name != 'run_manifest.json'},**provenance()},
            replace_owned=(output/'run_manifest.json').exists())
    checkpoint('WORKING_UNFROZEN')
    for cohort in COHORTS:
        post, models, report, primary_beta, primary_cov, primary_diag = cached.pop(cohort)
        primary = core_estimate(primary_beta,primary_cov,models['PO01']['mapping']['original_columns'])
        rows.append({'cohort':cohort,'spec_id':'Primary','participants':report['anchor']['participants'],
            'fi_observations':report['anchor']['primary_rows'],**primary,'status':'FROZEN_REFERENCE',
            'SE_ratio_to_primary':1.,'estimate_difference_from_primary':0.,
            'interpretation':interpretation('Primary')['core_estimand'],'source_kind':'FROZEN_PRIMARY_REFERENCE'})
        for spec in SPECS:
            print(f'START {cohort}/{spec}',flush=True)
            folder = output/spec/cohort
            o = fit_unit(post,models[spec],report,primary_diag,cohort,spec,folder,permit)
            write_json(folder/'file_validation.json',validate_saved(folder))
            core = o['core_estimate'] or {k:None for k in ('estimate','se','ci_lower','ci_upper')}
            rows.append({'cohort':cohort,'spec_id':spec,'participants':report['anchor']['participants'],
                'fi_observations':len(post),**core,'status':o['status'],'source_kind':'NEW_LMM_FIT',
                'interpretation':interpretation(spec)['core_estimand'],
                'SE_ratio_to_primary':core['se']/primary['se'] if core['se'] else None,
                'estimate_difference_from_primary':core['estimate']-primary['estimate'] if core['estimate'] is not None else None})
            units.append({'cohort':cohort,'spec_id':spec,'status':o['status'],'path':rel(folder)})
            diagnostics.append({'cohort':cohort,'spec_id':spec,**read_json(folder/'diagnostics.json')})
            checkpoint('WORKING_UNFROZEN')
            print(f"FINISH {cohort}/{spec}: {o['status']}",flush=True)
    write_csv(output/'cohort_estimates.csv',rows)
    write_json(output/'model_diagnostics.json',diagnostics)
    write_json(output/'protected_assets_validation.json',check_protected(before))
    checkpoint('STAGE2_COMPLETE_AWAITING_STAGE3')
    return {'status':'STAGE2_COMPLETE_AWAITING_STAGE3','units':units,'source_kind':'REAL_MODEL_EXECUTION','release_eligible':False}


def main():
    parser = argparse.ArgumentParser(description='Round1C Stage0–2；真实执行需要独立六模型授权')
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('preflight-lock'); p.add_argument('--audit-id',required=True)
    commands.add_parser('verify')
    p = commands.add_parser('fit'); p.add_argument('--run-id',required=True)
    p.add_argument('--authorization',type=Path)
    args = parser.parse_args()
    if args.command == 'preflight-lock':
        value = preflight_lock(args.audit_id)
    elif args.command == 'verify':
        verify_lock(); value = {'status':'LOCK_AND_SOURCES_VALID','model_fit_calls':0}
    else:
        value = fit(args.run_id,args.authorization)
    print(json.dumps(clean(value),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
