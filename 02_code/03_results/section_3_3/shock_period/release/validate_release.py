"""Round1E发布的独立只读验证：包内数值与历史、外部来源分开核验。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import csv
import numpy as np
from scipy.stats import norm
from sp_contract import ROOT,read_json,sha

RELEASE_ID='shock_period_round1e_v1'
COHORTS=('HRS','ELSA','KLoSA')


def require(condition,message):
    if not condition:raise ValueError(message)


def safe(base,name):
    base=Path(base).resolve();path=(base/name).resolve()
    require(path.is_relative_to(base),'发布包路径越界')
    return path


def reject_existing(final,staging):
    require(not Path(final).exists() and not Path(staging).exists(),'拒绝覆盖发布或暂存目录')


def core(beta,cov,names):
    j=list(names).index('recovery:time_years');b=float(beta[j]);v=float(cov[j,j])
    require(np.isfinite([b,v]).all() and v>0,'核心推断不合法')
    se=float(np.sqrt(v));z=norm.ppf(.975)
    return {'estimate':b,'se':se,'ci_lower':b-z*se,'ci_upper':b+z*se,'p_value':float(2*norm.sf(abs(b/se)))}


def exception_lineage(original,accepted,approval,exception,qa,bfgs,lbfgs):
    require(original['status']=='HOLD_NOT_ESTIMABLE' and original['core_estimate'] is None,'原HRS HOLD被覆盖')
    require(original['failure_reason_codes']==['OPTIMIZER_NOT_CONVERGED'],'HRS存在未批准的其他失败')
    require(accepted['accepted_status']=='PASS_WITH_CAVEAT' and accepted['original_stage2_status']=='HOLD_NOT_ESTIMABLE','接受层丢失限定或历史')
    require(accepted['original_double_optimizer_convergence'] is False,'把原始单收敛伪装双收敛')
    require(approval['author_decision']=='QA_APPROVAL_WITH_CAVEATS' and approval['HRS_SP01_numerical_exception_accepted'],'未批准HRS具体结果')
    require(exception['disposition']=='AUTHOR_ACCEPTED_NUMERICAL_EXCEPTION' and exception['cohort']=='HRS' and
        exception['specification']=='SP01' and exception['accepted_for_stage3'],'缺少正确单元的数值例外')
    require(qa['technical_status']=='PASS_WITH_AUTHORIZED_NUMERICAL_EXCEPTION' and qa['numeric_checks_passed'],'独立QA未通过')
    require(not bfgs['converged'] and bfgs['terminal']['status']==2 and 'precision loss' in bfgs['terminal']['message'],'BFGS历史终止状态变化')
    require(lbfgs['converged'] and lbfgs['terminal']['status']==0 and
        lbfgs['terminal']['task']=='CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH','L-BFGS终止条件变化')


def validate(base,mode='release-only',payload_only=False):
    base=Path(base).resolve();require(mode in ('release-only','full'),'未知验证模式')
    manifest=read_json(base/'release_manifest.json');require(manifest['release_id']==RELEASE_ID,'发布身份错误')
    listed=set()
    for r in manifest['inventory']:
        require(r['path'] not in listed,'清单路径重复');listed.add(r['path'])
        p=safe(base,r['path']);require(p.is_file() and sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],'包内哈希或大小变化：'+r['path'])
        require(p.suffix.lower() not in ('.dta','.parquet','.rds','.pkl'),'发布不得包含微观Panel或拟合对象')
    expected=listed|{'release_manifest.json'}
    if not payload_only:expected|={'release_validation.json','freeze.json'}
    require({p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==expected,'包含未登记或缺失文件')
    approval=read_json(base/'Evidence/Approvals/author_result_and_freeze_approval.json')
    require(sha(base/'Evidence/Approvals/author_result_and_freeze_approval.json')==manifest['author_approval_sha256'],'批准SHA变化')
    require(approval['author_decision']=='QA_APPROVAL_WITH_CAVEATS' and
        approval['freeze_authorization']=='GRANTED_FOR_SHOCK_PERIOD_ROUND1E_V1' and
        approval['scope']==[c+'/SP01' for c in COHORTS],'未获三个单元冻结批准')
    require(approval['stage5_candidate_table_authorized'] and approval['original_HRS_SP01_HOLD_retained'],'呈现或历史批准不完整')
    require(sha(base/'Evidence/Stage3/stage3_manifest.json')==approval['approved_qa_manifest_sha256'],'批准的QA身份不符')
    require(sha(base/'Evidence/Stage3/scientific_interpretation.md')==approval['approved_scientific_interpretation_sha256'],'批准科学审阅身份不符')
    require(sha(base/'Evidence/Contracts/shock_period_round1e_v1_contract.json')==approval['contract_sha256'],'合同身份不符')
    require(sha(base/'Evidence/Stage1/round1e_stage1_20261015_02/preflight_manifest.json')==approval['preflight_manifest_sha256'],'Preflight身份不符')
    with (base/'release_analysis_asset_index.csv').open(encoding='utf-8',newline='') as f:index=list(csv.DictReader(f))
    with (base/'Comparisons/cohort_estimates.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    identities={(c,s) for c in COHORTS for s in ('Primary','SP01')}
    require(len(index)==len(rows)==6 and {(r['cohort'],r['specification']) for r in rows}==identities and
        {(r['cohort'],r['specification']) for r in index}==identities,'必须六行唯一Primary/SP01')
    primary=read_json(base/'Primary Reference/primary_lmm_result_object.json')
    metrics=read_json(base/'QA/review_metrics.json')
    for r in rows:
        c,s=r['cohort'],r['specification'];q=read_json(base/'QA'/c/'independent_validation.json')
        require(q['numeric_checks_passed'] and all(a['passed'] for a in q['checks']),'存在未解决独立QA失败')
        reference=q['Primary_SP01_comparison'][s]
        if s=='Primary':
            fixed=[v for v in primary['fixed_effects'] if v['cohort']==c];names=[v['term'] for v in fixed]
            pairs={(v['term_i'],v['term_j']):v['covariance'] for v in primary['fixed_effect_covariance'] if v['cohort']==c}
            estimate=core(np.array([v['estimate'] for v in fixed]),np.array([[pairs[a,b] for b in names] for a in names]),names)
            require(r['accepted_status']=='FROZEN_REFERENCE','Primary不得被重新批准为新拟合')
        else:
            unit=base/'Model Results'/c/'SP01';original=read_json(unit/'result_object.json');accepted=read_json(unit/'accepted_result.json')
            a=np.load(unit/'model_parameters.npz',allow_pickle=False);names=a['fixed_effect_names'].tolist()
            estimate=core(a['fixed_effects'],a['fixed_effect_covariance'],names)
            require(len(names)=={'HRS':64,'ELSA':54,'KLoSA':52}[c],'执行列数改变')
            require(sha(unit/'result_object.json')==accepted['original_result_sha256'] and
                sha(unit/'model_parameters.npz')==accepted['original_parameters_sha256'],'接受层原模型SHA不符')
            require(accepted['original_stage2_status']==original['status'] and accepted['accepted_status']=='PASS_WITH_CAVEAT','接受层覆盖历史或丢失限定')
            require(accepted['author_result_approval']=='QA_APPROVAL_WITH_CAVEATS' and
                accepted['author_approval_sha256']==manifest['author_approval_sha256'],'模型具体接受绑定不符')
            require(q['sample_design']['anchor']==original['sample'],'原始样本锚点不一致')
            if c=='HRS':
                exception=read_json(base/'Evidence/Numerical Exception/author_numerical_exception_approval_v1.json')
                exception_lineage(original,accepted,approval,exception,q,
                    read_json(unit/'bfgs_optimization.json'),read_json(unit/'lbfgs_optimization.json'))
                require(sha(unit/'lbfgs_candidate.npz')==exception['candidate_sha256'],'例外批准候选SHA不符')
                require(sha(base/'Evidence/Numerical Exception/check_v1/check_manifest.json')==exception['independent_check_manifest_sha256'],'独立核查链不符')
            else:require(original['status']=='PASS' and q['technical_status']=='PASS','普通单元状态未通过')
            for k in estimate:np.testing.assert_allclose(estimate[k],accepted['core_estimate'][k],rtol=1e-12,atol=1e-14)
        for k in estimate:
            np.testing.assert_allclose(float(r[k]),estimate[k],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(reference[k],estimate[k],rtol=1e-12,atol=1e-14)
        anchor=q['sample_design']['anchor']
        require(int(r['participants'])==anchor['participants'] and int(r['FI_observations'])==anchor['rows'],'样本汇总错误')
        require(metrics['cohorts'][c]['technical_status']==q['technical_status'],'QA索引状态矛盾')
    elsa=read_json(base/'QA/ELSA/independent_validation.json')['ELSA_time_verification']
    require(elsa['W10_2021_2023_rows']==elsa['participants']==272 and elsa['years']=={'2021':23,'2022':229,'2023':20},'ELSA晚期FI未保留')
    external=0
    if mode=='full':
        for group in ('external_statistical_sources','external_execution_sources'):
            for name,digest in manifest[group].items():
                p=(ROOT/name).resolve();require(p.is_file() and sha(p)==digest,'外部来源哈希变化：'+name);external+=1
    if not payload_only:
        freeze=read_json(base/'freeze.json');validation=read_json(base/'release_validation.json')
        require(freeze['release_id']==RELEASE_ID and freeze['status']=='FROZEN_WITH_CAVEATS','冻结身份错误')
        require(freeze['release_manifest_sha256']==sha(base/'release_manifest.json') and
            freeze['release_validation_sha256']==sha(base/'release_validation.json'),'冻结依赖哈希不符')
        require(validation['status']=='PASS' and freeze['author_result_approval']=='QA_APPROVAL_WITH_CAVEATS','冻结验收或批准无效')
    return {'status':'PASS','mode':mode,'inventory_files':len(listed),'analysis_rows_verified':6,
        'new_SP01_models_accepted':3,'external_files_checked':external,'HRS_original_HOLD_retained':True,
        'HRS_author_exception_and_scientific_approval_verified':True,'new_model_fits':0,
        'release_manifest_sha256':sha(base/'release_manifest.json')}


if __name__=='__main__':
    import argparse,json
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('release');p.add_argument('--mode',choices=['release-only','full'],default='release-only')
    a=p.parse_args();print(json.dumps(validate(a.release,a.mode),ensure_ascii=False,indent=2))
