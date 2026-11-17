"""Round1C发布独立只读验证；包内来源与外部来源分别核验。"""
from pathlib import Path
import csv
import numpy as np
from scipy.stats import norm
from lf_common import ROOT,read_json,sha,require,TERM,Z95

RELEASE_ID='landmark_fi_round1c_v1'
COHORTS=('HRS','ELSA','KLoSA')
SPECS=('Primary','PO01','LM01')


def safe(base,name):
    base=Path(base).resolve();path=(base/name).resolve()
    require(path.is_relative_to(base),'发布路径越界')
    return path


def core_from_arrays(beta,cov,names):
    j=list(names).index(TERM);b=float(beta[j]);v=float(cov[j,j])
    require(np.isfinite([b,v]).all() and v>0,'正式结果核心SE非法')
    se=float(np.sqrt(v))
    return dict(estimate=b,se=se,ci_lower=b-Z95*se,ci_upper=b+Z95*se,p_value=float(2*norm.sf(abs(b/se))))


def check_exception_lineage(original,accepted,exception,approval,independent):
    require(original['status']=='HOLD_NOT_ESTIMABLE' and original['core_estimate'] is None,'原HRS/PO01 HOLD被回写')
    require(exception['disposition']=='AUTHOR_ACCEPTED_NUMERICAL_EXCEPTION' and exception['cohort']=='HRS' and
            exception['specification']=='PO01' and exception['accepted_for_stage3'] is True,'缺少HRS数值例外')
    require(accepted['original_stage2_status']=='HOLD_NOT_ESTIMABLE' and accepted['accepted_status']=='PASS_WITH_CAVEAT',
            'HRS发布接受状态或历史身份错误')
    require(accepted['original_double_optimizer_convergence'] is False and
            accepted['original_SciPy_stop_reason']=='UNRECOVERABLE_FROM_RETAINED_LOGS','优化限制丢失')
    require(approval['author_decision']=='QA_APPROVAL_WITH_CAVEATS' and approval['HRS_PO01_limited_result_accepted'] is True,
            '缺少本轮具体结果接受')
    require(independent['numerical']['numerical_status']=='QA_PASS_WITH_CAVEAT_RECOMMENDED','独立数值例外未通过')


def validate(release,mode='release-only',payload_only=False):
    release=Path(release).resolve();require(mode in ('release-only','full'),'未知验证模式')
    m=read_json(release/'release_manifest.json');require(m['release_id']==RELEASE_ID,'发布身份不符')
    listed=set()
    for record in m['inventory']:
        require(record['path'] not in listed,'清单路径重复');listed.add(record['path'])
        path=safe(release,record['path'])
        require(path.is_file() and sha(path)==record['sha256'],'包内资产SHA不符：'+record['path'])
    expected=listed|{'release_manifest.json'}
    if not payload_only:expected|={'release_validation.json','freeze.json'}
    actual={p.relative_to(release).as_posix() for p in release.rglob('*') if p.is_file()}
    require(actual==expected,'发布包含未登记或缺失文件')
    approval=read_json(safe(release,m['author_approval']['path']))
    require(sha(safe(release,m['author_approval']['path']))==m['author_approval']['sha256'],'批准身份不符')
    require(approval['author_decision']=='QA_APPROVAL_WITH_CAVEATS' and
            approval['freeze_authorization']=='GRANTED_FOR_LANDMARK_FI_ROUND1C_V1','未获本轮冻结批准')
    require(approval['scientific_models_approved']==6 and approval['stage5_candidate_table_authorized'] is True,'批准范围不符')
    with (release/'release_analysis_asset_index.csv').open(encoding='utf-8',newline='') as f:index=list(csv.DictReader(f))
    with (release/'Comparisons/cohort_estimates.csv').open(encoding='utf-8',newline='') as f:summary=list(csv.DictReader(f))
    identities={(c,s) for c in COHORTS for s in SPECS}
    require(len(index)==len(summary)==9 and {(r['cohort'],r['specification']) for r in index}==identities and
            {(r['cohort'],r['specification']) for r in summary}==identities,'必须为九个唯一队列/规格条目')
    primary=read_json(release/'Primary Reference/primary_lmm_result_object.json')
    comparisons=read_json(release/'QA/sequential_comparison.json')
    qa=read_json(release/'QA/independent_validation.json')
    require(qa['status']=='QA_APPROVAL_WITH_CAVEATS_RECOMMENDED' and len(qa['units'])==6,'Stage3独立QA不完整')
    checked=0
    for row in index:
        cohort,spec=row['cohort'],row['specification'];cell=next(r for r in summary if (r['cohort'],r['specification'])==(cohort,spec))
        reference=next(r for r in comparisons[cohort]['models'] if r['specification']==spec)
        require(int(cell['participants'])==reference['Participants'] and int(cell['FI_observations'])==reference['FI_observations'],'样本锚点不一致')
        if spec=='Primary':
            rows=[r for r in primary['fixed_effects'] if r['cohort']==cohort];names=[r['term'] for r in rows]
            beta=np.array([r['estimate'] for r in rows]);pairs={(r['term_i'],r['term_j']):r['covariance']
                for r in primary['fixed_effect_covariance'] if r['cohort']==cohort}
            cov=np.array([[pairs[(a,b)] for b in names] for a in names]);core=core_from_arrays(beta,cov,names)
            require(row['accepted_status']=='FROZEN_REFERENCE','Primary不得重新批准为新模型')
        else:
            folder=safe(release,row['result_folder'])
            original=read_json(folder/'result_object.json');accepted=read_json(folder/'accepted_result.json')
            with np.load(folder/'model_parameters.npz',allow_pickle=False) as a:
                names=a['fixed_effect_names'].tolist();core=core_from_arrays(a['fixed_effects'],a['fixed_effect_covariance'],names)
            require(original['cohort']==cohort and original['spec_id']==spec,'模型身份不符')
            require(accepted['accepted_status']=='PASS_WITH_CAVEAT' and accepted['author_result_approval']=='QA_APPROVAL_WITH_CAVEATS',
                    '活动模型未具体批准')
            require(accepted['original_stage2_status']==original['status'],'发布覆盖历史状态')
            independent=next(r for r in qa['units'] if (r['cohort'],r['specification'])==(cohort,spec))
            require(independent['status']=='PASS_WITH_CAVEAT_RECOMMENDED','Stage3存在未解决HOLD')
            if (cohort,spec)==('HRS','PO01'):
                exception=read_json(release/'Evidence/Numerical Exception/acceptance_v1/author_numeric_exception_approval.json')
                check_exception_lineage(original,accepted,exception,approval,independent)
            else:require(original['status']=='PASS' and independent['numerical']['numerical_status']=='PASS','普通活动模型数值未通过')
            for key in core:
                np.testing.assert_allclose(core[key],accepted['core_estimate'][key],rtol=1e-12,atol=1e-14)
            if cohort=='KLoSA':require('Q1_effect_at_23_over_12' in names and len(names)==(41 if spec=='PO01' else 43),'KLoSA基底错误')
        for key in core:
            np.testing.assert_allclose(float(cell[key]),core[key],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(reference[key],core[key],rtol=1e-12,atol=1e-14)
        if spec!='Primary':
            anchor=read_json(safe(release,row['result_folder'])/'sample_anchor.json')
            require(anchor['rows']==int(cell['FI_observations']) and anchor['participants']==int(cell['participants']),'保存样本与汇总冲突')
        checked+=1
    for cohort in COHORTS:
        anchors=[read_json(release/'Model Results'/cohort/s/'sample_anchor.json') for s in ('PO01','LM01')]
        require(anchors[0]==anchors[1],'PO01/LM01同行集丢失')
    external=0
    if mode=='full':
        for group in ['external_statistical_sources','external_execution_sources']:
            for name,digest in m[group].items():
                path=safe(ROOT,name);require(path.is_file() and sha(path)==digest,'外部来源变化：'+name);external+=1
    if not payload_only:
        freeze=read_json(release/'freeze.json')
        require(freeze['release_id']==RELEASE_ID and freeze['status']=='FROZEN_WITH_CAVEATS','冻结身份不符')
        require(freeze['author_result_approval']=='QA_APPROVAL_WITH_CAVEATS','冻结结果接受状态不符')
        require(freeze['release_manifest_sha256']==sha(release/'release_manifest.json') and
                freeze['release_validation_sha256']==sha(release/'release_validation.json'),'冻结依赖SHA不符')
    return {'status':'PASS','mode':mode,'inventory_files':len(listed),'analysis_rows_verified':checked,
            'external_files_checked':external,'HRS_original_HOLD_retained':True,'HRS_author_exception_and_result_approval_verified':True,
            'new_model_fits':0,'meta_calls':0,'release_manifest_sha256':sha(release/'release_manifest.json')}


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('release',type=Path)
    parser.add_argument('--mode',choices=['full','release-only'],default='release-only')
    args=parser.parse_args();print(json.dumps(validate(args.release,args.mode),ensure_ascii=False,indent=2))
