"""Round1E作者批准后复制冻结；接受覆盖层不改写原HOLD或PENDING历史。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import csv
import shutil
import xml.etree.ElementTree as ET
from validate_release import RELEASE_ID,validate,reject_existing,require
from sp_contract import ROOT,CONTRACT,verify_contract,read_json,write_json,sha,rel
from sp_audit import verify_records
from common import provenance,check_protected


def main():
    process=ROOT/'06_project_progress/round1e_shock_period_v1/Stage4/freeze_v1'
    final=ROOT/'03_output/3.3 Robustness of the Main Findings/releases'/RELEASE_ID
    staging=final.with_name('.'+RELEASE_ID+'_staging');reject_existing(final,staging)
    contract=verify_contract();approval=read_json(process/'author_result_and_freeze_approval.json')
    qa_process=ROOT/'06_project_progress/round1e_shock_period_v1/Stage3/qa_v1';qm=read_json(qa_process/'stage3_manifest.json')
    require(sha(qa_process/'stage3_manifest.json')==approval['approved_qa_manifest_sha256'],'批准的QA manifest变化')
    require(qm['scientific_recommendation']=='QA_APPROVAL_WITH_CAVEATS_RECOMMENDED','QA未建议附限定接受')
    require(qm['technical_statuses']=={'HRS':'PASS_WITH_AUTHORIZED_NUMERICAL_EXCEPTION','ELSA':'PASS','KLoSA':'PASS'},'未解决HOLD禁止完整冻结')
    for key in ('sources','outputs','code'):verify_records(qm[key])
    tests=ET.parse(process/'necessary_tests.xml').getroot();suites=tests.findall('.//testsuite')
    require(suites and all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped')),'发布必要测试未通过')
    require(approval['author_decision']=='QA_APPROVAL_WITH_CAVEATS' and approval['release_id']==RELEASE_ID,'没有结果接受')
    require(approval['contract_sha256']==sha(CONTRACT),'批准合同身份错误')
    check_protected(read_json(process/'protected_assets_before.json'))
    staging.mkdir(parents=True,exist_ok=False);inventory=[];external={}
    def register(path,target,role,source=None):
        inventory.append({'path':target,'sha256':sha(path),'bytes':path.stat().st_size,'role':role,
            **({'source_path':rel(source)} if source else {})})
    def copy(source,target,role):
        source=Path(source);destination=staging/target
        require(source.is_file() and not destination.exists(),'缺少来源或试图覆盖')
        require(source.suffix.lower() not in ('.parquet','.dta','.rds','.pkl'),'禁止复制微观数据')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        require(sha(source)==sha(destination),'复制后SHA不匹配')
        register(destination,target,role,source);external[rel(source)]=sha(source)
    def tree(source,target,role):
        for p in sorted(Path(source).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
                copy(p,target+'/'+p.relative_to(source).as_posix(),role)
    run=ROOT/'03_output/3.3 Robustness of the Main Findings/Shock Calendar Period/runs/round1e_stage2_20261015_01'
    original=read_json(run/'execution_manifest.json');verify_records(original['outputs'])
    qa=ROOT/'03_output/3.3 Robustness of the Main Findings/Shock Calendar Period/Stage3/qa_v1'
    metrics=read_json(qa/'review_metrics.json')
    for c in ('HRS','ELSA','KLoSA'):
        source=run/c/'SP01';target=f'Model Results/{c}/SP01';tree(source,target,'ORIGINAL_STAGE2_BYTES_UNCHANGED')
        original_result=read_json(source/'result_object.json');numerical=read_json(qa/c/'independent_validation.json')
        accepted={'schema':'round1e_accepted_result_overlay_v1','cohort':c,'specification':'SP01','accepted_status':'PASS_WITH_CAVEAT',
            'author_result_approval':'QA_APPROVAL_WITH_CAVEATS','original_stage2_status':original_result['status'],
            'original_run_id':run.name,'original_result_sha256':sha(source/'result_object.json'),
            'original_parameters_sha256':sha(source/'model_parameters.npz'),'independent_qa_id':'qa_v1',
            'independent_numerical_status':numerical['technical_status'],'core_estimate':metrics['cohorts'][c]['comparison']['SP01'],
            'author_approval_sha256':sha(process/'author_result_and_freeze_approval.json'),
            'original_double_optimizer_convergence':all(original_result['convergence'].values()),
            'numerical_exception':'AUTHOR_ACCEPTED_NUMERICAL_EXCEPTION' if c=='HRS' else None,
            'accepted_caveats':approval['accepted_caveats'],'participant_records_saved':False,
            'interpretation':contract['scientific_specification']['interpretation']}
        p=staging/target/'accepted_result.json';write_json(p,accepted);register(p,target+'/accepted_result.json','AUTHOR_APPROVED_ACCEPTANCE_LAYER')
    tree(qa,'QA','INDEPENDENT_STAGE3_NUMERIC_AND_AGGREGATE_DIAGNOSTICS')
    tree(qa_process,'Evidence/Stage3','STAGE3_HISTORICAL_REVIEW_AND_MANIFEST')
    for p in run.iterdir():
        if p.is_file():copy(p,'Evidence/Stage2/Run/'+p.name,'ORIGINAL_STAGE2_RUNNING_LINEAGE')
    tree(ROOT/'06_project_progress/round1e_shock_period_v1/Stage2/round1e_stage2_20261015_01','Evidence/Stage2/Process','STAGE2_EXECUTION_EVIDENCE')
    tree(ROOT/'06_project_progress/round1e_shock_period_v1/Stage2_Exception_Resolution/HRS_SP01','Evidence/Numerical Exception','FULL_HRS_EXCEPTION_HISTORY')
    tree(ROOT/'06_project_progress/round1e_shock_period_v1/Stage1','Evidence/Stage1','ACTIVE_AND_SUPERSEDED_STAGE1_EVIDENCE')
    tree(ROOT/'06_project_progress/round1e_shock_period_v1/Stage0/round1e_stage0_20261014_01','Evidence/Stage0/Process','SOURCE_AUDIT_HISTORY')
    tree(ROOT/'03_output/3.3 Robustness of the Main Findings/Shock Calendar Period/Audit/round1e_stage0_20261014_01','Evidence/Stage0/Audit','AGGREGATE_CALENDAR_SOURCE_AUDIT')
    for p in CONTRACT.parent.glob('*.json'):copy(p,'Evidence/Contracts/'+p.name,'LOCKED_OR_HISTORICAL_CONTRACT')
    codes={**contract['code'],**qm['code']}
    for name,digest in codes.items():
        p=ROOT/name;require(sha(p)==digest,'执行代码变化');copy(p,'Evidence/Code/'+name,'ACTUAL_MODEL_AND_QA_CODE')
    for p in Path(__file__).parent.glob('*.py'):copy(p,'Evidence/Code/Stage4/'+p.name,'FREEZE_AND_INDEPENDENT_VALIDATION_CODE')
    primary=ROOT/'03_output/3.2 Wealth Recovery and Subsequent Frailty Progression'
    for name in ('primary_lmm_result_object.json','primary_lmm_fixed_effects.csv','primary_lmm_fixed_effect_covariance.csv',
        'primary_lmm_standardization_constants.csv','primary_lmm_diagnostics.json'):
        copy(primary/'Primary Models'/name,'Primary Reference/'+name,'FROZEN_PRIMARY_REFERENCE')
    for name in ('section_3_2_freeze.json','section_3_2_manifest.json'):copy(primary/name,'Primary Reference/'+name,'PRIMARY_FREEZE_IDENTITY')
    copy(process/'author_result_and_freeze_approval.json','Evidence/Approvals/author_result_and_freeze_approval.json','AUTHOR_RESULT_AND_FREEZE_APPROVAL')
    copy(process/'necessary_tests.xml','Evidence/Stage4/necessary_tests.xml','RELEASE_BOUNDARY_TEST_EVIDENCE')
    rows=[];index=[]
    for c in ('HRS','ELSA','KLoSA'):
        comp=metrics['cohorts'][c]['comparison'];anchor=read_json(run/c/'SP01/sample_design.json')['anchor']
        for s in ('Primary','SP01'):
            row={'cohort':c,'specification':s,'participants':anchor['participants'],'FI_observations':anchor['rows'],
                **comp[s],'units':'FI units/year','accepted_status':'FROZEN_REFERENCE' if s=='Primary' else 'PASS_WITH_CAVEAT',
                'original_stage2_status':'FROZEN_REFERENCE' if s=='Primary' else metrics['cohorts'][c]['original_stage2_status'],
                'numerical_exception':'AUTHOR_ACCEPTED_NUMERICAL_EXCEPTION' if (c,s)==('HRS','SP01') else ''}
            rows.append(row);index.append({'cohort':c,'specification':s,'source_kind':'FROZEN_PRIMARY_REFERENCE' if s=='Primary' else 'APPROVED_SP01_RESULT',
                'result_folder':'Primary Reference' if s=='Primary' else f'Model Results/{c}/SP01','accepted_status':row['accepted_status'],
                'original_stage2_status':row['original_stage2_status']})
    for name,data in [('Comparisons/cohort_estimates.csv',rows),('release_analysis_asset_index.csv',index)]:
        p=staging/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        register(p,name,'ACCEPTED_RELEASE_INDEX_OR_FULL_PRECISION_COMPARISON')
    text='''# shock_period_round1e_v1

状态：FROZEN_WITH_CAVEATS。作者按QA_APPROVAL_WITH_CAVEATS接受三个SP01及其限制。
正式入口：freeze.json、release_manifest.json、release_analysis_asset_index.csv及Comparisons/cohort_estimates.csv。
Model Results中的原Stage2文件逐字节复制；accepted_result.json为独立接受层。HRS原HOLD、空core_estimate和BFGS precision loss终止历史均保持。不得把接受结果伪装为原始双优化器PASS。
Evidence保存原合同、Stage0/1/2/3、独立数值例外及作者批准；历史PENDING/NOT_GRANTED字段只表示当时状态，当前批准权威是Evidence/Approvals中的本次作者决定。
Primary仅作为冻结参照，没有重新拟合或替换。全部三个最晚波次只有landmark加一次后续FI；晚期参数CI较宽，敏感性尚未充分排除。Gaussian残差偏态/重尾与模型推断限制保留。
SP01假定跨波次共同恢复年度速度差，不估计时期特异恢复关联或金融危机因果效应；本轮不消除Round1B随机斜率与非线性敏感性。
未保存新的微观Panel或个人残差/BLUP。细分聚合支持仅为内部审阅，不是公开投稿细表。
Stage5只读本freeze，在包外生成英文六行Primary/SP01候选表和时期支持摘要；不写回本包，不使用latest或工作目录回退，不修改论文。
manifest→validation→freeze为单向SHA依赖；validate_release.py支持release-only与full外部来源核验。
'''
    p=staging/'README.md';p.write_text(text,encoding='utf-8',newline='\n');register(p,'README.md','RELEASE_AUTHORITY_AND_CAVEATS')
    manifest={'schema':'round1e_release_manifest_v1','release_id':RELEASE_ID,'status':'AUTHOR_APPROVED_READY_TO_FREEZE',
        'inventory':inventory,'author_approval_sha256':sha(process/'author_result_and_freeze_approval.json'),
        'external_statistical_sources':qm['sources'],'external_execution_sources':external,
        'author_result_approval':'QA_APPROVAL_WITH_CAVEATS','accepted_status':'PASS_WITH_CAVEAT',
        'original_status_counts':{'PASS':2,'HOLD_NOT_ESTIMABLE':1},'numerical_exception_accepted_models':['HRS/SP01'],
        'accepted_caveats':approval['accepted_caveats'],'stage5_candidate_table_authorized':True,
        'new_model_fits':0,'primary_replaced':False,'submission_ready':False,**provenance()}
    write_json(staging/'release_manifest.json',manifest)
    write_json(staging/'release_validation.json',validate(staging,'full',True))
    write_json(staging/'freeze.json',{'release_id':RELEASE_ID,'status':'FROZEN_WITH_CAVEATS',
        'release_manifest_sha256':sha(staging/'release_manifest.json'),'release_validation_sha256':sha(staging/'release_validation.json'),
        'author_result_approval':'QA_APPROVAL_WITH_CAVEATS','three_SP01_models_accepted':'PASS_WITH_CAVEAT',
        'original_HRS_SP01_HOLD_retained':True,'primary_replaced':False,'submission_ready':False,**provenance()})
    validate(staging,'release-only');validate(staging,'full')
    require(staging.resolve().parent==final.resolve().parent and final.name==RELEASE_ID and not final.exists(),'最终移动目标未验证')
    staging.rename(final)
    validation=validate(final,'full');protected=check_protected(read_json(process/'protected_assets_before.json'))
    write_json(process/'freeze_delivery_manifest.json',{'status':'FROZEN_WITH_CAVEATS','release_path':rel(final),
        'freeze_sha256':sha(final/'freeze.json'),'release_manifest_sha256':sha(final/'release_manifest.json'),
        'validation':validation,'protected_check':protected,'new_model_fits':0,'manuscript_changed':False,
        'stage5_completed':False,**provenance()})
    print({'release':str(final),'status':'FROZEN_WITH_CAVEATS','inventory':len(inventory),'validation':'PASS','protected_files':protected['files_checked']})


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8');main()
