"""Round1C作者批准后冻结；保留原HOLD与批准覆盖层，不重新拟合。"""
from pathlib import Path
import csv
import shutil
from lf_common import ROOT,CODE,PROGRESS,RUNS,CONTRACT,read_json,write_json,sha,rel,require,assert_hashes,check_protected,provenance
from run_landmark_fi import verify_lock
from validate_round1c_release import RELEASE_ID,validate

FINAL=ROOT/'03_output/3.3 Robustness of the Main Findings/releases'/RELEASE_ID
STAGING=FINAL.with_name('.'+RELEASE_ID+'_staging')
QA_ID='round1c_stage3_20261014_02'
EXPECTED_QA_SHA='b76c31ef63ceb9915e3a84246c0e007c9a4a8ec0b349a70158e3790ead651ab3'
HUMAN_INSTRUCTION='我赞成按 `QA_APPROVAL_WITH_CAVEATS` 接受Round 1C的结果，并授权Stage 4冻结。 冻结后再生成一张九行英文候选补充表。按照此前确定的工作顺序，暂时不修改论文正文'


def reject_existing(final,staging):
    require(not final.exists() and not staging.exists(),'拒绝覆盖已有发布或暂存目录')


def main():
    reject_existing(FINAL,STAGING)
    process=PROGRESS/'Stage4';require(not process.exists(),'拒绝覆盖Stage4记录')
    contract=verify_lock();qa_process=PROGRESS/'Stage3'/QA_ID
    qmanifest=read_json(qa_process/'stage3_manifest.json')
    require(sha(qa_process/'stage3_manifest.json')==EXPECTED_QA_SHA and
            qmanifest['recommendation']=='QA_APPROVAL_WITH_CAVEATS_RECOMMENDED','作者审阅的QA身份变化')
    for key in ['source_hashes','qa_code','outputs','reports']:assert_hashes(qmanifest[key])
    qa=ROOT/'03_output/3.3 Robustness of the Main Findings/Landmark FI Sensitivity/QA'/QA_ID
    data=read_json(qa/'independent_validation.json')
    require(len(data['units'])==6 and all(u['status']=='PASS_WITH_CAVEAT_RECOMMENDED' for u in data['units']),'QA有HOLD，不默认完整发布')
    run=RUNS/'round1c_stage2_20261014_01';original=read_json(run/'execution_manifest.json')
    assert_hashes(original['outputs']);assert_hashes(original['process_files'])
    protected=read_json(run/'protected_assets_before.json');check_protected(protected)
    approval={'schema':'round1c_author_result_and_freeze_approval_v1','author_decision':'QA_APPROVAL_WITH_CAVEATS',
       'freeze_authorization':'GRANTED_FOR_LANDMARK_FI_ROUND1C_V1','release_id':RELEASE_ID,
       'human_instruction':HUMAN_INSTRUCTION,'evidence_source':'Direct typed human instruction in the current Codex chat',
       'thread_id':'01a13693-5007-7713-a023-0bb0734ee0e3','scientific_models_approved':6,
       'scope':[f'{c}/{s}' for c in ['HRS','ELSA','KLoSA'] for s in ['PO01','LM01']],
       'approved_qa_id':QA_ID,'approved_qa_manifest_sha256':EXPECTED_QA_SHA,
       'approved_scientific_interpretation_sha256':sha(qa_process/'scientific_interpretation.md'),
       'contract_sha256':sha(CONTRACT),'preflight_manifest_sha256':contract['audit']['manifest_sha256'],
       'HRS_PO01_limited_result_accepted':True,'original_HRS_PO01_HOLD_retained':True,
       'stage5_candidate_table_authorized':True,'stage5_scope':'One English nine-row candidate supplementary table read only from this release; stop at author review',
       'manuscript_changes_authorized':False,'new_models_authorized':False,
       'accepted_caveats':['HRS/PO01 BFGS converged but L-BFGS did not; exact SciPy stop reason was not retained.',
           'Independent observed-information validation supports the candidate; it does not establish global optimality.',
           'Gaussian conditional residual skewness, heavy tails and occasional fitted FI outside [0,1] remain.',
           'ELSA has sparse distant follow-up; KLoSA shows heavy tails and long-follow-up influence risk.',
           'Fixed-covariance influence screening does not exclude truncated-follow-up or deletion sensitivity.',
           'LM01 is a post-landmark conditional association, not a causally identified recovery effect.',
           'PO01/LM01 intercept and recovery main effects extrapolate to time zero; no landmark level claims.',
           'Round1C does not resolve Round1B sensitivity to random slope/nonlinear time; no joint models fitted.'],
       **provenance()}
    process.mkdir(parents=True,exist_ok=False);write_json(process/'author_approval.json',approval)
    STAGING.mkdir(parents=True,exist_ok=False)
    inventory=[];external={}
    def register(path,target,role,source=None):
        inventory.append({'path':target,'sha256':sha(path),'bytes':path.stat().st_size,'role':role,
                          **({'source_path':rel(source)} if source is not None else {})})
    def copy(source,target,role):
        source=Path(source);destination=STAGING/target
        require(source.is_file() and not destination.exists(),'缺少来源或拒绝覆盖')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        require(sha(source)==sha(destination),'发布副本不一致')
        register(destination,target,role,source);external[rel(source)]=sha(source)
    for cohort in ['HRS','ELSA','KLoSA']:
        for spec in ['PO01','LM01']:
            source=run/spec/cohort;target=f'Model Results/{cohort}/{spec}'
            for path in sorted(source.iterdir()):
                if path.is_file():copy(path,target+'/'+path.name,'ORIGINAL_STAGE2_RESULT_UNCHANGED')
            unit=next(u for u in data['units'] if (u['cohort'],u['specification'])==(cohort,spec))
            special=(cohort,spec)==('HRS','PO01')
            accepted={'schema':'round1c_accepted_result_overlay_v1','cohort':cohort,'specification':spec,
               'accepted_status':'PASS_WITH_CAVEAT','author_result_approval':'QA_APPROVAL_WITH_CAVEATS',
               'original_stage2_status':unit['original_stage2_status'],'original_stage2_run_id':run.name,
               'original_result_sha256':sha(source/'result_object.json'),'original_parameters_sha256':sha(source/'model_parameters.npz'),
               'independent_qa_id':QA_ID,'independent_numerical_status':unit['numerical']['numerical_status'],
               'author_approval_sha256':sha(process/'author_approval.json'),'core_estimate':unit['numerical']['core'],
               'original_double_optimizer_convergence':unit['numerical']['ordinary_double_convergence_requirement_met'],
               'original_SciPy_stop_reason':'UNRECOVERABLE_FROM_RETAINED_LOGS' if special else 'NO_NUMERICAL_EXCEPTION',
               'numerical_exception':'AUTHOR_ACCEPTED_NUMERICAL_EXCEPTION' if special else None,
               'accepted_caveats':approval['accepted_caveats'],'participant_records_saved':False}
            path=STAGING/target/'accepted_result.json';write_json(path,accepted)
            register(path,target+'/accepted_result.json','AUTHOR_APPROVED_OVERLAY_DOES_NOT_REWRITE_HISTORY')
    for path in qa.rglob('*'):
        if path.is_file():copy(path,'QA/'+path.relative_to(qa).as_posix(),'INDEPENDENT_QA_SUMMARY')
    for path in qa_process.iterdir():
        if path.is_file():copy(path,'Evidence/Stage3/'+path.name,'STAGE3_REVIEW_OR_SOURCE_IDENTITY')
    for path in (PROGRESS/'Stage2'/run.name).iterdir():
        if path.is_file():copy(path,'Evidence/Stage2/'+path.name,'STAGE2_EXECUTION_LINEAGE')
    for name in ['execution_authorization.json','run_manifest.json','execution_manifest.json','protected_assets_validation.json']:
        copy(run/name,'Evidence/Original Execution/'+name,'ORIGINAL_EXECUTION_LINEAGE')
    for path in (PROGRESS/'Stage2_Exception_Resolution/HRS_PO01').rglob('*'):
        if path.is_file():copy(path,'Evidence/Numerical Exception/'+path.relative_to(PROGRESS/'Stage2_Exception_Resolution/HRS_PO01').as_posix(),'NUMERICAL_EXCEPTION_CHAIN')
    copy(CONTRACT,'Evidence/Contracts/'+CONTRACT.name,'LOCKED_CONTRACT')
    preflight=ROOT/contract['audit']['path']
    for path in preflight.rglob('*'):
        if path.is_file():copy(path,'Evidence/Preflight/'+path.relative_to(preflight).as_posix(),'PREFLIGHT_EVIDENCE')
    code={**contract['code'],**qmanifest['qa_code']}
    for name,digest in code.items():
        source=ROOT/name;require(sha(source)==digest,'实际源码身份变化')
        copy(source,'Evidence/Code/'+name,'MODEL_OR_QA_SOURCE')
    for name in ['freeze_round1c.py','validate_round1c_release.py','test_round1c_release.py']:
        copy(CODE/name,'Evidence/Code/Stage4/'+name,'PUBLISH_OR_VALIDATE_CODE')
    primary=ROOT/'03_output/3.2 Wealth Recovery and Subsequent Frailty Progression'
    for name in ['primary_lmm_result_object.json','primary_lmm_fixed_effects.csv','primary_lmm_fixed_effect_covariance.csv','primary_lmm_standardization_constants.csv','primary_lmm_diagnostics.json']:
        copy(primary/'Primary Models'/name,'Primary Reference/'+name,'FROZEN_PRIMARY_REFERENCE')
    for name in ['section_3_2_freeze.json','section_3_2_manifest.json']:
        copy(primary/name,'Primary Reference/'+name,'PRIMARY_FREEZE_REFERENCE')
    copy(process/'author_approval.json','Evidence/Approvals/author_approval.json','AUTHOR_RESULT_AND_FREEZE_APPROVAL')
    comparisons=read_json(qa/'sequential_comparison.json');summary=[];index=[]
    for cohort in ['HRS','ELSA','KLoSA']:
        for row in comparisons[cohort]['models']:
            spec=row['specification'];primary_row=spec=='Primary'
            entry={'cohort':cohort,'specification':spec,'participants':row['Participants'],'FI_observations':row['FI_observations'],
                   **{k:row[k] for k in ['estimate','se','ci_lower','ci_upper','p_value']},
                   'units':'FI units/year','accepted_status':'FROZEN_REFERENCE' if primary_row else 'PASS_WITH_CAVEAT',
                   'original_stage2_status':'FROZEN_REFERENCE' if primary_row else next(u['original_stage2_status'] for u in data['units'] if (u['cohort'],u['specification'])==(cohort,spec))}
            summary.append(entry)
            index.append({'cohort':cohort,'specification':spec,'source_kind':'FROZEN_PRIMARY_REFERENCE' if primary_row else 'NEW_LMM_FIT',
                          'result_folder':'Primary Reference' if primary_row else f'Model Results/{cohort}/{spec}',
                          'accepted_status':entry['accepted_status'],'original_stage2_status':entry['original_stage2_status']})
    def csv_write(target,rows):
        path=STAGING/target;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        register(path,target,'AUTHOR_APPROVED_RELEASE_INDEX_OR_SUMMARY')
    csv_write('Comparisons/cohort_estimates.csv',summary);csv_write('release_analysis_asset_index.csv',index)
    readme='''# landmark_fi_round1c_v1

作者按QA_APPROVAL_WITH_CAVEATS批准六个PO01/LM01结果；冻结状态FROZEN_WITH_CAVEATS。
只读入口为freeze.json、release_manifest.json及release_analysis_asset_index.csv。Primary只作为既有冻结参照。
Model Results保存原Stage2字节；accepted_result.json是新增批准覆盖层。HRS/PO01原HOLD、空core_estimate、L-BFGS未收敛及终止原因缺失均保留，绝不伪装普通PASS。
Evidence保存合同、活动Preflight、执行、数值例外、独立QA及本次作者批准；QA保留聚合诊断，不含新的微观Panel或逐人残差/BLUP。
LM01为post-landmark conditional association，不是已识别因果效应。post-only的时间0截距及恢复主效应不能解释为实测landmark健康水平差。
Gaussian残差、ELSA稀疏远期及KLoSA重尾/长随访影响限制保留。Round1C不消除Round1B的随机斜率及非线性敏感性。
没有新模型、meta或Primary重拟合。论文正文、既有Figure2/3与TableS6未修改。
Stage5候选表授权在包外执行，只读本release并停在作者审阅稿；本冻结包不含候选表，不会被后续制表写回。
manifest→validation→freeze为单向SHA依赖。validate_round1c_release.py支持release-only包内验证和full外部来源验证。
'''
    path=STAGING/'README.md';path.write_text(readme,encoding='utf-8');register(path,'README.md','RELEASE_README')
    manifest={'schema':'round1c_release_manifest_v1','release_id':RELEASE_ID,'status':'AUTHOR_APPROVED_READY_TO_FREEZE',
       'inventory':inventory,'author_approval':{'path':'Evidence/Approvals/author_approval.json','sha256':sha(process/'author_approval.json')},
       'external_statistical_sources':contract['inputs'],'external_execution_sources':external,
       'scientific_review':'QA_APPROVAL_WITH_CAVEATS','six_models_accepted':'PASS_WITH_CAVEAT',
       'original_status_counts':{'PASS':5,'HOLD_NOT_ESTIMABLE':1},'numerical_exception_accepted_models':['HRS/PO01'],
       'accepted_caveats':approval['accepted_caveats'],'primary_replaced':False,'submission_ready':False,
       'stage5_candidate_table_authorized':True,'new_model_fits':0,'meta_calls':0,**provenance()}
    write_json(STAGING/'release_manifest.json',manifest)
    write_json(STAGING/'release_validation.json',validate(STAGING,'full',True))
    write_json(STAGING/'freeze.json',{'release_id':RELEASE_ID,'status':'FROZEN_WITH_CAVEATS',
       'release_manifest_sha256':sha(STAGING/'release_manifest.json'),'release_validation_sha256':sha(STAGING/'release_validation.json'),
       'author_result_approval':'QA_APPROVAL_WITH_CAVEATS','six_models_accepted':'PASS_WITH_CAVEAT',
       'original_HRS_PO01_HOLD_retained':True,'primary_replaced':False,'submission_ready':False,**provenance()})
    validate(STAGING,'release-only');validate(STAGING,'full');STAGING.rename(FINAL)
    final_validation=validate(FINAL,'full');protected_validation=check_protected(protected)
    write_json(process/'freeze_delivery_manifest.json',{'status':'FROZEN_WITH_CAVEATS','release_path':rel(FINAL),
       'freeze_sha256':sha(FINAL/'freeze.json'),'release_manifest_sha256':sha(FINAL/'release_manifest.json'),
       'validation':final_validation,'protected_assets':protected_validation,'new_model_fits':0,'manuscript_changed':False,**provenance()})
    print({'release':str(FINAL),'status':'FROZEN_WITH_CAVEATS','inventory':len(inventory),'validation':'PASS'})


if __name__=='__main__':main()
